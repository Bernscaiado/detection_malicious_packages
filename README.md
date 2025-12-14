# Malicious Package Version Transition Detection

A lightweight, registry-driven early warning system for detecting suspicious package upgrades in npm and PyPI ecosystems using metadata-only analysis.

---

## Overview

This project builds an end-to-end pipeline to detect **malicious version transitions** in open-source ecosystems (PyPI, npm) using **registry metadata** and **version-to-version delta features** only.

The v4 "all transitions" model is trained on every version-to-version jump in the delta table and reaches **0.81 accuracy**. For benign transitions, precision/recall are 0.86/0.84; for malicious transitions they are 0.73/0.77. The confusion matrix [[76 15], [12 41]] means 76/91 benign transitions are correctly kept benign, 41/53 malicious transitions are correctly flagged, and benign false positives are held to about 16% (15/91).

Unlike the earlier benign→malicious v3 model, which only considered upgrades from a known-good baseline, **v4 learns over the full mix of transition types**, which is closer to real registry monitoring. Even under this harder, more realistic setting, it still attains 0.77 recall on malicious and 0.84 recall on benign transitions.

### Key Features

* **Metadata-only analysis** – No source code or binary inspection required
* **Delta-focused modeling** – Captures changes between versions rather than absolute values
* **Multi-ecosystem support** – Works across npm and PyPI with unified features
* **Practical scoring API** – Simple function to score any package transition

---

## Project Structure

```
Jupyter_v2/
├── data/
│   ├── external/
│   │   ├── malicious-software-packages-dataset/  # DataDog malicious packages
│   │   └── ossf-malicious-packages/              # OSSF OSV malicious packages
│   └── meta/
│       ├── labels_package.csv                    # Package-level labels (v0 output)
│       ├── labels_package_v1.csv                 # Package-level labels (v1)
│       ├── labels_version.csv                    # Version-level labels (v0 output)
│       ├── labels_version_v1.csv                 # Version-level labels (v1)
│       ├── version_delta_features_live.csv       # Delta transition table (v3 output)
│       ├── version_delta_features_v4.csv         # Enhanced delta features (v4)
│       └── selected_delta_features_v4.csv        # Selected features for final model
├── notebooks/
│   ├── refactored_build_labels_v0.ipynb          # Label construction and cleaning
│   ├── version_diff_live_registries_v3.ipynb     # Delta table construction
│   └── new_features_v4.ipynb                     # Feature refinement + final model
├── download_ossf_osv.sh                          # Download OSSF data (Unix/Linux/macOS)
├── download_ossf_osv_v2.sh                       # Download OSSF data (Windows-safe)
├── requirements.txt                              # Python dependencies
└── README.md                                     # This file
```

---

## Pipeline Overview

### Step 0: Build Labels (`refactored_build_labels_v0.ipynb`)

**Goal:** Build consistent labels at package and version levels.

**Data Sources:**
* [OSSF Malicious Packages](https://github.com/ossf/malicious-packages)
* [DataDog Malicious Software Packages](https://github.com/DataDog/malicious-software-packages-dataset)
* [PyPI JSON API](https://docs.pypi.org/api/json/)
* [npm Registry](https://docs.npmjs.com/cli/v8/using-npm/registry)

**Process:**
* Ingests raw OSV / OpenSSF malicious package data
* Normalizes records into package-level and version-level labels
* Applies cleaning (skips malformed versions, normalizes ecosystem identifiers)
* Ensures one row per `(ecosystem, package_name, version)` with `label_malicious ∈ {0,1}`

**Outputs:**
* `data/meta/labels_package.csv` – Package-level labels
* `data/meta/labels_version.csv` – Version-level labels

---

### Step 1: Build Delta Table (`version_diff_live_registries_v3.ipynb`)

**Goal:** Convert per-version metadata into transitions and compute deltas.

**Process:**

1. **Load version-level metadata + labels**
   * Import per-version static features from registry scraping
   * Join with `labels_version.csv` to obtain malicious labels

2. **Sort and pair consecutive versions**
   * Group by `(ecosystem, package_name)` and sort by version
   * For each version `v_i`, find predecessor `v_{i-1}`
   * Build transition rows with prev/current version pairs

3. **Compute delta and ratio features**
   * Text/metadata deltas: `delta_description_len`, `delta_num_dependencies`, etc.
   * Version string features: `delta_version_len`, `delta_version_num_dots`
   * Size deltas: `static_size_delta_vs_prev`, `static_size_ratio_vs_prev`
   * Registry-derived metrics: `delta_npm_unpacked_size_bytes`, `ratio_npm_file_count`
   * Entropy proxies: `ratio_static_size_uncompressed_bytes`, bytes-per-file indicators

4. **Define transition label**
   * `y_malicious_next = label_malicious_current` (Is the next version malicious?)

**Output:**
* `data/meta/version_delta_features_live.csv` – Pure delta table with ~O(#transitions) rows

---

### Step 2: Feature Refinement & Final Model (`new_features_v4.ipynb`)

**Goal:** Build a clean v4 transition model using only delta/ratio features.

**Process:**

1. **Load delta table** from v3 output

2. **Curate and expand feature set**
   * Unified size deltas across ecosystems
   * Registry size + file count (deltas & ratios)
   * Compression/entropy proxies
   * Text/metadata deltas
   * **Design rule:** Delta table stays delta-only; no raw registry columns merged

3. **Train/validation split & feature selection**
   * Stratified train/test split
   * Univariate selection (e.g., `SelectKBest(f_classif)`) to pick top K signals
   * Manual refinement based on stability/interpretability

4. **Final classifier**
   * Train `RandomForestClassifier` on selected delta features
   * Evaluate: confusion matrix, precision/recall/F1, ROC-AUC
   * Balance catching malicious transitions vs. avoiding false positives

5. **Scoring helper – `score_transition`**
   * Simple API to score any package transition
   * Returns `P(malicious | delta)` for a given version upgrade

**Outputs:**
* `data/meta/version_delta_features_v4.csv` – Enhanced delta features
* `data/meta/selected_delta_features_v4.csv` – Selected features
* Trained model artifacts (e.g., `models/clf_transition.joblib`)

---

## How to Use the Model

### Inputs Required

* `ecosystem` – `"npm"` or `"pypi"`
* `package_name` – Canonical package name
* `base_version` – Known benign version (baseline)
* `next_version` – Candidate version to evaluate
* `clf_all` – Loaded scikit-learn model
* `delta_df` – Delta table with selected features
* `feature_cols` – List of selected features

### Usage Pattern

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load delta table (generated from v3 notebook)
delta_df = pd.read_csv("data/meta/version_delta_features_live.csv")

# Load selected features (generated from v4 notebook)
selected_features_df = pd.read_csv("data/meta/selected_delta_features_v4.csv")
selected_features = selected_features_df['selected_feature'].tolist()

# Train model (or load pre-trained model)
# This example shows training - in production you'd load a saved model
X = delta_df[selected_features].fillna(0)
y = delta_df['label_malicious']

clf_all = RandomForestClassifier(n_estimators=100, random_state=42)
clf_all.fit(X, y)

# Score a transition using the helper function from v4 notebook
p_mal = score_transition(
    model=clf_all,
    df_delta=delta_df,
    ecosystem="pypi",
    package_name="examplepkg",
    base_version="1.0.0",
    next_version="1.1.0",
    feature_cols=selected_features,
)

if p_mal >= 0.5:   # threshold can be tuned
    print("High-risk transition – investigate before upgrading.")
else:
    print("Transition looks benign based on metadata deltas.")
```

### Integration Options

* CI pipelines (blocking or warning on high-risk transitions)
* Internal dependency dashboards
* Offline batch scoring for threat-hunting

Thresholds should be tuned based on your risk tolerance and class imbalance.

---

## Reproducing the Experiments

### Environment Setup

**Requirements:**
* Python 3.10+
* Dependencies in `requirements.txt`:
  * `pandas` – Data manipulation
  * `numpy` – Numerical operations
  * `scikit-learn` – Machine learning models
  * `joblib` – Model serialization
  * `packaging` – Version parsing and sorting
  * `requests` – Registry API calls
  * `matplotlib` – Visualization
  * `jupyterlab`, `ipykernel` – Notebook environment (optional)

**Installation:**

```bash
pip install -r requirements.txt
```

### Data Setup

Download OSSF malicious package datasets:

```bash
# Unix/Linux/macOS:
cd data/external
bash ../../download_ossf_osv.sh

# Windows (Git Bash or WSL):
cd data/external
bash ../../download_ossf_osv_v2.sh
```

This populates `data/external/ossf-malicious-packages/` with OSV JSON files for PyPI and npm.

### Execution Order

**Quick Start (using existing data):**

If `data/meta/labels_version.csv` and `data/meta/version_delta_features_live.csv` already exist, skip directly to step 3.

**Full Pipeline:**

1. **Download malicious package data** (see Data Setup above)

2. **Build labels** (if regenerating from scratch)
   * Open and run `notebooks/refactored_build_labels_v0.ipynb`
   * Outputs: `data/meta/labels_package.csv`, `data/meta/labels_version.csv`

3. **Build delta table** (if regenerating from scratch)
   * Open and run `notebooks/version_diff_live_registries_v3.ipynb`
   * Fetches metadata from npm/PyPI registries
   * Computes transition deltas
   * Output: `data/meta/version_delta_features_live.csv`

4. **Train final model**
   * Open and run `notebooks/new_features_v4.ipynb`
   * Loads delta table from step 3
   * Applies feature engineering and selection
   * Trains `clf_all` classifier

5. **Use the model** (see "How to Use the Model" section)

**Note:** Steps 2-3 can be skipped if you already have the required CSV files in `data/meta/`. The v4 notebook can run independently with existing data.

---

## Key Features & Signals

### What the Model Detects

The transition classifier identifies suspicious patterns in version-to-version changes:

1. **Size anomalies**
   * Sudden package size spikes or drops
   * Unified size deltas across npm (unpacked size) and PyPI (compressed size)
   * Log-magnitude features capture large jumps without outlier skew

2. **Structural shifts**
   * File count changes (many small files → one large blob)
   * Density changes (bytes-per-file ratios)
   * Large-jump boolean flags for dramatic changes

3. **Metadata anomalies**
   * Dependency count changes (`delta_num_dependencies`, `delta_num_dev_dependencies`)
   * Script additions (npm `postinstall`, `preinstall` hooks)
   * Classifier/keyword changes in PyPI packages

4. **Version pattern anomalies**
   * Unusual version string changes (`delta_version_len`, `delta_version_num_dots`)
   * Prerelease flag changes

### Example Features

* `delta_description_len` – Description text length change
* `delta_num_dev_dependencies` – Dev dependency count change
* `static_size_delta_vs_prev` – Absolute size change
* `static_size_ratio_vs_prev` – Relative size change
* `ratio_npm_unpacked_size_bytes` – npm unpacked size ratio
* `delta_npm_unpacked_size_bytes` – npm unpacked size delta

### What Are "Structural Shifts"?

Big changes in how a package is organized between versions (A → B):

1. **Files & layout**
   * File count jumps/drops significantly
   * New large bundled blob appears (e.g., one huge .js instead of many small files)

2. **Density**
   * Same file count but way more bytes per file

3. **Metadata/config**
   * `delta_num_scripts` spikes (new npm `postinstall`/`preinstall` hooks)
   * `delta_num_dependencies` / `delta_num_dev_dependencies` change sharply
   * `delta_num_classifiers` / `delta_num_keywords` change oddly

4. **Version/release pattern**
   * Unusual version bump pattern (`delta_version_len`, `delta_version_num_dots`, prerelease flags)

These features detect when "the internal structure and wiring of this package changed a lot in one jump" – a common indicator of injected malicious logic.

---

## Why v4 Performs Better

v4's model improved over v3 by adding smarter delta features:

* **Unified size deltas/ratios** across PyPI + npm → clean "how much did this release change?" signal
* **Density/bytes-per-file proxies** → catch packages that suddenly become denser/packed
* **Log-magnitude + sign features** → model sees change magnitude and direction without outlier skew
* **Large-jump boolean flags** → crisp "this change was huge" indicators that trees love
* **Automatic feature selection** → keeps strongest ~30 signals from expanded pool

**Net effect:** More sensitive to suspicious size, density, and structural shifts while being more robust across ecosystems.

---

## Limitations & Future Work

### Current Limitations

* **Dataset size & bias:** Limited number of labeled transitions; more diverse data would improve generalization
* **Metadata-only:** Cannot detect all malicious patterns (e.g., obfuscated code, logic bombs)
* **Version sorting:** May mis-order irregular version strings
* **Ecosystem coverage:** Currently supports npm and PyPI only
* **Cold start:** Requires at least one known-good version as baseline

### Possible Extensions

* **More ecosystems:** RubyGems, crates.io, Maven Central, NuGet
* **Richer features:** Manifest structure analysis, simple AST statistics
* **Model improvements:** Calibrated probabilities, ensemble methods, active learning
* **Integration:** CI/CD plugins, dependency scanning tools, security dashboards
* **Real-time monitoring:** Stream processing for new package releases

---

## Summary

This project provides a **lightweight, registry-driven early warning system** for suspicious package upgrades:

* **v0** builds robust labels from OSSF and DataDog datasets
* **v3** constructs a clean delta-only transition table
* **v4** refines features and trains a final transition classifier with a simple scoring API

Everything is driven by Python notebooks and CSV artifacts, making it straightforward to reproduce, extend, or integrate into your own security tooling.

---

## Getting Help

For questions or issues:

1. Check notebook comments and markdown cells for detailed explanations
2. Review `data/meta/` CSV files to understand data structure
3. Examine feature engineering code in `new_features_v4.ipynb`
4. Verify file paths match your local setup
