# Malicious Package Version Transition Detection

A lightweight, registry-driven early warning system for detecting suspicious package upgrades in npm and PyPI ecosystems using metadata analysis.

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
├── case_study/
│   └── shai_hulud.md                             # Case study (motivation behind model)
├── data/
│   ├── external/
│   │   ├── malicious-software-packages-dataset/  # DataDog malicious packages
│   │   └── ossf-malicious-packages/              # OSSF OSV malicious packages
│   └── meta/
│       ├── labels_package.csv                    # Package-level labels (patial-data)
│       ├── labels_package_v1.csv                 # Package-level labels (full-data)
│       ├── labels_version.csv                    # Version-level labels (partial-data)
│       ├── labels_version_v1.csv                 # Version-level labels (full-data)
│       ├── version_delta_features_live.csv       # Delta transition table (v3 output)
│       ├── version_delta_features_v4.csv         # Enhanced delta features (v4 output)
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
   * **Design rule:** Delta table stays delta-only

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

---

## How to Use the Model (v4)

### Inputs / Files

You need:

* `data/meta/version_delta_features_v4.csv` (the delta-feature table)

  * must include: `ecosystem`, `package_name`, `prev_version`, `version`
  * must include all columns in `selected_features`
  * typically includes the label column `y_malicious` (current-version label per transition)
* `data/meta/selected_delta_features_v4.csv` (one column: `selected_feature`)
* A trained scikit-learn classifier (e.g., `clf_all`), either trained in-notebook or loaded from disk.

### Helper: score a single version-to-version transition

`score_transition(...)` wraps a trained classifier + the delta table to return:

**P(malicious)** for: `ecosystem:package_name base_version → next_version`

What it does:

* Finds the **single** matching row in `df_delta` where:

  * `ecosystem == ecosystem`
  * `package_name == package_name`
  * `prev_version == base_version`
  * `version == next_version`
* Extracts only `feature_cols`, fills missing with `0`
* Calls `model.predict_proba(...)` and returns the probability for class `1` (malicious)
* Raises `ValueError` if the transition row is missing (so you catch gaps in your delta table early)

### Example usage

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load v4 delta table + selected features
delta_df = pd.read_csv("data/meta/version_delta_features_v4.csv")
selected_features = pd.read_csv(
    "data/meta/selected_delta_features_v4.csv"
)["selected_feature"].tolist()

# Train (or load) a classifier
X = delta_df[selected_features].fillna(0)
y = delta_df["y_malicious"].astype(int)

clf_all = RandomForestClassifier(n_estimators=200, random_state=42)
clf_all.fit(X, y)

# Score a real transition that exists in delta_df
try:
    p_mal = score_transition(
        model=clf_all,
        df_delta=delta_df,
        ecosystem="pypi",
        package_name="examplepkg",
        base_version="1.0.0",
        next_version="1.1.0",
        feature_cols=selected_features,
    )
    print(f"P(malicious): {p_mal:.3f}")

    if p_mal >= 0.5:  # threshold is tunable
        print("High-risk transition — investigate before upgrading.")
    else:
        print("Transition looks benign based on metadata deltas.")
except ValueError as e:
    print(f"[ERROR] {e}")
    print("This transition row is missing from the delta table.")
```

### Demo-only: synthetic transition row (optional)

If you want a quick sanity-check without relying on a real package row:

1. Define `prev_info` and `curr_info` (raw per-version values)
2. Use `build_dummy_features_from_prev_curr(prev_info, curr_info)` to compute the full delta/ratio feature dict
3. Append a synthetic row into `delta_df` with:

   * `ecosystem`, `package_name`, `prev_version`, `version`
   * feature columns in `selected_features`
   * optionally `y_malicious = 1` (for demonstration)
4. Call `score_transition(...)` on the synthetic identifiers.

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

If `data/meta/labels_version.csv` and `data/meta/version_delta_features_live.csv` already exist, skip directly to step 4 (confirm correct file paths are used).

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

**Note:** Steps 1-3 can be skipped if you already have the required CSV files in `data/meta/`. The v4 notebook can run independently with existing data.

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
* **Ecosystem coverage:** Currently supports npm and PyPI only

### Possible Extensions

* **More ecosystems:** RubyGems, crates.io, Maven Central, NuGet
* **Richer features:** Dynamic Analysis features (process creation / command-lines, file system writes, registry diffs, network/DNS/HTTP connections & IOCs), Manifest structure analysis, simple AST statistics 
* **Model improvements:** Calibrated probabilities, ensemble methods, active learning
* **Integration:** CI/CD plugins, dependency scanning tools, security dashboards
* **Real-time monitoring:** Stream processing for new package releases

---

## Summary


This project provides a **lightweight, registry-driven early warning system** for suspicious package upgrades:

* v0 builds robust labels.
* v3 constructs a clean **delta-only transition table**.
* v4 refines features and trains a **final transition classifier** with a simple scoring API.

Everything is driven by the attached Python notebooks and their underlying CSV/serialized artifacts, making it straightforward to reproduce, extend, or integrate into your own security tooling.

[Demo Video](https://www.youtube.com/watch?v=L1adPJaGESw)
