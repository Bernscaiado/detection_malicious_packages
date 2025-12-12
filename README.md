# Malicious Package Version Transition Detection

This project builds an end-to-end pipeline to detect **malicious version transitions** in open-source ecosystems (PyPI, npm) using **registry metadata** and **version-to-version delta features** only.

The v4 “all transitions” model is trained on every version-to-version jump in the delta table and reaches 0.81 accuracy. For benign transitions, precision/recall are 0.86/0.84; for malicious transitions they are 0.73/0.77. The confusion matrix [[76 15], [12 41]] means 76/91 benign transitions are correctly kept benign, 41/53 malicious transitions are correctly flagged, and benign false positives are held to about 16% (15/91). Unlike the earlier benign→malicious v3 model, which only considered upgrades from a known-good baseline, v4 learns over the full mix of transition types, which is closer to real registry monitoring. Even under this harder, more realistic setting, it still attains 0.77 recall on malicious and 0.84 recall on benign transitions.

Given a **known good (benign) or unknown version** of a package and a **candidate next version**, the final model predicts the probability that the transition is **malicious vs benign**, based purely on:

* How static metadata changes between versions (sizes, file counts, etc.)
* Simple ratio / “entropy-proxy” features (bytes per file, size ratios, etc.)
* Text/metadata deltas (description length, classifiers, dependencies, etc.)

The project is organized into three main Jupyter notebooks (exported as HTML) that mirror the pipeline:

* `refactored_build_labels_v0.html` – build clean package & version labels
* `version_diff_live_registries_v3.html` – construct version-to-version delta table
* `new_features_v4.html` – refine features and train the final transition model

---

## 1. Pipeline Overview

### Step 0 – Labels & Ground Truth (`refactored_build_labels_v0.html`)

**Goal:** Build consistent labels at **package** and **version** levels.

This notebook:

* Ingests raw OSV / OpenSSF malicious package data and other sources.
* Normalizes records into:

  * `labels_package.csv` – package-level label (e.g., *ever malicious*, ecosystem, etc.).
  * `labels_version.csv` – version-level labels (benign vs malicious).
* Applies basic cleaning:

  * Skips malformed or range-only version records.
  * Normalizes ecosystem identifiers (`npm`, `pypi`, etc.).
  * Ensures one row per `(ecosystem, package_name, version)` with `label_malicious ∈ {0,1}`.

These CSV outputs are the **only inputs** required by later notebooks for labels.

---

### Step 1 – Version-to-Version Delta Table (`version_diff_live_registries_v3.html`)

**Goal:** Convert per-version metadata into **transitions** and compute deltas.

Key operations (conceptual):

1. **Load version-level metadata + labels**

   * Import a table of per-version static features (from prior scraping / registry sync).
   * Join with `labels_version.csv` to obtain `label_malicious` for each version.

2. **Sort and pair consecutive versions**

   * Group by `(ecosystem, package_name)` and sort by the version string.
   * For each version `v_i`, find its immediate predecessor `v_{i-1}`.
   * Build a transition row:

     * `prev_version`, `version`
     * `prev_label_malicious`, `label_malicious` (current)
     * Static features for `prev_` and current.

3. **Compute delta and ratio features**

   For each selected static feature `X` (e.g., size, file count, description length):

   * `delta_X = X_current - X_prev`
   * `ratio_X = safe_ratio(X_current, X_prev)`

   These include:

   * Text/meta deltas:

     * `delta_description_len`, `delta_num_dependencies`, `delta_num_dev_dependencies`,
       `delta_num_keywords`, `delta_num_classifiers`, …
   * Version string features:

     * `delta_version_len`, `delta_version_num_dots`
   * Static size deltas:

     * `static_size_delta_vs_prev`, `static_size_ratio_vs_prev`
   * Registry-derived size & file count (where available):

     * `delta_npm_unpacked_size_bytes`, `ratio_npm_unpacked_size_bytes`
     * `delta_npm_file_count`, `ratio_npm_file_count`
     * Analogous features for PyPI or generic “static_size_*” fields.
   * Entropy proxies:

     * `ratio_static_size_uncompressed_bytes`
     * Bytes per file style indicators (size / file_count).

4. **Define the transition label**

   * `y_malicious_next = label_malicious_current`
     (“Is the **next** version malicious?”)
   * Optionally include `prev_label_malicious` to filter:

     * **benign → malicious** transitions.
     * **benign → benign** “normal” transitions for better negative sampling.

The output of this notebook is a **pure delta table** (often stored as something like
`version_delta_features_live.csv`) with ~O(#transitions) rows and only **delta / ratio** + label columns.

---

### Step 2 – Feature Refinement & Final Model (`new_features_v4.html`)

**Goal:** Build a **clean v4 transition model** that:

* Uses **only delta / ratio features** (no raw registry columns merged in).
* Adds richer size / “entropy-like” proxies.
* Trains and evaluates a final classifier that can be used in a scoring function.

Conceptually, this notebook does:

1. **Load delta table**

   * Import the v3 delta CSV (e.g., `version_delta_features_live.csv`).
   * Filter to transitions where the **previous version is benign** (known-good baseline).

2. **Curate and expand feature set**

   Starting from the existing delta/ratio columns, v4 focuses on:

   * **Unified size deltas:**

     * `static_size_delta_vs_prev`
     * `static_size_ratio_vs_prev`
   * **Registry size + file count (deltas & ratios):**

     * `delta_npm_unpacked_size_bytes`, `ratio_npm_unpacked_size_bytes`
     * `delta_npm_file_count`, `ratio_npm_file_count`
     * Optional PyPI analogues (if present).
   * **Compression / entropy proxies:**

     * `ratio_static_size_uncompressed_bytes`
     * “bytes per file” style metrics (size / file_count) turned into deltas/ratios.
   * **Existing text/metadata deltas:**

     * `delta_description_len`, `delta_num_dependencies`, `delta_num_dev_dependencies`,
       `delta_num_keywords`, `delta_num_classifiers`, …
     * `delta_version_len`, `delta_version_num_dots`

   **Important design rule in v4:**

   > The **delta table stays delta-only**. Raw registry metadata is used *only* to build new delta / ratio features, **not** merged directly into the final modeling table.

3. **Train / validation split & feature selection**

   * Split transitions into train / test sets (e.g., stratified by label).
   * Start from a pool of delta / ratio features.
   * Use univariate selection (e.g., `SelectKBest(f_classif)`) to pick the **top K** signals.
   * Optionally refine manually based on stability / interpretability.

   Example of selected features (illustrative):

   * `delta_description_len`
   * `delta_num_dev_dependencies`
   * `delta_num_keywords`
   * `delta_version_len`
   * `delta_version_num_dots`
   * `static_size_delta_vs_prev`
   * `static_size_ratio_vs_prev`
   * `ratio_static_size_uncompressed_bytes`
   * `delta_npm_unpacked_size_bytes`
   * `ratio_npm_unpacked_size_bytes`

4. **Final classifier**

   * Train a **transition classifier** (e.g., `RandomForestClassifier` from scikit-learn)
     on the selected delta features.
   * Evaluate:

     * Confusion matrix
     * Precision/recall/F1 for malicious class
     * ROC-AUC
   * Use test-set metrics to discuss the trade-off between:

     * Catching true malicious transitions (recall)
     * Avoiding false positives (precision/null impact on developers).

5. **Scoring helper – `score_transition_from_known_good`**

   v4 defines a small helper to make the model easy to use in tooling:

   ```python
   p_mal = score_transition_from_known_good(
       model=clf_transition,
       df_delta=delta_df,
       ecosystem="npm",
       package_name="left-pad",
       base_version="1.1.0",   # known good
       next_version="1.1.1",   # candidate
       feature_cols=selected_features,
   )
   print(
       f"P(malicious) for transition npm:left-pad "
       f"{base_version} -> {next_version}: {p_mal:.3f}"
   )
   ```

   Internally this function:

   * Looks up the matching transition row in `delta_df`.
   * Extracts only `feature_cols`.
   * Calls `model.predict_proba([...])` to obtain `P(malicious | delta)`.

   **This is the main “API” of the project** for external consumers.

---

## 2. Directory / File Guide

* `data/`

  * `meta/`

    * `labels_package.csv` – package-level labels from v0 notebook.
    * `labels_version.csv` – version-level labels from v0 notebook.
  * `processed/`

    * `version_delta_features_live.csv` – v3 delta-only transition table.

* `notebooks/` (or root as HTML exports)

  * `refactored_build_labels_v0.html`
    Label construction and cleaning.
  * `version_diff_live_registries_v3.html`
    Builds the per-transition delta table and basic baselines.
  * `new_features_v4.html`
    Feature refinement + final transition model and scoring helper.

* `README.md`

---

## 3. Final Model: How to Use It

### 3.1 Inputs

To score a transition you need:

* `ecosystem` – `"npm"` or `"pypi"`.
* `package_name` – canonical package name.
* `base_version` – **known benign** version (ideally validated or long-lived stable).
* `next_version` – candidate version you want to evaluate.
* `clf_transition` – loaded scikit-learn model from disk.
* `delta_df` – the delta table built in v3 / v4 (must include the selected features).

### 3.2 Typical usage pattern

1. **Load artifacts:**

   ```python
   import joblib
   import pandas as pd
   import json

   delta_df = pd.read_csv("data/processed/version_delta_features_live.csv")
   clf_transition = joblib.load("models/clf_transition.joblib")
   with open("models/feature_cols_transition.json") as f:
       feature_cols = json.load(f)
   ```

2. **Score a transition:**

   ```python
   p_mal = score_transition_from_known_good(
       model=clf_transition,
       df_delta=delta_df,
       ecosystem="pypi",
       package_name="examplepkg",
       base_version="1.0.0",
       next_version="1.1.0",
       feature_cols=feature_cols,
   )

   if p_mal >= 0.5:   # threshold can be tuned
       print("High-risk transition – investigate before upgrading.")
   else:
       print("Transition looks benign based on metadata deltas.")
   ```

3. **Integrations**

   This pattern can be embedded into:

   * CI pipelines (blocking or warning on high-risk transitions).
   * Internal dependency dashboards.
   * Offline batch scoring for threat-hunting.

Thresholds should be tuned based on your risk tolerance and class imbalance.

---

## 4. Reproducing the Experiments

### 4.1 Environment

Suggested:

* Python 3.10+
* Key libraries:

  * `pandas`
  * `numpy`
  * `scikit-learn`
  * `joblib`
  * `packaging` (for version sorting, if used)
  * `requests` (for any optional live registry calls)

Install via:

```bash
pip install -r requirements.txt
```

### 4.2 Recommended run order

NOTE: You don’t need to run v0 or v3. Notebook v4 can run independently as long as the two input .csv files are present on the machine and the file paths are configured correctly.

1. **Build labels (if needed)**
   Run the logic in `refactored_build_labels_v0.ipynb` to regenerate
   `labels_package.csv` and `labels_version.csv`.

2. **Build delta table**
   Run `version_diff_live_registries_v3.ipynb` to:

   * Load per-version static metadata.
   * Merge with `labels_version.csv`.
   * Compute deltas/ratios.
   * Write `version_delta_features_live.csv`.

3. **Train / update final model**
   Run `new_features_v4.ipynb`:

   * Load delta table.
   * Apply feature selection.
   * Train `clf_transition`.

4. **Use the model**.

---

## 5. Limitations & Future Work

* **Dataset size & bias:** Current delta table has a limited number of transitions; more data from diverse ecosystems would improve generalization.
* **Metadata-only:** The model explicitly avoids source code / binary inspection. It **cannot** detect all malicious patterns; it focuses on anomalous metadata changes.
* **Version sorting:** Simple lexicographic or `packaging.version` comparisons may mis-order highly irregular version strings.

Possible extensions:

* Expand to more ecosystems (e.g., RubyGems, crates.io).
* Add richer static features (e.g., manifest structure, simple AST stats) while keeping the “delta” design.
* Explore calibrated models and active learning for better thresholds and analyst feedback.

---

## 6. Summary

This project provides a **lightweight, registry-driven early warning system** for suspicious package upgrades:

* v0 builds robust labels.
* v3 constructs a clean **delta-only transition table**.
* v4 refines features and trains a **final transition classifier** with a simple scoring API.

v4’s model got better mainly because it added smarter delta features on top of v3:
Unified size deltas/ratios across PyPI + npm → one clean “how much did this release change in size?” signal.
* Density / bytes-per-file proxies → catch packages that suddenly become much denser/packed.
* Log-magnitude + sign features for big jumps → model sees how big the change is and in which direction (grow vs shrink) without being skewed by outliers.
* Large-jump boolean flags → crisp “this change was huge” indicators that trees love.
* Automatic feature selection over both old v3 deltas and the new ones → keeps the strongest ~30 signals.
* Net effect: the model is more sensitive to suspicious size, density, and structural shifts between versions, while being more robust across ecosystems.

In this notebook context, “structural shifts” = big changes in how the package is put together, not just how big it is.
Examples between version A → B:
1. Files & layout
   * Number of files jumps or drops a lot
   * New big bundled blob appears (e.g., one huge .js instead of many small ones)
2. Density
   * Same file count but way more bytes per file (your bytes-per-file proxies)
4. Metadata / config
  * delta_num_scripts spikes (new npm scripts like postinstall, preinstall)
  * delta_num_dependencies / delta_num_dev_dependencies change sharply
  * delta_num_classifiers / delta_num_keywords change in odd ways
4. Version / release pattern
  * Weird version bump pattern (delta_version_len, delta_version_num_dots, prerelease flags)

Those features are all trying to say: “the internal structure and wiring of this package changed a lot in one jump" which is often what happens when someone injects malicious logic.

Everything is driven by the attached Python notebooks and their underlying CSV/serialized artifacts, making it straightforward to reproduce, extend, or integrate into your own security tooling.
