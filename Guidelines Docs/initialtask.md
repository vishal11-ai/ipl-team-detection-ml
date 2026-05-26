# PML Project — Initial Task Plan

## Assumptions
- Dataset: IPL 2025 season images, one folder per team (e.g., `dataset/CSK/`, `dataset/MI/`, etc.)
- Language: Python 3.x
- Libraries available: `opencv-python`, `scikit-image`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `joblib`
- No deep learning (CNN/PyTorch/TensorFlow) — hard constraint

---

## Phase 1 — Dataset Setup & Validation

**Goal:** Get images into the right shape so every downstream step can assume clean, uniform input.

### Tasks

- [ ] **1.1 Audit the downloaded dataset**
  - Count images per team folder
  - Identify any teams with < 100 images — flag for top-up
  - Check for duplicates (by filename or hash)

- [ ] **1.2 Write `preprocess.py`**
  - Walk each team folder
  - For each image:
    - Skip if resolution < 800×600 (log skipped files)
    - Downscale to exactly 800×600 if larger (maintain 4:3; crop width if needed)
    - Save to `dataset_clean/<TEAM>/` as `.jpg`
  - Print a summary: per-team count before/after

- [ ] **1.3 Write `dataset/README.txt`**
  - List sources (e.g., Getty Images, IPL official website, Google Images)
  - Note: IPL 2025 season only
  - Mention any exclusion criteria applied

- [ ] **1.4 Create a train/test split**
  - 80% train, 20% test, stratified by team
  - Save split as `splits/train.csv` and `splits/test.csv` (columns: `filepath`, `team_label`)

---

## Phase 2 — Grid Labeling

**Goal:** For each training image, assign a team label (0–10) to each of the 64 grid cells.

### Tasks

- [ ] **2.1 Decide labeling strategy**
  - Since images come from team-specific folders, a practical first approach:
    - If the image is from folder `CSK/`, label the cells that visually contain jersey as `1`; all others as `0`
  - For multi-team images (if any collected): manually annotate or use a simple bounding-box tool
  - Recommended tool for manual annotation: **LabelImg** (outputs YOLO/Pascal VOC bounding boxes) — then convert to grid-cell labels

- [ ] **2.2 Write `label_grid.py`**
  - Input: image path + bounding boxes (or team folder as a proxy)
  - Logic: for each of the 64 cells (100×75 px each), check overlap with bounding boxes → assign dominant team label
  - Output: `labels/<image_name>.npy` — array of shape (64,) with values 0–10
  - For folder-proxy labeling (no bounding boxes): label ALL non-background cells as the folder's team — this is the fast-start approach

- [ ] **2.3 Verify labels visually**
  - Write a small viz script that overlays the 8×8 grid on an image with color-coded cell labels
  - Spot-check ~20 images to confirm labeling makes sense

---

## Phase 3 — Feature Extraction

**Goal:** Convert each 100×75 cell into a fixed-length feature vector using hand-crafted image processing only.

### Tasks

- [ ] **3.1 Write `extract_features.py`**
  - For each image → split into 64 cells (100×75 px)
  - For each cell extract and concatenate:
    - **Color histogram** in HSV space — 3 channels × 32 bins = 96 values (most discriminative for jerseys)
    - **HOG** — captures texture/shape; use `skimage.feature.hog`
    - **LBP** (Local Binary Patterns) — texture descriptor; use `skimage.feature.local_binary_pattern`
    - *(Optional for v2):* Canny edge density, Gabor filter responses, dominant color via k-means
  - Output: `features/train_features.npy` (shape: N_cells × feature_dim) and `features/train_labels.npy`
  - Also build a mapping file `features/image_cell_index.csv` → `(image_name, cell_index, row_in_features)`

- [ ] **3.2 Normalize features**
  - Fit a `StandardScaler` on train features only
  - Save scaler as `scaler.pkl`
  - Apply to both train and test features

---

## Phase 4 — Model Training

**Goal:** Train a classifier that takes a cell's feature vector and predicts 0–10.

### Tasks

- [ ] **4.1 Baseline: train an SVM**
  - `sklearn.svm.SVC(kernel='rbf', class_weight='balanced')`
  - Cross-validate on training set (5-fold), report per-class accuracy and macro F1
  - Class imbalance expected (class 0 = "no team" will dominate)

- [ ] **4.2 Compare alternatives**
  - Random Forest, Gradient Boosting (XGBoost/LightGBM if available)
  - Pick the best by macro F1 on validation fold

- [ ] **4.3 Handle class imbalance**
  - Use `class_weight='balanced'` or oversample class 0 minority team cells
  - Or: train a two-stage model — first detect "team present vs not" (binary), then classify which team

- [ ] **4.4 Save the best model**
  - `pickle.dump(model, open('model_<teamname>.pkl', 'wb'))`
  - Include the scaler inside the same pickle as a tuple: `(scaler, model)`

---

## Phase 5 — Inference & Output

**Goal:** Run the trained model on all images and produce the submission CSV.

### Tasks

- [ ] **5.1 Write `inference.py`**
  - Load `model_<teamname>.pkl`
  - For each image in train + test:
    - Resize to 800×600 if needed
    - Split into 64 cells, extract features (same pipeline as Phase 3), normalize
    - Predict label for each cell
  - Write `predictions.csv`:
    ```
    Image File Name, Train Or Test, c01, c02, ..., c64
    ```

- [ ] **5.2 Sanity check predictions**
  - For a few known images, visualize the 8×8 grid colored by predicted label
  - Confirm dominant label matches expected team

---

## Phase 6 — Evaluation & Documentation

### Tasks

- [ ] **6.1 Compute metrics**
  - Per-class precision, recall, F1 on train and test sets
  - Confusion matrix
  - Overall accuracy and macro F1

- [ ] **6.2 Prepare presentation**
  - Title slide (roll numbers + names)
  - ≤ 3 executive summary slides
  - Full methodology: data collection, preprocessing, feature choices, model selection
  - Results: metrics tables, confusion matrix, example predictions
  - **Include failed approaches and dead-ends** (required by graders)
  - Challenges and learnings

- [ ] **6.3 Record video**
  - ≤ 5 min 15 sec (hard cap — penalty if over)
  - Cover: data → features → model → results → challenges → learnings

---

## Suggested Folder Structure

```
PML/
├── dataset/               # raw downloaded images, one subfolder per team
├── dataset_clean/         # preprocessed 800×600 images
├── splits/
│   ├── train.csv
│   └── test.csv
├── labels/                # per-image .npy arrays of shape (64,)
├── features/
│   ├── train_features.npy
│   ├── test_features.npy
│   ├── train_labels.npy
│   └── image_cell_index.csv
├── preprocess.py
├── label_grid.py
├── extract_features.py
├── train.py
├── inference.py
├── scaler.pkl
├── model_<teamname>.pkl
├── predictions.csv
├── Guidelines Docs/
└── CLAUDE.md
```

---

## Priority Order for First Week

1. Phase 1 (preprocess + validate dataset) — unblocks everything
2. Phase 2.2 (folder-proxy labeling) — fastest path to trainable labels
3. Phase 3.1 (color histogram features only first) — simplest useful feature
4. Phase 4.1 (SVM baseline) — get a number on the board
5. Iterate: add HOG + LBP features, retrain, compare
