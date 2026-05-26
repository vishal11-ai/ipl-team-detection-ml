# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IIT Bombay CMInDS e-PGDiploma course project (40 marks, due June 06, 2026). Goal: classify IPL team jerseys in cricket images using a grid-based ML approach — **no deep learning allowed**.

## Expected Pipeline Commands

Once scripts are created, the pipeline should run as:

```bash
python preprocess.py          # resize/validate images → dataset/
python extract_features.py    # extract hand-crafted features → features.pkl
python train.py               # train classifier → model_<teamname>.pkl
python inference.py           # run on train+test → predictions.csv
jupyter notebook              # for exploration notebooks
```

## Architecture

### Core Approach: Grid-Based Classification

Every 800×600 image is divided into an **8×8 grid = 64 cells** (each cell is 100×75 px). The model independently classifies each cell as one of 11 classes (0 = no team, 1–10 = IPL franchise). This is a flat multiclass classification problem, not object detection.

### ML Pipeline Stages

1. **Preprocessing** (`preprocess.py`) — resize images to 800×600, enforce 4:3 aspect ratio. Never upscale images below 800×600; discard them instead. Output goes to `dataset/` with subfolders per team.

2. **Feature extraction** (`extract_features.py`) — for each image, extract features from all 64 cells independently. Concatenate cell-level feature vectors to build training rows. Allowed techniques: HOG, color histograms, Canny edges, Sobel filters, LBP (Local Binary Patterns), GLCM, Fourier transforms. **CNN/deep learning features are strictly forbidden** — this is a hard course requirement that will fail the submission if violated.

3. **Labeling** — the label per cell (0–10) must be assigned per the team enumeration below. When multiple teams occupy a single cell, predict any one of them.

4. **Model training** (`train.py`) — classical ML classifiers only (SVM, Random Forest, Logistic Regression, etc.). Save final model as `model_<teamname>.pkl` using `pickle`.

5. **Inference** (`inference.py`) — load the `.pkl`, run on all train and test images, write `predictions.csv`.

### Output Format

`predictions.csv` must follow this exact schema:
```
Image File Name, Train Or Test, c01, c02, c03, ..., c64
```
Each `cNN` column holds an integer 0–10. Column indices are 1-based and zero-padded (c01–c64), reading left-to-right, top-to-bottom across the 8×8 grid.

### Team Label Mapping

| Label | Team |
|-------|------|
| 0 | No team |
| 1 | Chennai Super Kings (CSK) |
| 2 | Delhi Capitals (DC) |
| 3 | Gujarat Titans (GT) |
| 4 | Kolkata Knight Riders (KKR) |
| 5 | Lucknow Super Giants (LSG) |
| 6 | Mumbai Indians (MI) |
| 7 | Punjab Kings (PBKS) |
| 8 | Rajasthan Royals (RR) |
| 9 | Royal Challengers Bengaluru (RCB) |
| 10 | Sunrisers Hyderabad (SRH) |

## Dataset Requirements

- **Single IPL season only** — pick one season and use only images from that year (jerseys vary by season).
- Minimum 100 images per franchise (1000+ total recommended).
- All images: 800×600 px, 4:3 aspect ratio. Downscale is fine; never upscale.
- Include no-player images (empty pitch, crowd) to help the model learn class 0.
- Exclude images where only crowd (not players) wear jerseys.
- `dataset/README.txt` must describe image sources.

## Deliverables Checklist

- `dataset/` — images with folder structure + `README.txt`
- `extract_features.py` / notebooks — feature engineering code
- `train.py` / notebooks — training code
- `inference.py` — pipeline code that loads `.pkl` and produces CSV output
- `model_<teamname>.pkl` — serialized trained model
- `predictions.csv` — predictions on train + test sets
- Presentation slide deck (title slide → ≤3 exec summary slides → full methodology)
- Video ≤ 5 min 15 sec (penalty if exceeded)

## Hard Constraints

- **No CNNs or equivalent automatic feature learners** — not even for pre-labelling. Features must come entirely from hand-crafted image processing.
- Model saved as `.pkl` (pickle format).
- Presentation must document dead-ends and failed approaches, not just the final solution.
