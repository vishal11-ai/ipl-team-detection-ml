# Single Image Exploration Plan

A hands-on sandbox for any team member to understand the full pipeline on one image before committing to the full dataset. Run this in a Jupyter notebook: `notebooks/explore.ipynb`.

---

## Setup

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2hsv
```

Pick any one cricket image with at least one visible jersey. Place it at `sample/test_image.jpg`.

---

## Step 1 — Load & Inspect the Image

```python
img = cv2.imread("sample/test_image.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("Shape:", img.shape)          # should be (600, 800, 3) after resize
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()
```

**Resize to 800×600 if needed:**
```python
if img.shape[:2] != (600, 800):
    img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("Resized to 800x600")
```

---

## Step 2 — Draw the 8×8 Grid

Visualize all 64 cells overlaid on the image. Each cell is 100×75 px.

```python
GRID_ROWS, GRID_COLS = 8, 8
CELL_W, CELL_H = 800 // GRID_COLS, 600 // GRID_ROWS   # 100 x 75

fig, ax = plt.subplots(1, 1, figsize=(12, 9))
ax.imshow(img_rgb)

for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        cell_idx = r * GRID_COLS + c + 1  # 1-based (c01 to c64)
        x, y = c * CELL_W, r * CELL_H
        rect = patches.Rectangle((x, y), CELL_W, CELL_H,
                                  linewidth=1, edgecolor='white', facecolor='none')
        ax.add_patch(rect)
        ax.text(x + 2, y + 12, f"c{cell_idx:02d}",
                color='yellow', fontsize=6, fontweight='bold')

plt.title("8×8 Grid Overlay")
plt.axis("off")
plt.show()
```

---

## Step 3 — Extract a Single Cell

Crop cell `c29` (row 4, col 5 — 0-indexed: row=3, col=4) as an example.

```python
def get_cell(img, cell_idx_1based):
    """cell_idx_1based: 1 to 64, left-to-right top-to-bottom."""
    idx = cell_idx_1based - 1
    row = idx // GRID_COLS
    col = idx % GRID_COLS
    y1, y2 = row * CELL_H, (row + 1) * CELL_H
    x1, x2 = col * CELL_W, (col + 1) * CELL_W
    return img[y1:y2, x1:x2]

cell = get_cell(img_rgb, 29)
plt.imshow(cell)
plt.title("Cell c29 (100×75 px)")
plt.axis("off")
plt.show()
```

---

## Step 4 — Try Feature Extraction on the Cell

### 4a. Color Histogram (HSV)

```python
cell_hsv = rgb2hsv(cell)

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
channel_names = ["Hue", "Saturation", "Value"]
colors = ["r", "g", "b"]
feature_color = []

for i, (ax, name, col) in enumerate(zip(axes, channel_names, colors)):
    hist, _ = np.histogram(cell_hsv[:, :, i], bins=32, range=(0, 1))
    hist = hist / hist.sum()  # normalize
    feature_color.extend(hist.tolist())
    ax.bar(range(32), hist, color=col, alpha=0.7)
    ax.set_title(name)

print("Color feature vector length:", len(feature_color))  # 96
```

### 4b. HOG Features

```python
from skimage.feature import hog
from skimage import exposure

cell_gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)

hog_features, hog_image = hog(
    cell_gray,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True,
    feature_vector=True
)

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(cell_gray, cmap='gray'); ax1.set_title("Cell Grayscale")
ax2.imshow(hog_image_rescaled, cmap='gray'); ax2.set_title(f"HOG ({len(hog_features)} features)")
plt.show()
```

### 4c. LBP Features

```python
lbp = local_binary_pattern(cell_gray, P=8, R=1, method='uniform')
lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
lbp_hist = lbp_hist / lbp_hist.sum()

plt.bar(range(10), lbp_hist)
plt.title(f"LBP Histogram (10 bins)")
plt.show()
```

### 4d. Combine into One Feature Vector

```python
full_feature = np.concatenate([feature_color, hog_features, lbp_hist])
print("Total feature vector length:", len(full_feature))
```

---

## Step 5 — Compare Two Cells Side-by-Side

Pick one cell that clearly has a jersey (e.g., c29 = CSK yellow) and one background cell (e.g., c01 = sky/grass).

```python
cell_jersey = get_cell(img_rgb, 29)   # adjust to a cell with jersey
cell_bg     = get_cell(img_rgb, 1)    # adjust to a background cell

fig, axes = plt.subplots(2, 4, figsize=(16, 6))
for row_idx, (cell, title) in enumerate([(cell_jersey, "Jersey Cell"), (cell_bg, "Background Cell")]):
    cell_hsv = rgb2hsv(cell)
    cell_gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
    
    axes[row_idx][0].imshow(cell); axes[row_idx][0].set_title(f"{title} - RGB")
    
    for ch, name in enumerate(["Hue", "Sat", "Val"]):
        hist, _ = np.histogram(cell_hsv[:, :, ch], bins=32, range=(0,1))
        axes[row_idx][ch+1].bar(range(32), hist/hist.sum(), alpha=0.7)
        axes[row_idx][ch+1].set_title(f"{name} Histogram")

plt.tight_layout()
plt.show()
```

**What to look for:** Jersey cells from the same team should have similar hue peaks. Background cells should look flat/random.

---

## Step 6 — Full Image: Extract Features for All 64 Cells

```python
def extract_cell_features(cell_rgb):
    cell_hsv = rgb2hsv(cell_rgb)
    cell_gray = cv2.cvtColor(cell_rgb, cv2.COLOR_RGB2GRAY)
    
    color_hist = []
    for ch in range(3):
        h, _ = np.histogram(cell_hsv[:, :, ch], bins=32, range=(0, 1))
        color_hist.extend((h / h.sum()).tolist())
    
    hog_feats = hog(cell_gray, orientations=9, pixels_per_cell=(8,8),
                    cells_per_block=(2,2), feature_vector=True)
    
    lbp = local_binary_pattern(cell_gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0,10))
    lbp_hist = lbp_hist / lbp_hist.sum()
    
    return np.concatenate([color_hist, hog_feats, lbp_hist])

all_features = []
for i in range(1, 65):
    cell = get_cell(img_rgb, i)
    all_features.append(extract_cell_features(cell))

all_features = np.array(all_features)
print("Feature matrix shape:", all_features.shape)  # (64, feature_dim)
```

---

## Step 7 — Visualize Dominant Hue per Cell (Quick Team Hint)

Before any model, use dominant hue to see which cells are "colorful" (likely jersey) vs neutral (background).

```python
dominant_hue = []
for i in range(1, 65):
    cell = get_cell(img_rgb, i)
    hsv = rgb2hsv(cell)
    # high saturation pixels are likely jersey, not background
    mask = hsv[:, :, 1] > 0.3
    if mask.sum() > 10:
        dominant_hue.append(hsv[:, :, 0][mask].mean())
    else:
        dominant_hue.append(0)  # no strong color = background

hue_grid = np.array(dominant_hue).reshape(8, 8)

plt.figure(figsize=(8, 6))
plt.imshow(hue_grid, cmap='hsv', vmin=0, vmax=1)
plt.colorbar(label="Dominant Hue (0=red, 0.17=yellow, 0.33=green, 0.67=blue)")
plt.title("Dominant Jersey Hue per Cell\n(high saturation pixels only)")
for r in range(8):
    for c in range(8):
        plt.text(c, r, f"c{r*8+c+1:02d}", ha='center', va='center',
                 fontsize=7, color='black')
plt.show()
```

---

## What You've Learned After This Exploration

- How the 8×8 grid maps to actual image regions
- What feature vectors look like for jersey vs background cells
- That HSV color histograms are the most team-discriminative feature (jersey colors are strong and saturated)
- HOG adds texture/shape info that helps when colors are similar (e.g., CSK yellow vs GT gold)
- You now have the building blocks to wire up the full pipeline in Phase 3
