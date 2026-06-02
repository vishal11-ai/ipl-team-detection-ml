
## Important explorations and Edge-Cases

- image downscaling with LANCZOS filter for better quality (https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize)

- Stratified sampling for train-test split to maintain class distribution.
- Data Augemtaion needed ?

| Signal          | What it captures                              | Best technique          |
| --------------- | --------------------------------------------- | ----------------------- |
| Color           | Jersey base color (e.g., yellow=CSK, blue=MI) | Color Histograms in HSV |
| Texture/Pattern | Jersey stripes, logos, patterns               | LBP, HOG, Gabor filters |
| Edges/Shape     | Jersey boundaries, logo outlines              | HOG, Sobel, Canny       |

---
#### Category 1: Jersey Color Conflicts
These are the most dangerous — where two different teams have similar hues in HSV space.

| Conflict         | Teams                                            | Why Dangerous                        |
| ---------------- | ------------------------------------------------ | ------------------------------------ |
| Yellow vs Yellow | CSK vs GT (gold)                                 | H difference < 8 units               |
| Orange vs Orange | SRH vs PBKS (some seasons)                       | Nearly identical warm hue            |
| Blue vs Blue     | MI (royal blue) vs DC (navy) vs LSG (light blue) | All blue, different saturation/value |
| Red vs Red       | RCB vs PBKS (red variant)                        | Same H range, different pattern      |
| Purple vs Purple | KKR vs RCB (away jersey)                         | Overlapping H range                  |
| Gold vs Orange   | KKR gold trim vs SRH orange                      | H≈20-30 for both                     |

> Fix: Never use Hue alone. Always use joint H×S 2D histogram and add dominant color pairs (jersey always has 2+ colors — KKR is purple+gold, never just purple).

---
#### Category 2: Lighting & Environment Conditions
🌙 Night / Floodlit Matches
V (brightness) drops 40–60% — yellow CSK jersey looks olive-green under certain floodlight angles

Shadows across jersey create split cells — half bright, half dark of same jersey

> Fix: Always apply cv2.equalizeHist() on the V channel before feature extraction

☀️ Harsh Daylight / Overexposure
White jerseys (LSG white alternate) become fully saturated (V=255, S≈0) — indistinguishable from white pitch markings, sight screens, or umpire coats

> Fix: Use S channel threshold — genuine jersey white has slight tint; pure overexposed areas have S≈0

🌧️ Overcast / Dull Conditions
Colors appear desaturated — all jerseys shift toward gray

> Fix: Histogram stretching on S channel before feature extraction

🔦 Mixed Lighting (Dusk Matches)
One side of player lit by sunlight, other by floodlight — same jersey cell has two different color readings

> Fix: Extract features from left half and right half of cell separately, concatenate

---

#### Category 3: Player Occlusion

| Occlusion Type            | Example                                | Impact                                                    |
| ------------------------- | -------------------------------------- | --------------------------------------------------------- |
| Player-player overlap     | Two batsmen at crease                  | Cell has jersey colors from 2 teams blended               |
| Equipment occlusion       | Bat, helmet visor, pad blocking jersey | Cell reads as gray/black/white noise                      |
| Umpire overlapping player | White coat in front of colored jersey  | Cells flip to white                                       |
| Partial body in frame     | Only legs/feet visible at image edge   | Cell has only trouser color — less distinctive than chest |
| Fielder diving/sliding    | Player horizontal, jersey stretched    | HOG gradients distorted; texture changes                  |
| Wicket-keeper crouching   | Only back/helmet visible               | No chest jersey visible in cells                          |

> Fix for all occlusion cases: Label occluded cells as 0 if < 30% of the cell has visible jersey. Use Laplacian sharpness + saturation threshold to auto-detect non-jersey cells.
---

#### Category 4: Background Contamination

Stadium Hoardings & Sponsor Boards
RuPay boards = blue → looks like MI/DC/LSG

AngelOne boards = orange → looks like SRH

TATA NEU boards = purple/blue → looks like KKR

Yellow CSK hoarding wheels (as seen in your Image 3) → looks like CSK jersey

> Fix: Background cells are blurry (low Laplacian variance). Add blur score as a feature — blurry + colored = label 0

Crowd Wearing Team Colors
Fans in CSK yellow fill background cells → false CSK positives

> Fix: Jersey texture (HOG/LBP) from crowd is different — fan T-shirts are plain, player jerseys have logos, numbers, sponsor text. HOG captures this structural difference

Green Pitch / Outfield
H≈60-80 (green) is not close to any jersey, but bright green can bleed into adjacent cells

> Fix: Mask green hue range in background cells using simple HSV threshold

Sight Screen (Pure White Background)
White background cells → very similar to white jersey (LSG, DC away)

> Fix: Sight screen is always at image edges and is perfectly uniform — use cell variance (low variance = uniform background, not jersey)

---

#### Category 5: Equipment & Accessories

| Equipment             | Color                               | Confusion                            |
| --------------------- | ----------------------------------- | ------------------------------------ |
| Batting gloves        | Blue, red, orange trim              | MI, SRH, RCB false positives         |
| Batting pads          | White/cream                         | Sight screen, LSG away jersey        |
| Helmet                | Team-colored (CSK yellow, KKR gold) | Actually useful! Adds correct signal |
| Bat                   | Brown willow + colored grip         | Isolated vertical brown strip        |
| Wicket-keeping gloves | White with orange/red cuff          | Noise in keeper cells                |
| Sunglasses/visor      | Dark black/mirror                   | Pulls cell V down to near-zero       |
| Wristbands            | Various team colors                 | Too small to matter at 100×75 cell   |
| Shoes                 | White/black                         | Bottom-row cells often just shoes    |

> Fix: Bottom 2 rows of the grid (rows 7–8) predominantly contain feet, shoes, grass — these cells statistically tend toward label 0. Add grid position (row, col) as a meta-feature to give the model spatial awareness.

---

#### Category 7: Camera & Image Artifacts

| Artifact                      | Cause                              | Impact                                         |
| ----------------------------- | ---------------------------------- | ---------------------------------------------- |
| Motion blur                   | Fast bowler run-up, diving fielder | Jersey smeared across cells, HOG destroyed     |
| Compression artifacts         | Low quality JPEG source            | Block artifacts confuse LBP/HOG                |
| Watermarks/overlays           | Broadcaster logo, BCCI copyright   | Adds white/colored text on top of jersey cells |
| Zoom blur (bokeh)             | Telephoto lens background blur     | Background cells extremely blurry              |
| Fisheye/wide angle distortion | Spider-cam, boundary cameras       | Player proportions distorted                   |
| Frame from video              | Screen capture from video          | Interlacing artifacts, reduced sharpness       |

> Fix: Add image quality check in Phase 2 validation — reject images with Laplacian variance < 100 globally (globally blurry = motion blur or bad frame).



