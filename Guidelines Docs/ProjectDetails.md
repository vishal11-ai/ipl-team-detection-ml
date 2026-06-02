High-level project description
You must build a classical ML system that, given cricket match images, can

detect where players are in the image, and

identify which IPL team each detected player belongs to, using jersey colour, patterns, and logos as cues.

The system is not required to identify player names, only team affiliation and counts of players per image.

The example target behaviour: for an image containing multiple players from different franchises, your pipeline should detect all players, count them, and assign each one to the correct team.

Constraints and key rules
No deep learning for features:

You are not allowed to use CNNs or any models that automatically learn image features.

You must use hand-crafted feature engineering (e.g., colour histograms, edges, HOG, LBP, GLCM, etc.) and then train classical models like SVM, logistic regression, random forest, etc.

Single IPL season only:

All images must come from one chosen IPL season so that jersey designs are consistent.

Labelling scheme (team IDs):

0: No team

1: Chennai Super Kings (CSK)

2: Delhi Capitals (DC)

3: Gujarat Titans (GT)

4: Kolkata Knight Riders (KKR)

5: Lucknow Super Giants (LSG)

6: Mumbai Indians (MI)

7: Punjab Kings (PBKS)

8: Rajasthan Royals (RR)

9: Royal Challengers Bengaluru (RCB)

10: Sunrisers Hyderabad (SRH)

Dataset requirements and guidelines
You must build your own dataset by collecting cricket images from reliable online sources.

Each franchise must be adequately represented; it is recommended to have at least 100 instances per team.

You are encouraged to collaborate across teams to create a common, large, labelled dataset.

Images should include a mix of scenarios:

Multiple players from different franchises in a single frame

Distinct team jerseys (different colours/designs/logos)

Variations in pose, orientation, full/partial visibility

Different conditions: lighting, camera angles, crowd, occlusions

Single-player and multi-player scenes

Some no-player images (empty pitch, just crowd, etc.) to improve generalization

Image size and ratio:

All images must be 4:3 aspect ratio and resized to 800×600 pixels.

You can downsize higher-resolution images,

But you must not upscale images smaller than 800×600.

You must include a short README describing image sources and dataset structure.

Modelling task and grid labelling
Each 800×600 image is divided into an 8×8 grid, giving 64 cells (c01 to c64).

For each cell, you must predict a label from 0 to 10 (no team or one of 10 teams).

If multiple players/jerseys appear in a single cell, you can assign any one of the appropriate team labels for that cell.

You must design how to generate these cell-level labels and features (this is explicitly left to you).

The final predictions must be saved to a CSV with columns:

Image File Name, Train Or Test, c01 ... c64

Each cXX is one of 0–10 as per the team mapping.

Deliverables and evaluation
You are graded out of 40 marks (20% course weightage), with a submission deadline of June 06, 2026, 23:55 hrs.

You must submit:

Dataset: images, labels, folder structure, README

Source code: all scripts/notebooks for data prep, feature engineering, model training, and inference

Performance metrics: on train, test, and hold-out (unseen) data

Model weights: trained model as a .pkl file (e.g., model_teamname.pkl)

Inference pipeline: code that loads the .pkl, accepts test images, and outputs the required CSV

Predictions CSV: final outputs in the prescribed format

Presentation: slide deck covering problem, approach, experiments, metrics, analysis, failures, challenges, and learnings

Video (~5 minutes): walkthrough of data, features, model, results, challenges, and key learnings

Evaluation (40 marks) is based on:

15 marks: problem detailing, solution approach, completeness, and Data Science steps

15 marks: result quality on train/test and hold-out data; observations and conclusions

10 marks: documentation quality (slides + video)

FAQ-style clarifications (from the FAQ file)
What does “labelling” mean?
Assign the numeric team ID (0–10) to each relevant region / cell/image as per the mapping above.

What does “hand-crafted feature engineering” mean?
You manually compute features (e.g., colour histograms, HOG, edges, textures) and feed them into classical ML models, instead of using CNNs or similar feature-learners.

Can CNNs be used even just for pre-labelling or annotation?
No; CNNs or similar methods are strictly disallowed for feature creation or annotation. Only low-level image processing functions are permitted.

Is it necessary that players (humans) appear?
Players need not always be present; jersey-only images may be acceptable, but images with mostly crowd jerseys that confuse “players” should be discarded.

