# Dataset Analysis Report
### Group 18 — Naruto Hand Seal Detection

---

## 1) Condition of Kaggle Dataset

The Kaggle dataset ([vikranthkanumuru/naruto-hand-sign-dataset](https://www.kaggle.com/datasets/vikranthkanumuru/naruto-hand-sign-dataset)) comes pre-split into a `train` and `test` folder, each containing one subdirectory per seal class. All images are `640×480` px JPEGs.

| Split | Total Images | Classes | Min Class | Max Class | Mean | Imbalance Ratio |
|-------|-------------|---------|-----------|-----------|------|-----------------|
| Train | 2,159 | 13 | 117 (ram) | 263 (dog) | 166 | 2.25× |
| Test | 86 | 13 | 4 (bird) | 10 (boar/dog/dragon/hare) | 7 | 2.50× |

> **Note:** The dataset contains a `zero` class in addition to the 12 zodiac seals — this represents a neutral/no-seal state. recommend **keeping it** as it is useful for the live-feed inference stage (detecting when no seal is being performed).

---

## 2) Issues with Kaggle Dataset

#### Not enough data / bad class representation
The test split has only **86 images total** — as few as **4 for `bird`** => a single misclassification swings per-class accuracy by up to **25%**, making it statistically meaningless as an evaluation set.

**Decision**: Merge both folders into a single pool and re-split ourselves using a principled, clip-grouped strategy (see Section 4).


#### Not diverse

it was recorded by **very few people under similar conditions**. A model trained only on it will overfit to those specific hands, backgrounds, and lighting — and fail on our live webcam demo.

Self-recorded data will serve as **domain shift coverage**: different hands, skin tones, lighting setups, and backgrounds that bridge the gap between the Kaggle images and real webcam input.

---

## 3. Target & Deficit Analysis

We set a target of **250 images per class** after merging Kaggle train + test. This is achievable and sufficient for stable SVM and CNN baselines.

| Class | Kaggle (merged) | Target | Deficit | Priority |
|-------|----------------|--------|---------|----------|
| ram | 122 | 250 | 128 | 🔴 HIGH |
| rat | 130 | 250 | 120 | 🔴 HIGH |
| monkey | 141 | 250 | 109 | 🔴 HIGH |
| snake | 151 | 250 | 99 | 🟠 MED |
| dragon | 156 | 250 | 94 | 🟠 MED |
| horse | 162 | 250 | 88 | 🟠 MED |
| tiger | 172 | 250 | 78 | 🟠 MED |
| ox | 176 | 250 | 74 | 🟠 MED |
| boar | 182 | 250 | 68 | 🟠 MED |
| hare | 184 | 250 | 66 | 🟠 MED |
| bird | 192 | 250 | 58 | 🟡 LOW |
| zero | 204 | 250 | 46 | 🟡 LOW |
| dog | 273 | 250 | −23 | ✅ Already over target |

**Summary of what we need to record:**

| Priority | Classes | Images Needed |
|----------|---------|--------------|
| 🔴 HIGH | ram, rat, monkey | 357 |
| 🟠 MED | snake, dragon, horse, tiger, ox, boar, hare | 567 |
| 🟡 LOW | bird, zero | 104 |
| **Total** | 12 / 13 classes | **1,028** |

> `dog` is the only class already above target. It will be capped at 250 by random undersampling, or left as-is — minor imbalance at that scale is acceptable.

---

## 4. Recording Plan

### Per-person workload
With **5 team members** contributing equally:

| Metric | Value |
|--------|-------|
| Images per person needed | ~206 |
| Clip length | 8 seconds |
| Extraction FPS | 5fps |
| Trim per end | 2 seconds |
| Usable frames per clip | ~20 |
| Clips needed per person | ~11 |
| Clips per seal per person | ~1 (record 2 for safety margin) |

### Why we trim
The first and last 2 seconds of each clip capture the transition **into and out of** the seal position. Extracting those frames would produce mislabelled training data — blurry, mid-motion hands labelled as a specific seal. We trim them at the extraction stage; cleaner source footage means less waste.

### Recording priorities
Focus recording effort in this order:

1. **🔴 ram, rat, monkey** — these are the thinnest classes; each needs ~36–43 images per person
2. **🟠 snake, dragon, horse, tiger, ox, boar, hare** — 7 classes, ~16–18 images per person each
3. **🟡 bird, zero** — augmentation can cover most of the remaining gap here

---

## 5. Process After Recording

Once all clips are collected:

1. **Verify clips**: spot-check each person's clips for correct seal, framing, no transitions, correct filename
2. **Extract frames**: run extraction script at 5fps with 2s trim on each end
3. **Merge**: combine Kaggle (train + test) + self-recorded into one pool per class
4. **Re-audit**: rerun the analysis script on the merged pool to confirm counts
5. **Split**: 70% train / 15% val / 15% test, grouped by clip ID so frames from the same clip never span splits
6. **Augment**: apply augmentation to the training split only (flips, colour jitter, affine transforms) to cover any remaining deficit
7. **Freeze the test split**: it does not get touched again until final evaluation

---

## 6. Key Decisions Summary

| Decision | Choice | Reason |
|----------|--------|--------|
| Use Kaggle test split as-is | No | Too small (86 images), statistically unreliable |
| Merge Kaggle train + test | Yes | Maximises usable Kaggle data |
| Keep `zero` class | Yes | Needed for live-feed no-seal detection |
| Cap `dog` at 250 | Optional | Minor overrepresentation, low impact either way |
| Augment before splitting | No | Causes data leakage between train and val |
| Split by clip ID, not by frame | Yes | Prevents near-duplicate frames across splits |
| Everyone records all 12 seals | Yes | Maximises person-level diversity per class |