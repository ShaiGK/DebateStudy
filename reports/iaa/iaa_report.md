# Inter-annotator agreement report

- Human annotations: **20** debates
- Claude annotations: **20** debates
- Overlap (used for IAA): **20** debates

## Overall judgment (Pro / Con / Tie) — stated

- n = 20
- Raw accuracy: **0.750**  CI [0.550, 0.900]
- Cohen's κ (3-class): **0.590**  CI [0.310, 0.844]
- Gwet's AC1 (3-class): **0.644**  CI [0.350, 0.876]

**Confusion matrix (rows = human, cols = Claude):**

| human \ Claude | Pro | Con | Tie | total |
|---|---|---|---|---|
| **Pro** | 6 | 0 | 0 | 6 |
| **Con** | 2 | 8 | 2 | 12 |
| **Tie** | 1 | 0 | 1 | 2 |
| **total** | 9 | 8 | 3 | 20 |

**Per-class (human as reference):**

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| Pro | 0.667 | 1.000 | 0.800 | 6 |
| Con | 1.000 | 0.667 | 0.800 | 12 |
| Tie | 0.333 | 0.500 | 0.400 | 2 |

### 2-class variant (drop Tie debates)

- n = 16
- Accuracy: 0.875
- Cohen's κ: 0.750
- Gwet's AC1: 0.754

## Overall judgment — derived from mean sub-scores

_Each annotator's overall judgment redefined as 'whichever side has the higher mean across the 5 sub-dims'. A sanity check that sub-scores line up with the stated overall._

- n = 20
- Accuracy: **0.750**  CI [0.550, 0.900]
- Cohen's κ (3-class): **0.533**  CI [0.231, 0.810]

## Internal consistency (stated vs. derived)

_How often each annotator's stated overall judgment agrees with the one computed from the mean of their own sub-scores._

| annotator | n | accuracy | Cohen's κ |
|---|---|---|---|
| human | 20 | 0.850 | 0.694 |
| claude | 20 | 0.900 | 0.830 |

## Per-dimension agreement

**Pooled across the four 1–5 dimensions (160 cell observations; concession_and_common_ground excluded — see note below):**

- weighted κ (quadratic): **0.476**
- MAE: 0.613, RMSE: 0.851
- exact match: 0.444, within-1: 0.944
- pooled bias (Claude − human): -0.163

**concession_and_common_ground (1–3 scale, 40 cell observations, reported separately):**

- weighted κ (quadratic): **0.714**
- MAE: 0.125, RMSE: 0.354
- exact match: 0.875, within-1: 1.000
- bias (Claude − human): -0.025

| side | dimension | n | wκ | AC1 | α (ord) | Spearman | MAE | bias | exact | ≤1 | ≤2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pro | acknowledgment | 20 | 0.349 | 0.350 | 0.353 | 0.373 | 0.600 | -0.200 | 0.450 | 0.950 | 1.000 |
| pro | accuracy_of_representation | 20 | 0.454 | 0.218 | 0.451 | 0.552 | 0.700 | 0.300 | 0.350 | 0.950 | 1.000 |
| pro | responsiveness | 20 | 0.082 | 0.297 | 0.105 | 0.004 | 0.700 | 0.000 | 0.400 | 0.900 | 1.000 |
| pro | concession_and_common_ground | 20 | 0.706 | 0.814 | 0.707 | 0.784 | 0.150 | -0.150 | 0.850 | 1.000 | — |
| pro | respectful_engagement | 20 | 0.587 | 0.145 | 0.585 | 0.645 | 0.750 | -0.350 | 0.300 | 0.950 | 1.000 |
| con | acknowledgment | 20 | 0.421 | 0.528 | 0.429 | 0.429 | 0.450 | -0.150 | 0.600 | 0.950 | 1.000 |
| con | accuracy_of_representation | 20 | 0.192 | 0.551 | 0.179 | 0.204 | 0.400 | -0.200 | 0.600 | 1.000 | 1.000 |
| con | responsiveness | 20 | 0.529 | 0.215 | 0.487 | 0.700 | 0.700 | -0.600 | 0.350 | 0.950 | 1.000 |
| con | concession_and_common_ground | 20 | 0.737 | 0.877 | 0.740 | 0.764 | 0.100 | 0.100 | 0.900 | 1.000 | — |
| con | respectful_engagement | 20 | 0.502 | 0.395 | 0.512 | 0.518 | 0.600 | -0.100 | 0.500 | 0.900 | 1.000 |

### How to read this

- **weighted κ (quadratic)** is the primary metric for ordinal rubric scores. Rough Landis & Koch (1977) guide: 0.0–0.2 slight, 0.21–0.40 fair, 0.41–0.60 moderate, 0.61–0.80 substantial, 0.81–1.0 almost perfect.
- **Gwet's AC1** is reported alongside Cohen's κ because κ is deflated when one class dominates (the 'high agreement, low κ' paradox). AC1 is more stable under skewed marginals.
- **Krippendorff's α (ordinal)** is reported alongside weighted κ as a second ordinal-agreement metric. When a dimension has a compressed score range (e.g., only 3s and 4s are used), weighted κ can be deflated even though annotators largely agree; α and AC1 are less sensitive to this failure mode. Report all three in the thesis so the reader can see whether a low κ is genuine disagreement or restricted-range deflation.
- **bias (Claude − human)** — positive means Claude rates higher on average than you do on that cell. Large per-dimension biases are a signal that the rubric wording or the prompt needs tightening for that dimension.
- **exact / ≤1 / ≤2** is the forgiving-agreement ladder. On a 1–5 scale, within-1 ≈ 0.80+ is usually considered strong for ordinal rubrics.
- **Pooled metrics exclude concession_and_common_ground** because that dimension uses a 1–3 scale while the other four use 1–5. Mixing scales in a pooled weighted κ would distort the weight matrix. Concession metrics are reported separately in the pooled-summary section above.
- All CIs are non-parametric bootstrap percentile intervals resampled over debates.

