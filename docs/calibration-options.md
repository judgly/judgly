# Calibration options: the H2 head and the per-type temperature

Each built-in pack ships two ways of turning the model's answer-letter probabilities into
calibrated ones, per format (general and stance):

- **H2**, the head released in judgly 0.1.0: a temperature, a bias per letter and a small
  correction to each letter's output row of the model, fitted per question type and applied to
  each option order's letter logits before the orders are averaged
  ([methods.md](methods.md#the-heads)).
- **The per-type temperature** ("temperature" in the API, "Ttype" in the study files): one number
  per question type (choice, yes/no, score), applied once to the probabilities after they are
  averaged over the option orders.

A pack names one of them as its default per format. The defaults follow a pre-registered
comparison of the two on data that nothing in judgly had been fitted, tuned or chosen on:

| pack | general | stance |
|---|---|---|
| `gemma4-12b-q8` | **H2** (the temperature was not confirmed) | **temperature** (confirmed) |
| `qwen3-4b-q8` | **temperature** (confirmed) | **temperature** (confirmed) |

```python
from judgly import Engine

engine = Engine.load("qwen3-4b-q8")                               # the pack's default per format
engine = Engine.load("qwen3-4b-q8", calibration="h2")             # H2 for every format
engine = Engine.load("qwen3-4b-q8", calibration="temperature")    # the temperature for every format
engine = Engine.load("qwen3-4b-q8", calibration="raw")            # no calibration
print(engine.calibration)                                         # {'*': 'temperature', 'stance': 'temperature'}
```

This page says what the temperature is, what data it was fitted on, how it was compared with
H2, and where every number comes from. It records the order in which things were done where
that order matters for how much the numbers can be trusted.

## Contents

- [What the temperature does](#what-the-temperature-does)
- [How it was fitted](#how-it-was-fitted)
- [How the temperature came to be compared with H2](#how-the-temperature-came-to-be-compared-with-h2)
- [The pre-registered confirmation](#the-pre-registered-confirmation)
- [Both options on every tier](#both-options-on-every-tier)
- [What the results do and do not show](#what-the-results-do-and-do-not-show)
- [Reproducing the numbers](#reproducing-the-numbers)

## What the temperature does

Both options start from the same readout of the model: for each order in which the options are
shown, the model's scores for the answer letters. They differ in where the calibration acts.

```text
                   question, shown in up to 4 option orders
                                   |
                                   v
                model: letter scores z, one set per order
                                   |
            +----------------------+----------------------+
            | H2                                          | per-type temperature
            v                                             v
   per order: u = a * dot(w + d, h) + b          per order: softmax(z),
   (fitted temperature, letter biases,           mapped back to the options
   correction to the letter rows),                        |
   softmax(u), mapped back to the options                 v
            |                                    mean over orders = p (raw)
            v                                             |
   mean over orders                                       v
            |                                    p_T proportional to
            |                                    exp(log p / T[question type])
            v                                             v
   calibrated probabilities;                     calibrated probabilities;
   the top answer can change                     the top answer is raw's
```

For a question with K options, the engine reads the letter probabilities under up to four
cyclic orders of the options (all K orders for yes/no; one order, the natural one, for score
questions), maps each order's probabilities back to the options and averages them: that average
is the raw readout p, the same one the `raw` condition reports. The temperature T of the
question's type is then applied to p:

    p_T[k] = exp(log(max(p[k], 1e-12)) / T) / sum_j exp(log(max(p[j], 1e-12)) / T)

T > 1 flattens p; T = 1 leaves it unchanged. The transformation is monotone for each question,
so it never changes which option is on top: **the temperature's accuracy is always the raw
readout's.** H2, by contrast, acts on each order's letter logits, can change the top answer, and
its accuracy differs from raw.

The shipped temperatures, per question type:

| pack | format | choice | yes/no | score |
|---|---|---|---|---|
| gemma4-12b-q8 | general | 3.461 | 7.491 | 6.538 |
| gemma4-12b-q8 | stance | 7.151 | (1, no data) | (1, no data) |
| qwen3-4b-q8 | general | 6.846 | 13.36 | 24.22 |
| qwen3-4b-q8 | stance | 12.391 | (1, no data) | (1, no data) |

The stance format has only choice questions, so its yes/no and score temperatures are 1 (no
change), recorded as a fallback in the sidecar. A large temperature means that the raw readout
was very overconfident on that question type in the fit data (Qwen3-4B's score questions most of
all).

In the engine the temperature is a head file of its own type (`heads/temperature.bin`,
`heads/temperature-stance.bin`: the H1/H2 file layout with head type 3 and one float64 per
question type), with the same guardrails as H2: the SHA-256 of the model and template it was
fitted for is checked at load, the engine settings it was fitted under are recorded in its
sidecar (`*.bin.json`) and enforced, and a temperature outside [0.05, 100] is refused.
`s1_combine` (and the C API's answer builder) read every order without a head and apply the
temperature to the mean. The response's `rotation_spread` is then the spread of the raw
per-order probabilities, before the temperature.

## How it was fitted

**Data.** The fit tier's train split, the same items H2 was fitted on
([methods.md](methods.md#data-and-tiers)): a 70/15/15 split into train, validation and test by a
hash of the passage or of a recorded group, so that items sharing one stay in one split.

| format | train items (fit) | validation items (verdict only) | test items (in-distribution, reported) |
|---|---|---|---|
| general | 6,300: 4,156 choice, 1,093 yes/no, 1,051 score | 1,350: 892, 232, 226 | 1,350 |
| stance | 14,783 choice | 3,072 choice | 3,145 |

The sources of the train split are listed with their licences in each calibration record
(`calibration/<format>.json`, `head.fitted_on`); both options are fitted on the same items and
carry the same licence (Apache-2.0 general, CC BY-SA 4.0 stance).

**Objective.** Per question type, the T in [0.05, 100] that minimises the mean log loss
-log p_T[correct] over the train items, each item counted once and unweighted (H2, by contrast,
is fitted per option order and weights score levels equally). The loss is convex in 1/T; the
fitter (`s1-train --head temperature`, `csrc/s1_temperature.c`) finds the minimum by bisection
on its derivative to machine precision and rounds it to three decimals. The validation split is
used only for the same verdict as H2's: a type whose temperature is no better than the raw
readout on validation, or whose output hardly varies, falls back to T = 1 (none did).

**H2, for comparison** ([methods.md](methods.md#the-heads)): fitted on the same train items per
question type, by L-BFGS on the mean log loss over the option orders (score levels weighted
equally), from the H1 solution; its penalty on the row corrections is chosen from a short grid by
validation log loss, with early stopping on validation. So the validation split takes part in
fitting H2 but only in the fallback verdict for the temperature.

**Relation to the confirmed values.** The temperatures in the table above are exactly the values
frozen in `docs/results/calibration-study/confirmation/temperatures.json` before the
confirmation. Those were fitted by the exploratory script `tricks.py` (Nelder-Mead in log T,
from the same train items) and rounded to three decimals. Refitted by `s1-train`, all eight
unrounded optima round to the same three decimals (for example 6.84589 for Qwen3-4B choice,
frozen 6.846; 13.35999 for Qwen3-4B yes/no, frozen 13.36); the shipped files hold the rounded
values, and `docs/tools/confirmation_check.py` checks them against `temperatures.json`.

## How the temperature came to be compared with H2

The temperature is the result of looking at data, so the order of events is part of what the
numbers mean. All of this happened after judgly 0.1.0 was released with H2, using only the
readouts cached by the release runs (no model was run):

1. **Exploratory analysis 0** (`exploratory-0-combine`): temperature-only variants and a
   combination (log-linear pool) of the two packs, with parameters fitted on the fit tier's test
   split, reported on dev, final and final-seen. A temperature per question type (fitted there on
   the test split) had a lower log loss than H2 in 7 of the 12 pack, format and tier cases. Pooling Gemma 4 12B with Qwen3-4B was never more than 0.0022 more
   accurate than Gemma 4 12B alone, and its best variant's log loss was within 0.030 of the best
   Gemma-only variant's (lower in four of six cases, higher in two); it would need both models at
   run time and was not pursued.
2. **Exploratory analysis 1** (`exploratory-1-temperature`): one temperature for everything,
   the per-type temperature, and the per-type temperature with a bias per option position, all
   fitted on the train split and compared with H2. The rule written into the script before any
   result was seen, lowest log loss on the validation split, chose **H2 in all four pack and
   format cases** (validation is in-distribution, and H2 is fitted to that distribution). On the
   held-out tiers the per-type temperature looked better: over the 18 pack, format and tier
   comparisons (dev, final, final-flagged, final-seen, bench), its accuracy was at least H2's in
   15, and its log loss, ECE and Brier score were each lower in 11; in 6 of the 18 the 95%
   interval of the paired difference showed it worse than H2 on at least one of accuracy, ECE and
   Brier score (among them the stance final tier of both packs, Check-COVID).
3. **Exploratory analysis 2** (`exploratory-2-post-processing`, protocol frozen before it ran):
   four refinements on top of the per-type temperature (a temperature that grows with the
   disagreement between option orders, Platt scaling, isotonic regression, histogram binning),
   fitted on the train split and compared on dev. The protocol's rule (lowest dev log loss, ties
   within 0.002 to fewer parameters) chose Platt scaling, the per-type temperature, H2 and
   isotonic regression in the four cases. On the held-out tiers none was better than the plain
   per-type temperature with any consistency: over 18 comparisons, Platt scaling had a lower log
   loss (95% interval of the difference below zero) in 9 and a higher one in 6, and the other
   refinements were more often worse than better. Three bugs in the analysis script were fixed
   during this run, without changing the protocol. A sub-study simulated calibration on a user's
   own data (below).
4. **The decision to confirm.** The per-type temperature was chosen for confirmation because it
   was the simplest variant and did as well as the others across these analyses. That choice was
   made after reading the held-out tiers, including the fresh final tier, which by then had been
   read by the release run and by all three analyses. The final tier therefore cannot confirm the
   temperature, and a new, untouched tier was built for that purpose.

Each analysis has a README with its data, its dates and how to rerun it
([results/calibration-study](results/calibration-study/README.md)); rerun from the repository,
all four reproduced their committed output byte for byte.

## The pre-registered confirmation

The protocol ([CONFIRM.md](results/calibration-study/confirmation/CONFIRM.md)) was written and
frozen on 2026-09-29 before any confirmation data existed; Amendment 1 was added after the tier
was built and reviewed and before any model read it. `CONFIRM.sha256` records the SHA-256 and
freeze time of the protocol, of `temperatures.json` and of the scorer `score_confirm.py` (frozen
after a smoke test on the 0.1.0 dev tier only).

**Claim.** On task families that nothing in judgly was fitted, tuned or chosen on, the per-type
temperature is at least as good as the 0.1.0 H2 head: no loss of accuracy and no worse
calibration. Nothing is refitted: the temperatures of `temperatures.json`, the heads of the 0.1.0
release (commit 034d5e6), the release engine settings (up to four option orders, no content-free
pass).

**Data: the confirm tier**, built after every other tier (commits ad8c91a and dbf1a9f), cleaned
against every other tier (exact matches, shared sentences, shared passages, near duplicates) and
frozen by SHA-256 in `data/tiers-confirm.sha256`:

- general, 2,500 items in five families never used in this project, 500 each: kinship (CLUTRR,
  choice), code_outcome (CodeMMLU execution prediction, choice), spatial (SpartQA yes/no),
  argument_quality (IBM Argument Quality 30k, score, three levels) and humour (Humicroedit,
  score, three levels); 1,932 resampling groups (items that share a text, such as an argument
  topic or an original headline, are resampled together);
- stance, 1,780 claim and abstract pairs from the ClimateCheck test split (reserved in 0.1.0 for
  later use), resampled in 70 linked groups (claims joined by a shared abstract, because such
  items are not independent; the largest group holds 837 items).

The review of the tier flagged three general families before any model read them:
argument_quality and humour as possible relatives of used rating families, and code_outcome as
having a language shortcut (the verdict goes with the programming language). By Amendment 1 the
endpoints are judged on all five families, and the result is also reported with each of these
three left out, without a criterion. Each pack extracted the tier once (`s1-features`, release
settings) and the frozen scorer read the per-item output once.

**Criteria**, per pack and format, on the paired difference temperature minus H2 over the same
items, 95% percentile interval from 1,000 bootstrap resamples of the groups (seed 20260929):
accuracy, lower bound above -0.01; Brier score, upper bound below +0.01; ECE, upper bound below
+0.02. A case is confirmed when all three hold, and only confirmed cases switch.

**Result: confirmed in 3 of 4 cases.**

| pack | format | n (groups) | accuracy T / H2 | ECE T / H2 | Brier T / H2 | difference T - H2: accuracy, ECE, Brier [95% interval] | verdict |
|---|---|---|---|---|---|---|---|
| gemma4-12b-q8 | general | 2,500 (1,932) | 0.509 / 0.509 | 0.087 / 0.051 | 0.579 / 0.562 | +0.000 [-0.014, +0.015]; +0.037 [+0.015, +0.048]; +0.017 [+0.010, +0.024] | not confirmed (all three missed) |
| gemma4-12b-q8 | stance | 1,780 (70) | 0.655 / 0.629 | 0.052 / 0.078 | 0.476 / 0.490 | +0.026 [+0.011, +0.049]; -0.026 [-0.045, +0.007]; -0.014 [-0.044, -0.001] | confirmed |
| qwen3-4b-q8 | general | 2,500 (1,932) | 0.484 / 0.463 | 0.047 / 0.076 | 0.597 / 0.624 | +0.022 [+0.008, +0.036]; -0.029 [-0.049, -0.014]; -0.027 [-0.035, -0.020] | confirmed |
| qwen3-4b-q8 | stance | 1,780 (70) | 0.568 / 0.526 | 0.106 / 0.183 | 0.581 / 0.612 | +0.042 [+0.023, +0.090]; -0.078 [-0.097, -0.054]; -0.031 [-0.065, -0.018] | confirmed |

The table reports the scorer's output (`result-confirm.txt`), which rounds to four decimals and
then prints three; the calibration records round once, so Gemma 4 12B general H2 ECE (0.05048)
reads 0.051 here and 0.050 in the tables below. At four decimals every point value of the scorer
equals the record's.

Log loss, reported without a criterion: Gemma general 0.959 against 0.933 (+0.026 [+0.017,
+0.037], H2 better); Gemma stance 0.823 against 0.812 (+0.012 [-0.036, +0.033]); Qwen general
1.009 against 1.041 (-0.032 [-0.043, -0.023]); Qwen stance 0.966 against 0.989 (-0.023 [-0.070,
-0.001]). The raw readout was far worse on every case (ECE 0.30 to 0.41).

Secondary analyses, without a criterion: for Qwen3-4B general, all three criteria also held with
each flagged family left out in turn; for Gemma 4 12B general they failed in each. For stance,
resampling the 175 claims instead of the 70 linked groups met all three criteria for both packs.
Per family (general), the temperature was less accurate than H2 on spatial for Gemma (0.536
against 0.580) and more accurate on spatial for Qwen (0.612 against 0.488); the full per-family
table is in `result-confirm.txt`.

So Gemma 4 12B keeps H2 for general questions and the other three cases use the temperature by
default. The Gemma general case is not evidence that H2 is better in general: its accuracy was
equal, H2 was better calibrated on this tier, and the confirmation was not designed to test that
direction.

## Both options on every tier

The same numbers for every tier, from the calibration records (`docs/results/<pack>/<format>/
record.json` and `tables.md`; per-family numbers, reliability bins, selective accuracy and the
stance confusion matrices are there). Brackets are 95% percentile bootstrap intervals from 1,000
resamples, of groups on final, final-flagged, confirm and bench and of items elsewhere; these are
intervals of each condition on its own, not of the paired difference. The confirm-tier numbers
are recomputed from the same readouts with the same frozen head and temperatures; their point
values equal the confirmation's to the four decimals it reports, and their intervals differ from
the confirmation's paired ones.

How often each tier had been read before these numbers: **test** and **dev** were read during
development and by every analysis; **final**, **final-flagged**, **final-seen** and **bench** by
the 0.1.0 release run and then by the three exploratory analyses; **confirm** once, by the
confirmation. Only the confirm tier is untouched for the comparison of the two options.

**Gemma 4 12B, general**

| tier | items (groups) | condition | accuracy | log loss | Brier | ECE |
|---|---|---|---|---|---|---|
| test | 1,350 | raw | 0.724 [0.701, 0.750] | 1.740 [1.555, 1.929] | 0.472 [0.429, 0.512] | 0.196 [0.172, 0.218] |
| | | h2 | 0.760 [0.737, 0.782] | 0.661 [0.616, 0.705] | 0.334 [0.311, 0.354] | 0.054 [0.045, 0.075] |
| | | temperature | 0.724 [0.701, 0.750] | 0.733 [0.684, 0.780] | 0.377 [0.352, 0.401] | 0.034 [0.023, 0.058] |
| dev | 4,500 | raw | 0.772 [0.759, 0.783] | 1.687 [1.581, 1.808] | 0.414 [0.392, 0.436] | 0.187 [0.175, 0.199] |
| | | h2 | 0.764 [0.752, 0.775] | 0.654 [0.624, 0.685] | 0.357 [0.342, 0.372] | 0.080 [0.070, 0.092] |
| | | temperature | 0.772 [0.759, 0.783] | 0.604 [0.578, 0.635] | 0.340 [0.324, 0.356] | 0.062 [0.053, 0.076] |
| final | 7,879 (6,803) | raw | 0.760 [0.751, 0.770] | 1.553 [1.478, 1.636] | 0.431 [0.415, 0.448] | 0.194 [0.186, 0.204] |
| | | h2 | 0.737 [0.726, 0.746] | 0.631 [0.613, 0.651] | 0.348 [0.338, 0.359] | 0.020 [0.015, 0.031] |
| | | temperature | 0.760 [0.752, 0.770] | 0.596 [0.578, 0.613] | 0.324 [0.314, 0.334] | 0.017 [0.014, 0.027] |
| final-flagged | 1,000 (1,000) | raw | 0.519 [0.490, 0.549] | 3.373 [3.120, 3.612] | 0.889 [0.833, 0.945] | 0.423 [0.392, 0.452] |
| | | h2 | 0.512 [0.481, 0.544] | 1.054 [1.007, 1.100] | 0.621 [0.592, 0.646] | 0.106 [0.080, 0.138] |
| | | temperature | 0.519 [0.489, 0.548] | 1.006 [0.964, 1.049] | 0.603 [0.578, 0.628] | 0.104 [0.078, 0.136] |
| confirm | 2,500 (1,932) | raw | 0.509 [0.489, 0.531] | 2.238 [2.099, 2.361] | 0.834 [0.796, 0.868] | 0.373 [0.349, 0.394] |
| | | h2 | 0.509 [0.489, 0.530] | 0.933 [0.909, 0.956] | 0.562 [0.547, 0.577] | 0.050 [0.042, 0.072] |
| | | temperature | 0.509 [0.490, 0.530] | 0.959 [0.933, 0.982] | 0.579 [0.563, 0.595] | 0.087 [0.070, 0.106] |
| final-seen | 10,300 | raw | 0.714 [0.705, 0.723] | 1.643 [1.580, 1.708] | 0.460 [0.446, 0.474] | 0.180 [0.172, 0.189] |
| | | h2 | 0.713 [0.703, 0.722] | 0.799 [0.779, 0.818] | 0.386 [0.377, 0.396] | 0.026 [0.021, 0.034] |
| | | temperature | 0.714 [0.705, 0.723] | 0.806 [0.786, 0.827] | 0.390 [0.381, 0.400] | 0.016 [0.013, 0.025] |
| bench | 2,231 (595) | raw | 0.715 [0.694, 0.735] | 1.841 [1.673, 2.016] | 0.509 [0.473, 0.548] | 0.239 [0.220, 0.259] |
| | | h2 | 0.714 [0.696, 0.735] | 0.690 [0.658, 0.721] | 0.382 [0.363, 0.401] | 0.029 [0.021, 0.048] |
| | | temperature | 0.715 [0.695, 0.735] | 0.702 [0.671, 0.733] | 0.389 [0.370, 0.409] | 0.027 [0.018, 0.047] |

**Gemma 4 12B, stance**

| tier | items (groups) | condition | accuracy | log loss | Brier | ECE |
|---|---|---|---|---|---|---|
| test | 3,145 | raw | 0.721 [0.703, 0.735] | 2.786 [2.619, 2.976] | 0.521 [0.493, 0.553] | 0.247 [0.232, 0.263] |
| | | h2 | 0.786 [0.771, 0.800] | 0.550 [0.525, 0.580] | 0.305 [0.291, 0.322] | 0.021 [0.015, 0.036] |
| | | temperature | 0.721 [0.703, 0.735] | 0.715 [0.691, 0.743] | 0.403 [0.387, 0.422] | 0.060 [0.047, 0.075] |
| dev | 2,100 | raw | 0.654 [0.635, 0.674] | 3.607 [3.336, 3.867] | 0.646 [0.608, 0.683] | 0.306 [0.285, 0.326] |
| | | h2 | 0.624 [0.605, 0.647] | 1.029 [0.974, 1.082] | 0.544 [0.517, 0.570] | 0.122 [0.106, 0.143] |
| | | temperature | 0.654 [0.635, 0.674] | 0.860 [0.823, 0.895] | 0.498 [0.473, 0.522] | 0.069 [0.051, 0.091] |
| final (Check-COVID) | 1,343 (315) | raw | 0.827 [0.806, 0.848] | 1.603 [1.370, 1.855] | 0.310 [0.272, 0.348] | 0.142 [0.123, 0.162] |
| | | h2 | 0.812 [0.790, 0.836] | 0.509 [0.472, 0.550] | 0.287 [0.263, 0.313] | 0.048 [0.034, 0.072] |
| | | temperature | 0.827 [0.807, 0.848] | 0.536 [0.504, 0.571] | 0.280 [0.257, 0.303] | 0.087 [0.068, 0.108] |
| final-flagged (HealthFC) | 749 (749) | raw | 0.750 [0.722, 0.782] | 1.978 [1.665, 2.262] | 0.441 [0.388, 0.491] | 0.201 [0.174, 0.229] |
| | | h2 | 0.634 [0.597, 0.668] | 0.809 [0.762, 0.856] | 0.500 [0.467, 0.533] | 0.123 [0.093, 0.158] |
| | | temperature | 0.750 [0.717, 0.782] | 0.670 [0.625, 0.720] | 0.370 [0.339, 0.404] | 0.069 [0.057, 0.104] |
| confirm (ClimateCheck) | 1,780 (70) | raw | 0.655 [0.621, 0.705] | 3.223 [2.574, 3.499] | 0.631 [0.532, 0.688] | 0.300 [0.248, 0.331] |
| | | h2 | 0.629 [0.587, 0.671] | 0.812 [0.739, 0.869] | 0.491 [0.443, 0.529] | 0.078 [0.046, 0.102] |
| | | temperature | 0.655 [0.617, 0.704] | 0.823 [0.729, 0.876] | 0.476 [0.414, 0.513] | 0.052 [0.029, 0.080] |
| final-seen (HealthVer) | 2,100 | raw | 0.632 [0.610, 0.652] | 3.630 [3.404, 3.870] | 0.678 [0.640, 0.716] | 0.320 [0.302, 0.342] |
| | | h2 | 0.626 [0.605, 0.647] | 0.889 [0.851, 0.924] | 0.507 [0.485, 0.529] | 0.067 [0.051, 0.089] |
| | | temperature | 0.632 [0.610, 0.652] | 0.872 [0.838, 0.905] | 0.509 [0.487, 0.531] | 0.076 [0.060, 0.098] |

**Qwen3-4B, general**

| tier | items (groups) | condition | accuracy | log loss | Brier | ECE |
|---|---|---|---|---|---|---|
| test | 1,350 | raw | 0.672 [0.647, 0.697] | 4.085 [3.691, 4.482] | 0.586 [0.541, 0.629] | 0.245 [0.223, 0.269] |
| | | h2 | 0.697 [0.675, 0.723] | 0.776 [0.727, 0.819] | 0.389 [0.365, 0.411] | 0.040 [0.031, 0.064] |
| | | temperature | 0.672 [0.647, 0.697] | 0.889 [0.839, 0.937] | 0.454 [0.427, 0.477] | 0.062 [0.046, 0.088] |
| dev | 4,500 | raw | 0.730 [0.718, 0.743] | 3.287 [3.076, 3.473] | 0.492 [0.468, 0.514] | 0.220 [0.208, 0.234] |
| | | h2 | 0.731 [0.718, 0.745] | 0.636 [0.611, 0.659] | 0.380 [0.364, 0.395] | 0.055 [0.049, 0.070] |
| | | temperature | 0.730 [0.718, 0.743] | 0.656 [0.627, 0.681] | 0.385 [0.368, 0.399] | 0.068 [0.059, 0.082] |
| final | 7,879 (6,803) | raw | 0.656 [0.644, 0.667] | 3.474 [3.330, 3.639] | 0.602 [0.583, 0.622] | 0.268 [0.259, 0.279] |
| | | h2 | 0.639 [0.628, 0.650] | 0.797 [0.779, 0.815] | 0.450 [0.440, 0.460] | 0.030 [0.025, 0.040] |
| | | temperature | 0.656 [0.646, 0.668] | 0.783 [0.765, 0.802] | 0.440 [0.429, 0.450] | 0.034 [0.027, 0.042] |
| final-flagged | 1,000 (1,000) | raw | 0.492 [0.462, 0.523] | 5.688 [5.214, 6.145] | 0.935 [0.877, 0.991] | 0.442 [0.409, 0.472] |
| | | h2 | 0.487 [0.456, 0.519] | 1.032 [1.002, 1.064] | 0.620 [0.601, 0.640] | 0.046 [0.030, 0.080] |
| | | temperature | 0.492 [0.461, 0.523] | 1.025 [1.005, 1.046] | 0.617 [0.605, 0.631] | 0.037 [0.024, 0.068] |
| confirm | 2,500 (1,932) | raw | 0.484 [0.463, 0.507] | 4.674 [4.277, 5.083] | 0.888 [0.845, 0.928] | 0.406 [0.380, 0.429] |
| | | h2 | 0.463 [0.441, 0.483] | 1.041 [1.017, 1.065] | 0.624 [0.610, 0.639] | 0.076 [0.062, 0.100] |
| | | temperature | 0.484 [0.464, 0.506] | 1.009 [0.986, 1.031] | 0.597 [0.583, 0.610] | 0.047 [0.031, 0.068] |
| final-seen | 10,300 | raw | 0.649 [0.640, 0.658] | 3.559 [3.426, 3.687] | 0.569 [0.554, 0.584] | 0.231 [0.223, 0.240] |
| | | h2 | 0.654 [0.645, 0.663] | 0.932 [0.914, 0.950] | 0.457 [0.448, 0.465] | 0.019 [0.016, 0.030] |
| | | temperature | 0.649 [0.640, 0.658] | 0.969 [0.950, 0.988] | 0.470 [0.460, 0.479] | 0.037 [0.031, 0.046] |
| bench | 2,231 (595) | raw | 0.588 [0.564, 0.611] | 4.371 [4.059, 4.699] | 0.734 [0.692, 0.777] | 0.345 [0.322, 0.367] |
| | | h2 | 0.601 [0.578, 0.622] | 0.962 [0.925, 1.000] | 0.549 [0.528, 0.572] | 0.114 [0.099, 0.132] |
| | | temperature | 0.588 [0.564, 0.611] | 0.963 [0.933, 0.996] | 0.558 [0.537, 0.579] | 0.128 [0.113, 0.147] |

**Qwen3-4B, stance**

| tier | items (groups) | condition | accuracy | log loss | Brier | ECE |
|---|---|---|---|---|---|---|
| test | 3,145 | raw | 0.680 [0.664, 0.697] | 4.786 [4.502, 5.102] | 0.580 [0.550, 0.611] | 0.276 [0.261, 0.294] |
| | | h2 | 0.755 [0.739, 0.770] | 0.624 [0.598, 0.651] | 0.351 [0.335, 0.367] | 0.017 [0.012, 0.033] |
| | | temperature | 0.680 [0.664, 0.697] | 0.774 [0.749, 0.798] | 0.446 [0.430, 0.463] | 0.013 [0.010, 0.032] |
| dev | 2,100 | raw | 0.549 [0.527, 0.570] | 7.057 [6.665, 7.485] | 0.829 [0.788, 0.870] | 0.396 [0.375, 0.419] |
| | | h2 | 0.536 [0.514, 0.560] | 1.194 [1.145, 1.247] | 0.658 [0.630, 0.687] | 0.209 [0.189, 0.232] |
| | | temperature | 0.549 [0.527, 0.570] | 1.006 [0.975, 1.041] | 0.595 [0.574, 0.618] | 0.109 [0.091, 0.132] |
| final (Check-COVID) | 1,343 (315) | raw | 0.778 [0.755, 0.801] | 3.185 [2.812, 3.589] | 0.410 [0.371, 0.453] | 0.187 [0.167, 0.212] |
| | | h2 | 0.773 [0.749, 0.795] | 0.594 [0.562, 0.628] | 0.337 [0.316, 0.359] | 0.052 [0.034, 0.077] |
| | | temperature | 0.778 [0.754, 0.802] | 0.650 [0.617, 0.682] | 0.357 [0.334, 0.380] | 0.093 [0.068, 0.116] |
| final-flagged (HealthFC) | 749 (749) | raw | 0.718 [0.684, 0.752] | 3.605 [3.081, 4.081] | 0.510 [0.450, 0.569] | 0.227 [0.196, 0.260] |
| | | h2 | 0.541 [0.506, 0.575] | 0.928 [0.877, 0.979] | 0.591 [0.556, 0.625] | 0.150 [0.130, 0.192] |
| | | temperature | 0.718 [0.684, 0.750] | 0.705 [0.663, 0.749] | 0.410 [0.379, 0.441] | 0.052 [0.031, 0.085] |
| confirm (ClimateCheck) | 1,780 (70) | raw | 0.568 [0.523, 0.608] | 6.492 [5.719, 7.182] | 0.809 [0.732, 0.892] | 0.383 [0.344, 0.423] |
| | | h2 | 0.526 [0.452, 0.557] | 0.989 [0.938, 1.083] | 0.612 [0.579, 0.682] | 0.183 [0.155, 0.228] |
| | | temperature | 0.568 [0.524, 0.605] | 0.966 [0.914, 1.038] | 0.581 [0.544, 0.631] | 0.106 [0.078, 0.153] |
| final-seen (HealthVer) | 2,100 | raw | 0.571 [0.548, 0.592] | 5.891 [5.525, 6.277] | 0.766 [0.729, 0.807] | 0.354 [0.333, 0.377] |
| | | h2 | 0.548 [0.527, 0.570] | 0.997 [0.966, 1.032] | 0.592 [0.572, 0.613] | 0.100 [0.084, 0.122] |
| | | temperature | 0.571 [0.548, 0.592] | 0.965 [0.937, 0.996] | 0.572 [0.553, 0.593] | 0.058 [0.046, 0.083] |

## What the results do and do not show

- **The temperature keeps the raw ranking.** Where H2 changed the top answer, it sometimes helped
  (in-distribution test tier, stance: +0.065 accuracy for Gemma 4 12B, +0.075 for Qwen3-4B) and
  sometimes hurt (stance final-flagged, HealthFC: -0.116 and -0.177). On held-out families the
  temperature's accuracy was equal to or higher than H2's in most tiers; on the in-distribution
  test split H2 was more accurate in all four cases.
- **Neither option is better everywhere.** On the stance final tier (Check-COVID) H2 was better
  calibrated than the temperature in both packs (ECE 0.048 and 0.052 against 0.087 and 0.093),
  although the temperature is now the stance default. On the stance confirm tier (ClimateCheck)
  the temperature was better in both. Stance calibration depends strongly on the kind of claims
  and evidence, and either option can miss a bar of 0.05 on a new source.
- **One family carries the Qwen3-4B general accuracy gain.** On spatial (SpartQA yes/no) H2
  lowered Qwen3-4B's accuracy from 0.612 to 0.488; the five general families have 500 items each,
  and the mean accuracy difference of the other four is -0.004 (post hoc arithmetic, no interval).
  The protocol specified no analysis without spatial.
- **The confirmation is one tier per format.** Five general families and one stance source, with
  70 resampling groups for stance, so the stance intervals are wide. It tests "at least as good
  as H2" by the three criteria; it does not show that the temperature is better, and a case that
  was not confirmed does not show that H2 is better.
- **The temperatures were chosen on data that had been read.** The idea of the per-type
  temperature, and the decision to test it, came from analyses that read the final tier several
  times. Only the confirm tier is free of that; the final-tier numbers for the temperature above
  are supporting evidence, not a held-out result.
- **Calibrating on your own data.** In the exploratory sub-study (`tricks.txt`, "Tdomain"),
  each held-out family's items were split in two halves by a hash of their group; a single
  temperature fitted on the first 25, 50, 100 or 200 items of one half (or all of it) was judged
  on the other half. Averaged over families, it gave a lower log loss than the shipped per-type
  temperature from about 50 to 100 items in most pack, format and tier cases, and was mixed at 25
  (for example, Gemma 4 12B general on the final tier needed about 100). This was exploratory and
  was not confirmed; [calibration.md](calibration.md#fitting-a-head-on-your-own-data) shows how to
  fit a temperature on your own labelled cases, and checking it on held-back cases of your own is
  the part that matters.

## Reproducing the numbers

Everything below runs on the CPU from files a finished pack run has left; nothing runs the model.

```bash
make tools
make calibrate MODEL=gemma4-12b-q8     # fit the temperature, evaluate raw, h2 and temperature
make calibrate MODEL=qwen3-4b-q8       # on every tier, write the records and the pack
uv run --no-project --with numpy python docs/tools/confirmation_check.py
```

`make calibrate` fits `results/<pack>/<format>/temperature.bin` with `s1-train --head
temperature` from the fit tier's cached features, checks it (`scripts/check_heads.py`), runs
`s1-eval` for raw, h2 and temperature on every tier (the confirm tier's features, extracted by
the confirmation into `results-confirm/`, are linked into `results/`), writes the per-item dumps
`items-{raw,h2,temperature}-<tier>.tsv`, the reports `{raw,h2,temperature}-<tier>.json` and the
calibration record, and assembles the pack. Every feature file is passed to make as old, so no
extraction runs; a missing one stops it. The raw and H2 dumps and record sections it writes are
byte-identical to the 0.1.0 release's (the bootstrap draws of raw and H2 on the 0.1.0 tiers use
the same random stream as in 0.1.0; every other condition and tier has its own).

The analyses themselves are rerun, unchanged, by
`uv run --no-project --with numpy --with scipy --with scikit-learn python docs/results/calibration-study/rerun.py all`,
which compares each output with the committed one.

**How the analysis outputs relate to the records.** The per-type temperature's point values in
the records (every tier, both packs, both formats) were compared with the Ttype rows of
`temperature.json` and `tricks.json` and with `result-confirm.json`, all at four decimals. The
confirmation's values (overall and per family, raw, temperature and H2) are all equal. Of the
exploratory Ttype values, all but two (analysis 1) and three (analysis 2) are equal; those differ
by 1e-4 because the exploratory scripts applied their own unrounded optimum and the shipped files
hold the temperature rounded to three decimals. Three raw log losses of Qwen3-4B differ by up to
0.006 because the exploratory scripts clip probabilities at 1e-12 and the records do not. The
analyses' printed `.txt` files round four-decimal values again to three, so some printed values
differ by 0.001 from the records. Intervals come from different bootstrap draws and differ.

`docs/tools/confirmation_check.py` checks that the frozen study files still have their recorded
SHA-256, that the shipped temperature heads hold exactly the confirmed temperatures, and that the
engine's temperature output on the confirm and dev tiers equals the frozen scorer's `ttype()`
applied to the raw dumps (largest difference over all items: 2.2e-16). The exploratory and
confirmatory scripts themselves are in
[results/calibration-study](results/calibration-study/README.md), with their outputs.
