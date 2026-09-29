# CPU calibration tricks on judgly 0.1.0's cached readouts: protocol

Written 2026-09-29, before any result of this experiment was computed. Exploratory: judgly 0.1.0
is not changed by it. Nothing here runs the model; everything uses the letter scores already
cached by the 0.1.0 release runs (`results/<pack>/<format>/<tier>/features.feat`, one record per
option order of each question, `z` = the letter scores of that order).

## Question

After the letter scores are read, can a better post-processing step than the per-question-type
temperature (the best simple variant so far) give better calibrated probabilities on kinds of
questions no fitting saw, without losing accuracy?

## Data and roles

- Packs: gemma4-12b-q8, qwen3-4b-q8. Formats: general, stance.
- FIT: every parameter is fitted on the fit tier's TRAIN split only.
- CHOOSE: the dev tier (task families held out from fitting) is the only data used to compare
  variants. Rule, per pack and format: the variant with the lowest dev log loss; ties within
  0.002 go to the variant with fewer parameters.
- REPORT: every variant on dev, final, final-flagged, final-seen and (general) bench, with 95%
  bootstrap intervals (groups of related items resampled together where the tier records groups)
  and differences paired against the baseline. No variant is chosen or dropped by a result on
  these tiers.
- Honesty note: the final tier has been read in earlier analyses, so its numbers here are
  supporting evidence, not a confirmation. A change to judgly needs a later confirmation on
  data nobody has read (stated separately; not part of this run).

## Variants (all keep the model's top answer, except where noted)

Baseline
- `raw`: the release's raw readout (per order: softmax of the letter scores; mapped back to the
  options; mean over orders). Reproduced from the feature files and checked against the
  release's per-item dumps.
- `Ttype`: one temperature per question type (choice, yes/no, score), applied to the log of the
  averaged probabilities.
- `H2`: the shipped head (from the release dumps), for reference.

Tricks
1. `Tdis` (order disagreement): the temperature grows with how much the option orders disagree.
   Per question, d = mean over orders of the total-variation distance between that order's
   probabilities and the averaged ones (0 when all orders agree; score questions have one order,
   so d = 0). T = exp(alpha_type + beta_type * d), fitted per question type (2 numbers per type).
   Keeps the ranking.
2. Top-label recalibration, per question type; the top answer and the ratios among the others
   are kept, only the top probability is mapped:
   - `Platt`: logit(conf') = a * logit(conf) + b (2 numbers per type);
   - `Isotonic`: a monotone step map fitted by pool-adjacent-violators;
   - `Hist`: 10 equal-width bins, each mapped to its training accuracy.
   Applied on top of `Ttype`'s probabilities (so they refine the temperature, not replace it).
3. `Ttype+Tdis` is variant 1 itself (it contains Ttype as beta = 0). No other combinations are
   tested, to keep the family of comparisons small.

Separate sub-study: calibration on a user's own data (`Tdomain`)
- Simulates a user with a few labelled examples of their own task. For each held-out family in
  the dev tier (and, as supporting evidence, the final tier), items are split in two halves by a
  hash of their group (or id). On the first half, a single temperature is fitted using the first
  n items (n = 25, 50, 100, 200, all); it is evaluated on the second half, against `Ttype` (fitted
  on the train split) and `raw` on the same second half.
- Report per n: mean over families of log loss, ECE and Brier, and how often `Tdomain` beats
  `Ttype`.

## Not done here

- Anything that needs the model again (more option orders, the content-free pass, other answer
  labels): GPU; later, if these results justify it.
- Paraphrased questions: rejected. Rewriting the question needs a language model, which adds an
  uncontrolled source of variance at the very start of the pipeline.
- Dirichlet/matrix calibration: needs a fixed number of options; choice questions have 2 to 26,
  and per-position biases already overfitted in the previous analysis (`TBtype`).

## Metrics

Accuracy, ECE (top label, 10 equal-width bins), Brier (sum over options against the one-hot
label), log loss. Same definitions as the earlier analyses (combine/temperature.py).
