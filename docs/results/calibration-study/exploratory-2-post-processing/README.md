# Exploratory analysis 2: post-processing on top of the per-type temperature

**Kind.** Exploratory, with a protocol written and frozen before it ran: `PROTOCOL.md`, its
SHA-256 in `PROTOCOL.sha256` (frozen 2026-09-29 09:09, UTC+8). The protocol itself says that the
final tier had already been read and that its numbers here are supporting evidence, not a
confirmation.

**When.** 2026-09-29, after exploratory analyses 0 and 1. File times: `PROTOCOL.md` 09:09,
`tricks.py` 09:14, `tricks.txt` and `tricks.json` 09:15.

**Question** (from the protocol). Can a better post-processing step than the per-question-type
temperature give better calibrated probabilities on kinds of questions no fitting saw, without
losing accuracy?

**Variants.** Baselines `raw`, `Ttype` (one temperature per question type) and `H2` (the shipped
head). Refinements, each fitted per question type: `Tdis` (the temperature grows with how much
the option orders disagree, T = exp(alpha + beta d)); on top of Ttype's probabilities, top-label
recalibration by `Platt` scaling, `Isotonic` regression or `Hist` (10-bin histogram binning).

**Data it read.** It rebuilt the raw readout from the cached feature files of the 0.1.0 runs
(`results/<pack>/<format>/<set>/features.feat`, one record per option order) and checked it
against the release's per-item dumps (largest difference 1.8e-15), so no model was run.

| role | data | general items | stance items |
|---|---|---|---|
| fit, every variant | the fit tier's **train** split | 6,300 | 14,783 |
| choose (rule below) | **dev** (families held out from fitting) | 4,500 | 2,100 |
| report, every variant | dev, final (by group), final-flagged (by group), final-seen, bench (by group; general only) | 4,500, 7,879, 1,000, 10,300, 2,231 | 2,100, 1,343, 749, 2,100 |

This was the third reading of the final tier after the 0.1.0 release run (the second of
final-flagged and bench).

**Choice rule (frozen):** per pack and format, the lowest dev log loss; ties within 0.002 go to
the variant with fewer parameters.

**What it found.**

- The rule chose Platt (Gemma 4 12B general), Ttype (Gemma 4 12B stance), H2 (Qwen3-4B general)
  and Isotonic (Qwen3-4B stance): no single refinement.
- On the held-out tiers none was better than plain Ttype with any consistency. Over the 18 pack,
  format and tier comparisons, counting those where the 95% interval of the paired difference
  against Ttype lies wholly below (better) or above (worse) zero, in log loss: Platt better in 9,
  worse in 6; Tdis better in 2, worse in 5; Isotonic better in 2, worse in 12; Hist better in 1,
  worse in 8. In ECE: Hist better in 7, worse in 5; Isotonic better in 3, worse in 6; Platt
  better in 2, worse in 3; Tdis better in none, worse in 2.
- None of the refinements was taken further.

**Sub-study: a temperature fitted on a user's own items (`Tdomain`).** For each held-out family of
dev and final, the items were split into halves A and B by a hash of their group (or id). A
single temperature was fitted on the first n items of half A and judged on half B, against the
shipped per-type temperature and raw on the same half B. Mean log loss over families (in
brackets: in how many families Tdomain beat Ttype):

| pack | format | tier | families | n = 25 | 50 | 100 | 200 | all of A | shipped Ttype |
|---|---|---|---|---|---|---|---|---|---|
| gemma4-12b-q8 | general | dev | 6 | 0.694 (3/6) | 0.671 (5/6) | 0.670 (5/6) | 0.550 (3/4)* | 0.670 (4/6) | 0.740 |
| gemma4-12b-q8 | general | final | 8 | 0.628 (4/8) | 0.614 (3/8) | 0.592 (5/8) | 0.590 (5/8) | 0.590 (5/8) | 0.597 |
| gemma4-12b-q8 | stance | dev | 3 | 0.832 (2/3) | 0.837 (1/3) | 0.840 (1/3) | 0.844 (1/3) | 0.834 (1/3) | 0.858 |
| gemma4-12b-q8 | stance | final | 1 | 0.538 (1/1) | 0.538 (1/1) | 0.554 (1/1) | 0.536 (1/1) | 0.541 (1/1) | 0.557 |
| qwen3-4b-q8 | general | dev | 6 | 0.802 (4/6) | 0.736 (4/6) | 0.724 (4/6) | 0.586 (2/4)* | 0.722 (5/6) | 0.775 |
| qwen3-4b-q8 | general | final | 8 | 0.779 (5/8) | 0.777 (5/8) | 0.771 (6/8) | 0.767 (7/8) | 0.768 (7/8) | 0.788 |
| qwen3-4b-q8 | stance | dev | 3 | 0.967 (2/3) | 0.971 (2/3) | 0.978 (2/3) | 0.980 (2/3) | 0.970 (2/3) | 1.022 |
| qwen3-4b-q8 | stance | final | 1 | 0.647 (1/1) | 0.648 (1/1) | 0.659 (1/1) | 0.647 (1/1) | 0.648 (1/1) | 0.665 |

\* only the families whose half A has at least 200 items; their own Ttype mean is 0.649
(Gemma 4 12B) and 0.650 (Qwen3-4B). The shipped Ttype column is the mean over all families.

The mean log loss of Tdomain was below that of the shipped temperature in 6 of the 8 rows at
n = 25, 7 of 8 at n = 50 and 8 of 8 at n = 100; per family the picture is more mixed (for
example Gemma 4 12B stance dev: lower mean, but better in only one of three families). This is
exploratory and was not confirmed.

**Script bugs.** Three bugs in the analysis script (not in the protocol) were found and fixed
during this run; the frozen protocol was not changed. The committed `tricks.py` is the version
that produced the committed output; the earlier versions are not part of this record.

**Relation to the shipped numbers.** The script applies its own unrounded temperatures
(Nelder-Mead in log T), and its `.txt` rounds four-decimal values again to three. Its Ttype point
values equal the calibration records' to four decimals except in three cases, each 1e-4 apart
(Gemma 4 12B stance final-flagged and final-seen ECE, Qwen3-4B stance final log loss); raw log
loss differs in three Qwen3-4B cases because the script clips probabilities at 1e-12 and the
records do not.

**Rerun from the repository** (needs the tier files from `make data` and the feature files of a
finished pack run in `results/`; no model):

```bash
uv run --no-project --with numpy --with scipy --with scikit-learn \
    python docs/results/calibration-study/rerun.py tricks
```

`rerun.py` copies this directory (with `PROTOCOL.sha256`, which the script reads) into a scratch
workspace, links the feature directories and the gunzipped committed dumps into a scratch root,
runs the unchanged `tricks.py <root>` and compares `tricks.json` and the printed output with the
files here. On 2026-09-29 the rerun was byte-identical.
