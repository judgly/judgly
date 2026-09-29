# Exploratory analysis 1: temperature variants against the shipped H2 head

**Kind.** Exploratory. The fitting data, the choice rule and the tiers to report were written
into the script's docstring before any result was computed; the docstring was not frozen by a
checksum.

**When.** 2026-09-29, after the judgly 0.1.0 release and after exploratory analysis 0. File
times: `temperature.py` 08:42, `temperature.txt` and `temperature.json` 08:43 (UTC+8).

**Question.** Fitted on the same data as H2, does a temperature, or a temperature with a bias per
option position, calibrate as well as H2?

**Variants.** `raw` (the rotation-averaged letter probabilities); `T1` (one temperature);
`Ttype` (one temperature per question type: choice, yes/no, score; p_T proportional to
exp(log p / T)); `TBtype` (per question type a temperature and a bias per option position, 27
numbers per type, small L2 penalty on the biases); `H2` (the head shipped in 0.1.0).

**Data it read.**

| role | data | general items | stance items |
|---|---|---|---|
| fit (log loss, L-BFGS) | the fit tier's **train** split, the items H2 was fitted on | 6,300 (4,156 choice, 1,093 yes/no, 1,051 score) | 14,783 (choice) |
| choose (rule below) | the fit tier's **validation** split | 1,350 (892, 232, 226) | 3,072 |
| report, all variants | dev | 4,500 | 2,100 |
| report | final (fresh; resampled by group) | 7,879 | 1,343 |
| report | final-flagged (by group) | 1,000 | 749 |
| report | final-seen | 10,300 | 2,100 |
| report | bench (by group) | 2,231 | none |

The train and validation dumps (`s1-eval --split train|validation --rotations [--head h2.bin]
--dump-items`) are not committed; `rerun.py` writes them again. This analysis read the fresh
final tier, final-flagged, final-seen and bench a second time after the 0.1.0 release run
(exploratory analysis 0 had read final and final-seen too).

**Choice rule (in the docstring before any result):** per pack and format, the variant with the
lowest log loss on the validation split. The validation split is in-distribution (the same
families as training), and H2's penalty strength had itself been chosen on it.

**What it found.**

- The rule chose **H2 in all four pack and format cases** (validation log loss, Gemma 4 12B
  general 0.727 against Ttype 0.788; stance 0.528 against 0.693; Qwen3-4B general 0.831 against
  0.934; stance 0.585 against 0.751).
- On the held-out tiers the per-type temperature looked better than the rule's choice. Over the
  18 pack, format and tier comparisons (dev, final, final-flagged, final-seen, bench), Ttype's
  accuracy was at least H2's in 15, and its log loss, ECE and Brier score were each lower in 11.
  In 6 of the 18 the 95% interval of the paired difference showed Ttype worse than H2 on at
  least one of accuracy, ECE and Brier: Gemma 4 12B general final-seen (Brier), Gemma 4 12B
  stance final (ECE), Qwen3-4B general dev (ECE), final-seen (ECE, Brier) and bench (Brier), and
  Qwen3-4B stance final (ECE, Brier).
- `TBtype` had a lower log loss than Ttype on the general train and validation splits (stance:
  equal to three decimals) but a higher one in 13 of the 18 held-out comparisons: the position
  biases did not transfer. `T1`'s log loss lay between Ttype's and raw's in 15 of the 18.
- The Ttype temperatures, rounded to three decimals, are those later frozen for the
  confirmation (`../confirmation/temperatures.json`).

Because the temperature was taken further on the strength of held-out results, against the
choice rule, it had to be tested on data nobody had read: see `../confirmation/`.

**Relation to the shipped numbers.** This script applies its own unrounded optimum (L-BFGS); the shipped
files hold the values rounded to three decimals. Its Ttype point values therefore equal the
calibration records' to four decimals except in two cases, where they differ by 1e-4 (Gemma 4
12B stance final-seen ECE 0.0759 here, 0.07595 in the record; Qwen3-4B stance final log loss
0.6495 here, 0.64955 in the record); applying the unrounded temperature to the committed raw dump
reproduces this script's values. Its raw log loss clips probabilities at 1e-12 and the records do
not, which changes three raw log losses of Qwen3-4B (general final 3.4687 here, 3.4745 in the
record; general final-seen 3.5559 and 3.5588; stance dev 7.0548 and 7.0565). The printed
`.txt` rounds the four-decimal values again to three, so a few printed values differ by 0.001
from the record's own rounding. Intervals come from different bootstrap draws and differ.

**Rerun from the repository** (needs the tier files from `make data`, `make tools`, and the fit
tier's features from a finished pack run in `results/<pack>/<format>/fitdev/`; no model):

```bash
uv run --no-project --with numpy --with scipy --with scikit-learn \
    python docs/results/calibration-study/rerun.py temperature
```

`rerun.py` writes the train and validation dumps with `build/cli/s1-eval` and the shipped H2
heads, gunzips the committed held-out dumps into a scratch root, runs the unchanged script
(`python temperature.py <root> <fitdir>`, as its docstring says) and compares `temperature.json`
and the printed output with the files here. On 2026-09-29 the rerun was byte-identical.
