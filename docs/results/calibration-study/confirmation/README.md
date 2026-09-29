# Confirmation: the per-type temperature against H2 on untouched data

**Kind.** Confirmatory, pre-registered. `CONFIRM.md` is the protocol, with Amendment 1;
`temperatures.json` holds the temperatures under test; `score_confirm.py` is the scorer;
`CONFIRM.sha256` records the SHA-256 of these three files and when the amended protocol and the
scorer were frozen (it records no time for `temperatures.json`);
`result-confirm.txt` (printed) and `result-confirm.json` are the scorer's output of its confirmatory run
on the confirm tier.

**Order of events** (2026-09-29, UTC+8; commit times from git, extraction times from
`results-confirm/run.log`, other times are file modification times):

| time | event |
|---|---|
| 09:30 | `temperatures.json` written: exploratory analysis 2's Ttype fit, rounded to three decimals |
| after 09:30, before the tier was built | `CONFIRM.md` written and frozen, before any confirmation data existed (the file says so; no separate timestamp was kept). The SHA-256 of this first version was not kept: `CONFIRM.sha256` holds only the amended file's, so the text of Amendment 1 is the record of what changed |
| 10:04 | confirm tier built (judgly commit `ad8c91a`) |
| 10:37 | review fixes to the tier (commit `dbf1a9f`); tier files frozen in `data/tiers-confirm.sha256` |
| 10:38 | Amendment 1 added; `CONFIRM.md` re-frozen (SHA-256 in `CONFIRM.sha256`) |
| 10:38 to 13:12 | each model read the confirm tier once (`s1-features`, release engine settings: up to four option orders, no content-free pass) into `results-confirm/<pack>/<format>/confirm.feat`, in the order Gemma 4 12B general (10:38 to 11:31), Gemma 4 12B stance, Qwen3-4B general, Qwen3-4B stance (`results-confirm/run.log`); `s1-eval` wrote the raw and H2 per-item dumps. Extraction began after Amendment 1 was re-frozen and about two minutes before the scorer was frozen; no per-item output existed before 11:31 |
| 10:40 | `score_confirm.py` frozen, after a smoke test on the 0.1.0 dev tier only |
| 13:13 | `score_confirm.py` run once on the confirm tier; `result-confirm.txt`, `result-confirm.json` |
| later that day | the same cached readouts scored again, deterministically and after the result, by `make calibrate` (the calibration records), `docs/tools/confirmation_check.py` and `rerun.py confirm` (which re-ran the unchanged scorer and reproduced its output byte for byte) |

**Claim**, in the protocol's words: "On task families that nothing in judgly was fitted, tuned
or chosen on, the per-question-type temperature ("Ttype") is at least as good as the head shipped
in judgly 0.1.0 ("H2"): no loss of accuracy and no worse calibration." Nothing is refitted. The
criteria test this as non-inferiority within pre-set margins: a confirmed case is one where the
temperature was not worse than H2 by more than 0.01 in accuracy, 0.01 in Brier score or 0.02 in
ECE.

**Data: the confirm tier.** Built after every other tier by `scripts/prep_tiers.py`, from sources
not used before in this project (the registry, `data/registry.yaml`, and the list of earlier
datasets). Every item is dropped that matches any other tier (fit, dev, final, final-flagged,
final-seen, bench) as an exact text, a shared sentence, a shared passage (8-gram shingles,
boilerplate included) or a near duplicate; for stance also any text of the evaluation sources as
a whole (36,951 texts). `scripts/check_contamination.py` checks the same at the start of every
pipeline run.

| format | family | source | type | items | resampling groups |
|---|---|---|---|---|---|
| general | kinship | CLUTRR | choice (17 relations) | 500 | 500 |
| general | code_outcome | CodeMMLU execution prediction (Project CodeNet programs) | choice (4 verdicts) | 500 | 500 |
| general | spatial | SpartQA-YN | yes/no | 500 | 500 |
| general | argument_quality | IBM Argument Quality 30k | score (3 levels) | 500 | 15 (topics) |
| general | humour | Humicroedit (SemEval-2020 Task 7) | score (3 levels) | 500 | 417 |
| stance | confirm_climate | ClimateCheck test split (reserved in 0.1.0) | choice (3 labels) | 1,780 | 70 |

Stance groups are connected components of claims that share an abstract (the largest holds 837
items); the 175 claim groups are kept as `claim_group`. The first build had 1,785 stance items;
the review dropped five that quote IPCC text also held by Climate-FEVER (a dev source), before any
model read the tier. Sizes, labels and hashes: `data/tiers-confirm.sizes`,
`data/tiers-confirm.sha256`; licences: `docs/licences.md`.

**Amendment 1 and why.** After the tier was built and reviewed, and before any model read it:
(1) stance intervals resample the 70 linked groups, because items that share an abstract are not
independent; the 175 claim groups are a secondary analysis; (2) the review had flagged
argument_quality and humour as possible relatives of used rating families, and code_outcome as
having a language shortcut (the verdict goes with the programming language), so the general
result is also reported with each of those three left out in turn, without a criterion;
(3) nothing else changed.

**Criteria** (per pack and format; paired difference temperature minus H2 on the same items; 95%
percentile interval from 1,000 bootstrap resamples of the groups, seed 20260929): accuracy lower
bound above -0.01; Brier upper bound below +0.01; ECE upper bound below +0.02. A case is confirmed
when all three hold, and only confirmed cases switch.

**Result: 3 of 4 cases confirmed** (`result-confirm.txt`).

| pack | format | temperature - H2: accuracy | ECE | Brier | log loss (no criterion) | verdict |
|---|---|---|---|---|---|---|
| gemma4-12b-q8 | general | +0.000 [-0.014, +0.015] | +0.037 [+0.015, +0.048] | +0.017 [+0.010, +0.024] | +0.026 [+0.017, +0.037] | not confirmed (all three missed) |
| gemma4-12b-q8 | stance | +0.026 [+0.011, +0.049] | -0.026 [-0.045, +0.007] | -0.014 [-0.044, -0.001] | +0.012 [-0.036, +0.033] | confirmed |
| qwen3-4b-q8 | general | +0.022 [+0.008, +0.036] | -0.029 [-0.049, -0.014] | -0.027 [-0.035, -0.020] | -0.032 [-0.043, -0.023] | confirmed |
| qwen3-4b-q8 | stance | +0.042 [+0.023, +0.090] | -0.078 [-0.097, -0.054] | -0.031 [-0.065, -0.018] | -0.023 [-0.070, -0.001] | confirmed |

**Secondary analyses** (no criterion): with argument_quality, humour or code_outcome left out in
turn, all three criteria held for Qwen3-4B general each time and failed each time for Gemma 4 12B
general. Resampling the 175 claim groups instead of the 70 linked groups, all three held for both
stance cases. Per family (general; paired intervals in `result-confirm.json`), the largest
difference was on spatial: for Qwen3-4B the temperature was more accurate than H2 (0.612 against
0.488), for Gemma 4 12B less accurate (0.536 against 0.580). The families have 500 items each, so
Qwen3-4B's overall accuracy difference (+0.022) is the mean of the five; the mean of the four
other than spatial is -0.004 (post hoc arithmetic, no interval). spatial was not among the
families the protocol leaves out in turn.

**Rounding.** The scorer rounds to four decimals and prints three, so a value such as Gemma 4 12B
general H2 ECE 0.05048 is printed as 0.051 here and as 0.050 in the calibration records. Every
point value in `result-confirm.json` equals the corresponding record value rounded to four
decimals (checked for raw, temperature and H2, overall and per family).

**Checks and rerun from the repository.**

```bash
shasum -a 256 docs/results/calibration-study/confirmation/{CONFIRM.md,temperatures.json,score_confirm.py}
uv run --no-project --with numpy python docs/tools/confirmation_check.py
uv run --no-project --with numpy --with scipy --with scikit-learn \
    python docs/results/calibration-study/rerun.py confirm
```

The first compares with `CONFIRM.sha256`. `confirmation_check.py` also checks that the shipped
temperature files hold exactly `temperatures.json` and that the engine's temperature output on the
confirm and dev tiers equals the frozen scorer's. `rerun.py confirm` runs the unchanged
`score_confirm.py <root> <results> confirm confirm` in a scratch copy of this directory on the
committed raw and H2 dumps and the tier files (`make data`), and compares its output with the
files here; on 2026-09-29 it was byte-identical.
