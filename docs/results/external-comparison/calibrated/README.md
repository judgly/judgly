# The same calibration for every system: an equal-calibration control

In the comparison one directory up, the three Ollama decision models were scored as served,
with no calibration step, while judgly was scored with the calibration step it ships, fitted
by us on judgly's training split. This control asks how much of judgly's lower calibration
error came from that step rather than from its models. Every system gets the same step:
temperature scaling (Guo et al. 2017, *On Calibration of Modern Neural Networks*, ICML), with
one temperature per question type, fitted by the same code on the same sample of judgly's
training split, and scored on the same test items as the comparison. Temperature scaling is
Guo et al.'s method, not ours; judgly's per-type temperature is the same method applied per
question type. The control is descriptive: every system and test set is reported, and no
criterion decides anything.

## What was done, in order (2026-09-30)

1. **09:24, frozen.** `PROTOCOL.md`, `sample.py`, `run_calibration.py` and `score_calibrated.py`
   were written, `sample.py` wrote `sample.json`, and the five files were frozen with their
   SHA-256 in `PROTOCOL.sha256`, before any external model answered a training item.
   `shasum -a 256 -c PROTOCOL.sha256` checks them (its last line, the freeze time, is not a hash
   line and is reported as improperly formatted). The freeze time is that last line, written by
   the author, to the minute; the run log starts in the same minute. The frozen files were first
   committed together with the answers and results (commit 9b96604), so git does not timestamp
   the freeze independently of the run.
2. **09:24 to 09:58, the runs.** `run.sh` ran `run_calibration.py` for `nimble:9b`, `tev1:4b` and
   `tev1:0.8b`, one after the other (`run.log`), with the same models (tags and digests) and the
   same request code as the comparison (`run_external.py`'s `ask`, imported from the frozen file),
   one request per item.
3. **Before the scoring, a preview.** A provisional, informal preview of Nimble 9B's calibrated
   numbers was computed with the same functions before the frozen scorer was run. It is not part
   of the record, and the frozen files still match `PROTOCOL.sha256`.
4. **Scored, after the runs finished at 09:58.** `score_calibrated.py` was run once. It wrote
   `result.json`; its printed output is `result.txt`.

The external models' answers were then gzipped for the record (`gzip -n -9`, byte-reproducible;
the frozen runner and scorer read them uncompressed), and judgly's train-split readout of the
sampled items was extracted into `judgly-train/` so that the scoring can be rerun from the
repository alone (below).

## The training sample

From judgly's fit tier, train split (`data/tiers/<format>/fitdev.jsonl`, the items judgly's
calibration was fitted on), the first 500 items of each question type ordered by the SHA-256 of
the item id: 1,500 general items (500 choice, 500 yes/no, 500 score) and 500 stance items (all
choice). By source:

- general choice: HH-RLHF 214, LEDGAR 81, GoEmotions 67, Cosmos QA 57, MMLU 19, Banking77 18,
  QASC 16, generated 14, CommonsenseQA 14;
- general yes/no: Civil Comments 485, generated 11, StrategyQA 4;
- general score: Civil Comments toxicity ratings 461, HelpSteer2 34, generated 5;
- stance: SNLI 188, VitaminC 118, MultiNLI 112, FEVER 49, WANLI 33.

Two of these sources are in the external models' documented training data (MultiNLI for both,
Banking77 for both); those models may have seen these items in training.

**Refusals.** All three external models refused the same 49 general items with HTTP 400
(`{"error":"state must not be empty"}`): choice questions that have no text to read, only the
question and its options (MMLU 19, QASC 16, CommonsenseQA 14). The external models' general
temperatures were therefore fitted on 1,451 items (451 choice), and their stance temperatures on
all 500. judgly answers questions without a text, and its refit below uses all 1,500.

## The calibration

For each system, format and question type, the temperature T minimising the mean log loss on
the sample's items of that type, found by a bounded search over log T in [-3, 4] (scipy's
`minimize_scalar`, method `bounded`), that is T between 0.050 and exp(4) = 54.598. The calibrated
probabilities are p_T proportional to exp(log p / T), with p floored at 1e-12; a type without
items in the sample keeps T = 1. A temperature never changes which answer is on top, so accuracy
is unchanged by it.

Three calibrations of judgly are reported next to the external models:

- **judgly's default**, as in the comparison (Gemma 4 12B: H2 for general, temperature for
  stance; Qwen3-4B: temperature for both);
- **judgly's shipped temperature**, fitted by judgly's own trainer on the whole train split
  (6,300 general and 14,783 stance items; T within [0.05, 100]);
- **judgly's temperature refitted on this sample** with the same code as the external models, so
  that both sides are calibrated on identical data and by identical code.

judgly's H2 head cannot be fitted for the external models: it reads the model's internal state,
which Ollama does not expose. It appears here only as part of Gemma 4 12B's general default.

**Fitted temperatures** (from `result.json`; for reference, judgly's shipped temperatures):

| system | general choice | general yes/no | general score | stance |
|---|---|---|---|---|
| Nimble 9B (fitted on 1,451 / 500) | 2.116 | 1.487 | 12.905 | 2.650 |
| Tev1 4B (1,451 / 500) | 1.884 | 1.075 | **54.598** | 1.559 |
| Tev1 0.8B (1,451 / 500) | 1.359 | **54.598** | **54.598** | 1.606 |
| judgly Gemma 4 12B, refitted (1,500 / 500) | 4.368 | 8.296 | 6.487 | 7.849 |
| judgly Qwen3-4B, refitted (1,500 / 500) | 8.535 | 10.239 | **54.598** | 13.237 |
| judgly Gemma 4 12B, shipped (whole train split) | 3.461 | 7.491 | 6.538 | 7.151 |
| judgly Qwen3-4B, shipped (whole train split) | 6.846 | 13.360 | 24.220 | 12.391 |

**At the bound.** Four fitted temperatures (bold) stopped at the search's upper bound, exp(4):
the log loss would have fallen further with a larger T. At T = 54.6 the answers of that type are
almost flat: Tev1 0.8B's yes/no answers on the final tier have a mean top probability of 0.503,
and its answers to the three-level politeness ratings 0.335 (`compare-control-by-type.json` in
`docs/assets/results`). A low ECE for such answers says that they are uninformative, not that
they are useful; their Brier score and log loss stay near those of a uniform answer. The large
score temperatures of every system but judgly Gemma 4 12B mean that, on the sample's score
questions (461 of 500 are Civil Comments toxicity ratings), the raw answers were far more
confident than they were right.

## Evaluation

The same test items as the comparison (confirm, final, bench and final-flagged), each external
model on the items it answered (as in its per-model result file; Tev1 on 2,195 of the 2,231
bench items), judgly on the same items, from the committed test answers and judgly's committed
per-item dumps. The metrics are the frozen comparison scorer's functions, imported from
`score_external.py`: accuracy, ECE (top label, ten bins), Brier (summed over the options) and
log loss, with 95% percentile intervals of 1,000 bootstrap resamples of the tier's groups (numpy
`default_rng(20260929)`, one generator for all tiers and models in order) and paired differences
against judgly's defaults on the same resamples.

## Results

ECE, as served and with the equal calibration, and judgly's defaults (95% intervals in
`result.json`, and in [docs/methods.md](../../../methods.md#the-same-calibration-for-every-system)
with Brier, log loss and every paired difference). Accuracy is the same with and without the
temperature.

| test set | Nimble 9B | Tev1 4B | Tev1 0.8B | judgly Gemma 4 12B, default | judgly Qwen3-4B, default |
|---|---|---|---|---|---|
| confirm, general | 0.229 → 0.059 | 0.142 → 0.065 | 0.154 → 0.043 | 0.051 | 0.047 |
| confirm, stance | 0.264 → 0.068 | 0.185 → 0.104 | 0.304 → 0.204 | 0.052 | 0.106 |
| final, general | 0.129 → 0.043 | 0.052 → 0.070 | 0.086 → 0.040 | 0.020 | 0.034 |
| final, stance | 0.113 → 0.061 | 0.041 → 0.048 | 0.111 → 0.027 | 0.087 | 0.093 |
| bench (typed-decisions and JevBench) | 0.057 → 0.164 | 0.042 → 0.120 | 0.178 → 0.066 | 0.029 | 0.128 |
| final-flagged, general | 0.327 → 0.042 | 0.183 → 0.082 | 0.105 → 0.004 | 0.106 | 0.037 |
| final-flagged, stance | 0.404 → 0.215 | 0.138 → 0.062 | 0.076 → 0.039 | 0.069 | 0.052 |

(On bench, judgly's values are on all 2,231 items; on Tev1's 2,195 they are 0.029 and 0.130.)

What this shows, and does not:

- **Most of judgly's calibration lead came from its calibration step.** Given the same
  temperature fitted on the same data, the external models' ECE came close to judgly's defaults:
  level with them on general confirm (every paired interval includes 0); better than both judgly
  packs on stance final; between them on stance confirm (except Tev1 0.8B, 0.204); within their
  range or below it on general final-flagged (Tev1 0.8B's 0.004 is nearly flat answers at the
  temperature bound, mean top probability 0.335); on general final above judgly Gemma 4 12B's
  default for all three and, against judgly Qwen3-4B's, clearly above for Tev1 4B, marginally for
  Nimble 9B (+0.009 [+0.001, +0.022]) and level for Tev1 0.8B (+0.006 [-0.004, +0.018]); and, for
  Nimble 9B, far above both on stance final-flagged (HealthFC, 0.215).
- **judgly on the same footing.** judgly's defaults were fitted on the whole train split, 4 to 30
  times this sample, and are H2 for Gemma 4 12B general questions. judgly's own temperature
  refitted on this sample gave ECE 0.072 (Gemma 4 12B) and 0.020 (Qwen3-4B) on general confirm,
  0.044 and 0.092 on stance confirm, 0.038 and 0.044 on general final, 0.114 and 0.111 on stance
  final, 0.048 and 0.154 on bench, 0.107 and 0.104 on general final-flagged and 0.093 and 0.065 on
  stance final-flagged (`result.txt`). On general final, judgly's shipped temperature gave 0.017
  (Gemma 4 12B; H2, its default, 0.020) and 0.034 (Qwen3-4B), so the rise to 0.038 and 0.044 comes
  with the smaller fitting sample, not with H2; the remaining general-final gap of Nimble 9B
  (0.043) and Tev1 0.8B (0.040) matches it. For Gemma 4 12B's general questions, H2 was better
  calibrated than the shipped temperature only on the confirm tier (0.051 against 0.087). On
  stance final and general final-flagged the calibrated external models had a lower ECE than
  judgly's refit.
- **The temperature does not always carry over.** It made Tev1 4B worse calibrated on general
  final (0.052 to 0.070) and slightly on stance final (0.041 to 0.048), where it was already well
  calibrated as served, and Nimble 9B and Tev1 4B clearly worse on bench (0.057 to 0.164 and 0.042
  to 0.120), mostly on its score questions: the large score temperatures, fitted mostly on
  toxicity ratings, flatten bench's score answers (mean top probability 0.286 and 0.250), of which
  these models got 65% and 49% right, so the flattened answers are underconfident (ECE on bench's
  score questions 0.046 to 0.367 and 0.051 to 0.244; `compare-control-by-type.json`). Nimble 9B's
  and Tev1 4B's choice answers on bench also became worse calibrated (0.051 to 0.118 and 0.059 to
  0.094).
- **Accuracy is unchanged; the Brier score is not.** For Nimble 9B and Tev1 4B the temperature
  closed much of the Brier gap to judgly's defaults on the confirm tiers, and for Nimble 9B on
  general final-flagged (Nimble 9B on general confirm: +0.139 as served to +0.045 against judgly
  Gemma 4 12B's default; Tev1 4B on stance confirm +0.065 to +0.019). What remains, mainly
  on the final tiers, follows accuracy: the calibrated models' Brier scores stay above judgly Gemma
  4 12B's wherever it is clearly more accurate. On stance final, where Tev1 4B and judgly Gemma 4
  12B are level in accuracy (0.826 and 0.827), Tev1 4B's calibrated Brier score (0.265) is below
  Gemma 4 12B's (0.280; paired -0.014 [-0.030, +0.002]). By the Brier score the temperature also
  made Tev1 4B slightly worse on the final tiers (0.431 to 0.442 and 0.264 to 0.265) and Nimble 9B
  and Tev1 4B worse on bench (0.402 to 0.479 and 0.470 to 0.529).
- **Flat answers.** Some of the lowest ECE values (Tev1 0.8B on politeness, 0.004) are those of
  nearly uniform answers at the temperature bound (above).

## Files

| file | what |
|---|---|
| `PROTOCOL.md`, `PROTOCOL.sha256` | the protocol and the SHA-256 of the five frozen files, with the freeze time |
| `sample.py`, `sample.json` | the sample's ids (frozen) |
| `run_calibration.py`, `run.sh`, `run.log` | the runner (frozen), the script that ran it, and its log |
| `answers/<model>/<format>-train.jsonl.gz` | the 6,000 raw answers (2,000 per model): `id`, Ollama's `response` (or `null`), `error` (or `null`), `seconds` |
| `score_calibrated.py` | the scorer (frozen) |
| `result.json`, `result.txt` | its output: temperatures, and for every test set and external model the point values, intervals and paired differences of every system; the printed summary |
| `judgly-train/<pack>-<format>-train.tsv.gz` | judgly's raw readout of the 2,000 sampled training items: the rows of `s1-eval --features results/<pack>/<format>/fitdev/features.feat --split train --rotations --dump-items` for the sampled ids, from judgly's cached feature files (2.8 GB, not committed) |

## Rescoring (CPU, about a minute)

```bash
make data             # the tier files (not committed); make verify-data checks them
make compare-score    # the comparison, then this control
```

`make compare-score` runs `../reproduce.py score` with numpy 2.5.3 and scipy 1.18.1 on Python
3.13 (the versions the byte-for-byte check was made with). For this control it checks the frozen
files against `PROTOCOL.sha256` and the fit tiers against `data/tiers.sha256`, copies the
frozen scorer, `sample.json` and the gunzipped answers into a scratch workspace next to a copy
of the comparison's scorer, puts a stand-in for judgly's `s1-eval` in the workspace that writes
the committed `judgly-train/` rows where the scorer asks for its dump, runs the scorer once and
checks that its `result.json` and printed output equal the committed `result.json` and
`result.txt` byte for byte. Where judgly's own `build/cli/s1-eval` and the cached feature files
are present, the committed `judgly-train/` rows are first checked against a fresh `s1-eval`
dump. `reproduce.py score --part control` runs the control alone;
`tests/test_external_comparison.py` runs the same check. The frozen files are never edited.

`make compare-figures` fits the temperatures again from the same committed inputs, checks every
number of `result.json`, and draws the calibrated models in `compare-tiers`.

Rerunning the models (`run.sh`) needs Ollama 0.35.0 or later with the three models pulled; the
models' tags can move (see the comparison's README).
