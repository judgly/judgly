# External comparison: judgly and the Ollama decision models on the same test items

This directory is the complete record of one descriptive comparison: three decision models
served by Ollama (`nimble:9b`, `tev1:4b`, `tev1:0.8b`) and judgly's two packs (Gemma 4 12B and
Qwen3-4B) on the same held-out items of judgly's test tiers, scored by the same code. No
criterion decides anything; every number is reported. Hardware, versions, model digests, dates
and commands are in [ENVIRONMENT.md](ENVIRONMENT.md).

## The systems

- **Nimble** (`nimble:9b`), by Bespoke Labs: a fine-tune of Qwen3.5-9B, weights under
  Apache-2.0 ([Hugging Face](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B),
  [code and data](https://github.com/bespokelabsai/nimble)).
- **Tev1** (`tev1:4b`, and `tev1:0.8b` as a secondary system), by Together AI: fine-tunes of
  Qwen3.5-4B and Qwen3.5-0.8B; code under MIT, the weights' licence is described by Together AI
  as being finalized ([Hugging Face](https://huggingface.co/togethercomputer/Tev1-4B-experimental),
  [code and data recipe](https://github.com/togethercomputer/tev1)).
- **judgly**, Gemma 4 12B (`gemma4-12b-q8`) and Qwen3-4B (`qwen3-4b-q8`), each as raw, with the
  H2 head, with the per-type temperature and as the pack default (Gemma: H2 for general,
  temperature for stance; Qwen: temperature for both). judgly's language models are frozen and
  never trained; its calibration (the H2 head and the temperatures) was fitted on the train split
  of its fit tier only (general 6,300 items, stance 14,783 items; sources in
  [data/registry.yaml](../../../data/registry.yaml)). judgly was not run again for this
  comparison: its probabilities are the committed per-item dumps
  (`docs/results/<pack>/<format>/items-{raw,h2,temperature}-<tier>.tsv.gz`).

The external models were run as a user would run them: through Ollama 0.35.0's `/v1/systemone`
endpoint, default tags and settings, one request per item, no calibration fitted by us. The
base models are by the Qwen team (Alibaba Cloud); Ollama is by the Ollama project. We thank all
of them for releasing their work openly.

## The items

judgly's held-out test tiers, identical for every system (17,482 items): confirm (general 2,500;
stance 1,780), final (general 7,879; stance 1,343), bench (general 2,231: typed-decisions 2,000
and the JevBench public items 231) and final-flagged (general 1,000; stance 749). The fit, dev
and final-seen tiers were not used. How judgly used each tier before this comparison:

- **confirm**: built after everything else and read once by each judgly model, for the
  pre-registered confirmation of the temperature
  ([calibration-study](../calibration-study/README.md)). Each pack's default was then set from
  that read by the pre-registered rule (the temperature where it was confirmed, H2 otherwise), so
  on confirm the default rows are chosen by their own result; H2, the 0.1.0 default, was fixed
  before it.
- **final**: fresh at the release; read once in the release run and again in the exploratory
  calibration analyses.
- **bench**: external benchmarks (typed-decisions on Hugging Face, JevBench by F. Standhartinger).
- **final-flagged**: politeness and HealthFC, fresh families with known caveats (see
  `tables.md` of each pack).

According to their providers' documentation, none of the confirm, final or final-flagged items is
in either external model's fine-tuning data. Tev1 is documented as trained on the train splits of
MultiNLI, BoolQ, Banking77, AG News and SST-5 plus synthetic policy, routing and research data,
and as containing no Jev data and no JevBench items
([DATA_SOURCES.md](https://github.com/togethercomputer/tev1/blob/HEAD/DATA_SOURCES.md)). The Nimble checkpoint Ollama serves appears
to be release v3-12026, whose training components include public banking77, multinli, boolq,
ag_news, dbpedia and trec as well as local components that are not documented. Its schema
configuration lists a registered evaluation "jevbench-eval" (534 items; judgly uses the 231
public JevBench items) and marks its own "bespoke-eval" set as "reporting_only"; whether JevBench
items informed training or model selection is not stated. The frozen protocol read this as
Bespoke registering JevBench as "reporting only", and gives that as its reason for reporting bench
split into typed-decisions and JevBench (`by_source`); that reading was wrong, and the split is
kept as the protocol fixed it.

## Files

| file | what |
|---|---|
| `PROTOCOL.md` | the protocol, written before any external model answered a test item and amended once before it was frozen: the handling of items too long for Tev1's context changed from flagging them to reporting them as coverage, after a smoke test on three dev items showed that Ollama refuses such prompts instead of truncating them |
| `PROTOCOL.sha256` | SHA-256 of `PROTOCOL.md`, `run_external.py` and `score_external.py`, and the time they were frozen (2026-09-29 16:47) |
| `NOTES.md` | the one decision taken after freezing, at 16:54, while the `tev1:4b` run (started 16:47) was in progress and before any of its answers were looked at: every model is tested only through `/v1/systemone` |
| `run_external.py` | the frozen runner: one request per item, answers appended to `answers/<model>/<format>-<tier>.jsonl` |
| `score_external.py` | the frozen scorer |
| `chain.sh`, `run.log` | the script that ran the three models one after the other, and its log |
| `answers/<model>/<format>-<tier>.jsonl.gz` | the 52,446 raw responses (17,482 items for each of the three models), one JSON line per item: `id`, Ollama's `response` (or `null`), `error` (or `null`) and `seconds` (wall-clock time of the request) |
| `final/result-<model>.json`, `final/score-<model>.txt` | the scorer run with one model, on the items that model answered, with judgly on the same items |
| `final/result-all.json`, `final/score-all.txt` | the scorer run with all three models, on the items every model answered |
| `interim/result-external.json` | an interim scoring: the frozen scorer, run once at 20:56 while `nimble:9b` was still running, on Nimble 9B's answers then available (general and stance confirm, 7,424 of the 7,879 general final items); nothing was changed after it |
| `time_single.py`, `final/timing.json` | single-request timings (below) |
| `reproduce.py` | the wrapper behind `make compare-score` and `make compare-run` (below) |
| `ENVIRONMENT.md` | hardware, versions, model digests, dates and commands |

Every number in `final/` was computed by `score_external.py` from the per-item probabilities,
with the same code for every system: accuracy (top answer), ECE (top label, ten equal-width
bins), Brier score (sum over the options against the one-hot label) and log loss (probabilities
floored at 1e-12), each with a 95% bootstrap interval (1,000 resamples of the tier's groups,
seed 20260929; stance confirm by linked groups), paired differences against each judgly pack's
default, per family, and for bench per source. Items whose prompt exceeded a model's context are
counted (`items_over_context`).

## Coverage

Nimble answered all 17,482 items. Both Tev1 models answered all but 36 items, the same 36 for
both: JevBench items whose prompt Ollama counted at 2,410 to 4,042 tokens, over Tev1's limit of
2,050, which Ollama refused with HTTP 400 ("input is never truncated"). Those items are recorded
in the answers with their error; the per-model Tev1 results and the all-model results score
general bench on the other 2,195 items.

## Timing

Two kinds of timing are recorded, and neither compares like with like:

- `latency_s` in the result files: the median and 95th percentile of the `seconds` of each
  answered request in the full run (one request at a time over HTTP to Ollama). judgly has no
  such figure there; its runs were batched.
- `final/timing.json`: one question per request, one request at a time, on the same 100 items
  (the first 50 of each confirm tier ordered by the SHA-256 of the item id), after one warm-up
  request per system that is not counted (the warm-up item is the first of the 100 and is timed
  again). `p95_s` there is the 96th of the 100 sorted times, without interpolation; `latency_s`
  uses numpy's interpolated percentile. Medians: judgly Gemma 4 12B 0.960 s, judgly Qwen3-4B
  0.321 s, `nimble:9b` 0.533 s, `tev1:4b` 0.289 s, `tev1:0.8b` 0.078 s. judgly ran in-process
  through its Python API with the pack default calibration and asked each question in up to four
  option orders; the Ollama models were asked over HTTP and read each question once.

## Caveats

- **The tiers are ours.** judgly's test tiers were chosen and built by us, for judgly.
- **Calibration.** judgly's calibration was fitted by us on question types similar to those in
  the tiers; the external models were run as served, uncalibrated (plain softmax), and their
  providers do not present these probabilities as calibrated (Tev1's README: "Logprobs are model
  preferences, not calibrated confidence"; Nimble's model card: "This checkpoint has not had a
  separate temperature fit"). Fitting a temperature for them would likely lower their ECE. It
  would need no new runs (a temperature cross-fitted on these answers would do), but it was not
  in the protocol and has not been done.
- **Option orders.** judgly averages each question over up to four option orders, which is known
  to improve accuracy and calibration; the external models read each question once.
- **Model size.** judgly's Gemma 4 12B is larger than every external model; the like-for-like
  comparison by size is Qwen3-4B against Tev1 4B.
- **Tev1's prompt.** Ollama 0.35.0's `/v1/systemone` builds a `{"context", "schema"}` prompt
  (`decision/systemone.go`), while Tev1 was trained on `{"state", "question", "options"}`
  (Tev1 model card). Tev1 is therefore measured as Ollama serves it, which may understate what
  it does with its own prompt format. A run with Tev1's native format was considered and
  declined at 16:54, while the `tev1:4b` run was in progress ([NOTES.md](NOTES.md)).
- **Tev1's context.** 36 JevBench items were refused (above).
- **Training-data overlap** is as documented by the providers (above); undocumented components
  of Nimble's training data cannot be checked.
- **Timing** was measured differently in kind (in-process against HTTP, up to four option orders
  against one; above).

## Rescoring the record (CPU, seconds)

```bash
make data              # the tier files (not committed); make verify-data checks them
make compare-score
```

`make compare-score` runs `reproduce.py score` with numpy 2.5.3 on Python 3.13 (pinned: the
scorer's bootstrap uses numpy's `default_rng`, whose draws numpy does not promise to keep across
versions, so another numpy may break the byte identity): it checks the frozen scorer against
`PROTOCOL.sha256` and the tier files against `data/tiers.sha256` and
`data/tiers-confirm.sha256`, builds a scratch workspace (the tier files, judgly's committed dumps
gunzipped, and a copy of the scorer with the answers gunzipped next to it), runs the scorer once
per model and once with all three, and checks that every `result-*.json` and `score-*.txt` it
produces equals the committed one byte for byte. `tests/test_external_comparison.py` runs the
same check. The frozen files are never edited.

## Rerunning the models (Ollama, hours)

```bash
ollama pull nimble:9b && ollama pull tev1:4b && ollama pull tev1:0.8b
make compare-run                       # or COMPARE_OUT=DIR
```

`make compare-run` runs `reproduce.py run`: it needs Ollama 0.35.0 or later and the three
models pulled, compares their IDs with the recorded ones (Ollama tags can move, so a tag pulled
later may name other weights; a differing ID is reported), copies the frozen runner into
`COMPARE_OUT` (default `results-compare/`), asks every item (resumable; before each resume, answer
lines that record a failed request other than an HTTP 400 refusal are moved to
`<format>-<tier>.transient.jsonl` so that those items are asked again, since the frozen runner
skips every id it has written), scores the new answers
the same way into `COMPARE_OUT/final/` and reports whether they equal the record. It refuses an
output directory inside this one. The record's run took about five hours on the machine above.
No tolerance has been set for what counts as the same result on other hardware or Ollama
versions.
