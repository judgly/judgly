# External decision models on judgly's test tiers: protocol

Written 2026-09-29 before either external model answered any test question. Frozen with
PROTOCOL.sha256 together with the runner (run_external.py) and the scorer (score_external.py).
Descriptive comparison: no criterion decides anything; everything is reported.

## Systems

External (run exactly as a user would run them, through Ollama v0.35.0's `/v1/systemone`
endpoint, default tags, no settings changed, no calibration fitted by us):
- `nimble:9b` (Bespoke Labs; 9B, fine-tuned Qwen3.5-9B; q8_0; context 8,194 tokens).
- `tev1:4b` (Together AI; 4B, fine-tuned Qwen3.5-4B; context 2,050 tokens).
- `tev1:0.8b` (Together AI; 0.8B), secondary.

judgly (existing per-item results, branch calibration-options; nothing rerun): Gemma 4 12B and
Qwen3-4B, each as raw, H2 head, per-type temperature, and the pack default (Gemma: H2 for
general, temperature for stance; Qwen: temperature for both).

## Test items (judgly's held-out test tiers; identical items for every system)

In this order (so that the most important finish first): confirm (general 2,500; stance
1,780), final (general 7,879; stance 1,343), bench (general 2,231: typed-decisions 2,000 and
JevBench public 231), final-flagged (general 1,000; stance 749). Not used: fit, dev (used to
choose judgly's settings) and final-seen (read during judgly's development).

## Requests

One request per item: `state` = the item's state text; one question named `q`:
- choice: `type: choice`, `instructions` = the item's instructions, `criteria` = the item's
  options {key: description} in the tier file's order;
- yes/no: `type: noul`, `instructions` = the item's instructions, no criteria (Yes/No);
- score: `type: score`, `instructions` = the item's instructions, `criteria` = ["1", ..., "L"]
  (level numbers; the scale's meaning is in the instructions, as judgly gives it).
Answers: choice: `probabilities` by key; noul: P(true) = `noul`; score: `probabilities` by level
index 0..L-1, mapped to levels 1..L. Also recorded: `usage.input_tokens`, wall-clock latency,
errors. Requests are sent one at a time; deterministic endpoint (no sampling).

## Known handicaps (stated, not corrected)

- Tev1's context is 2,050 tokens. Ollama refuses a longer prompt with an error ("input is never
  truncated", checked before freezing). Refused items are reported as coverage (the share of a
  tier the model can answer); each model is scored against judgly on the items it answered, and
  all models together on the items every model answered (scorer run once per model and once
  with all models).
- Ollama sends Tev1 the Nimble-style prompt, not the format Tev1 was trained on.
- Neither external model is calibrated for these tasks (plain softmax, T = 1); they are compared
  as served. Their temperature is not fitted by us, since that would require running them on
  judgly's training data.
- Training-data overlap (documented by the providers): none of confirm, final or final-flagged
  is in either model's fine-tuning data; Nimble's served checkpoint has undocumented local
  components; JevBench public items are registered as "reporting only" by Bespoke (bench is
  reported split into typed-decisions and JevBench for that reason).

## Metrics (identical code for every system)

Against the tier's gold label: accuracy (top answer), ECE (top label, 10 equal-width bins),
Brier (sum over options against the one-hot label), log loss (probabilities floored at 1e-12).
Per tier and format, per family, with 95% bootstrap intervals over the tier's groups (stance
confirm: linked groups), and paired differences against judgly's defaults. Latency: median and
95th percentile per request on this Mac (Apple M3 Max, 64 GB), one request at a time; judgly's
figures come from batched runs and are not directly comparable (stated).
