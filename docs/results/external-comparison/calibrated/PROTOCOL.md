# The same calibration for every system: protocol

Written 2026-09-30, before either external model answered any item of judgly's training split.
Frozen with PROTOCOL.sha256 together with sample.py, run_calibration.py and score_calibrated.py.

## Question

judgly's calibration comes from a small step fitted after the model (a per-question-type
temperature, or the H2 head). The external decision models were compared as served, without
such a step. If the same per-type temperature, fitted the same way on the same training items,
is given to them, how much of judgly's calibration lead remains?

## Training items (identical for every system)

A fixed sample of judgly's fit-tier TRAIN split (the data judgly's calibration was fitted on):
general 1,500 (the first 500 of each question type, choice, yes/no and score, ordered by the
SHA-256 of the item id) and stance 500 (the first 500 by the same order). sample.py writes the
ids; the sample is fixed before any model answers it.

## Systems and calibration

- External: nimble:9b, tev1:4b, tev1:0.8b through Ollama 0.35.0's `/v1/systemone`, the same
  models (same tags and digests) and the same request mapping as the comparison
  (run_external.py's `ask`), one request per item.
- Calibration: for each system, format and question type, one temperature T minimising log loss
  on the sample (bounded search over log T in [-3, 4]); probabilities p_T proportional to
  exp(log p / T), with p floored at 1e-12. Types with no items in the sample: T = 1.
- judgly controls (from existing cached readouts, no model run): (a) judgly's shipped
  temperatures (fitted on the whole train split); (b) judgly's per-type temperature refitted on
  this same 2,000-item sample, so that both sides are calibrated on identical data.
- The trained H2 head cannot be fitted for the external models (it needs the model's internal
  state, which Ollama does not expose); judgly's H2 is reported for reference only.

## Evaluation

On the same test items as the comparison (confirm, final, bench, final-flagged), using the
committed answers: accuracy (unchanged by a temperature), ECE, Brier, log loss, with 95%
bootstrap intervals over the tiers' groups and paired differences against judgly's defaults,
using the comparison scorer's functions. Descriptive: every system and set is reported; no
criterion decides anything.

## Expectation stated in advance

If judgly's calibration lead is due to its calibration step rather than to its models, the
externally calibrated systems should come close to judgly's calibration; if a clear gap remains,
that part is not explained by the step.
