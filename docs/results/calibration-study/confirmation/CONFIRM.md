# Confirmation: per-type temperature against the shipped H2 head, on untouched data

Written and frozen 2026-09-29, before any confirmation data was built or read. The checksum of
this file and of `temperatures.json` is recorded in `CONFIRM.sha256`.

## Claim under test

On task families that nothing in judgly was fitted, tuned or chosen on, the per-question-type
temperature ("Ttype") is at least as good as the head shipped in judgly 0.1.0 ("H2"): no loss of
accuracy and no worse calibration.

## What is compared (fixed now; nothing is refitted)

- `Ttype`: the temperatures in `temperatures.json`, fitted on the 0.1.0 fit tier's train split
  (cpu-tricks/tricks.py). Probabilities: the raw readout (softmax per option order, mapped back,
  mean over orders) with each log-probability divided by the temperature of its question type.
- `H2`: the heads in src/judgly/packs/*/heads at judgly commit 034d5e6 (the 0.1.0 release).
- `raw`: the raw readout, for reference.
- Engine settings as released: up to 4 option orders, no content-free pass.

## Confirmation data (untouched)

- Stance: ClimateCheck (rabuahmad/climatecheck @ 93d0dc5, MIT), the reserved tier. Items that
  share a claim, a sentence or a passage with any fit, dev, final, final-flagged, final-seen or
  bench item are removed before extraction (the registry already names the Climate-FEVER overlap).
- General: a new tier of task families never used in this project (not in data/registry.yaml
  and not on the list of datasets used in earlier experiments), with licences that allow
  evaluation, chosen and built before either model reads any of it. Target: 4 to 6 families,
  about 500 items each, a mix of choice, yes/no and score questions. Same leakage checks.
- The tier files and their SHA-256 are recorded before extraction. Each is read once, by both
  packs; the results are reported whatever they are.

## Endpoints, per pack (Gemma 4 12B, Qwen3-4B) and format (general, stance)

Paired differences Ttype minus H2 on the same items, 95% bootstrap intervals (1,000 resamples of
groups of related items where the tier records groups, otherwise items):

1. Accuracy: confirmed if the lower bound of the interval is above -0.01.
2. Brier score: confirmed if the upper bound of the interval is below +0.01.
3. ECE: confirmed if the upper bound of the interval is below +0.02.

The claim is confirmed for a pack and format when all three hold. It is reported per pack and
format, and overall as the number of the four cases that are confirmed. Log loss, the raw
numbers and per-family results are reported as well, without a criterion.

## What would count against it

Any of the three bounds missed. If the claim fails in some cases, judgly 0.2 does not switch
those cases to the temperature on the strength of this work.

## Amendment 1 (2026-09-29, after the tiers were built and reviewed, before any model read them)

Data as built (judgly branch confirm-temperature, commit dbf1a9f; data/tiers-confirm.sha256):
general 2,500 items in five families (kinship/CLUTRR choice, code_outcome/CodeMMLU execution
prediction choice, spatial/SpartQA yes/no, argument_quality/IBM-ArgQ score, humour/Humicroedit
score), 500 each; stance 1,780 ClimateCheck pairs.

1. Stance resampling units: the linked groups (claims joined by a shared abstract; 70 groups,
   the largest with 837 items), because items sharing an abstract are not independent. The
   claim-only grouping (175 groups) is reported as a secondary analysis, without a criterion.
2. General: the endpoints are judged on all five families, as frozen. Because the review flagged
   argument_quality and humour as possible relatives of used rating families, and code_outcome
   as having a language shortcut, the result is also reported with each of those three families
   left out in turn, without a criterion.
3. Nothing else changes: same temperatures, same heads, same criteria.
