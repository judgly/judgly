# Changelog

All notable changes to judgly. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (0.x: the API may still change).

## Unreleased (0.2.1)

No change to the package, the packs or any judgly number.

### Added

- An equal-calibration control for the comparison with the Ollama decision models
  (`docs/results/external-comparison/calibrated/`): the same per-type temperature (temperature
  scaling, Guo et al. 2017) fitted for Nimble 9B, Tev1 4B, Tev1 0.8B and judgly on the same
  sample of judgly's training split (1,500 general and 500 stance items), and all of them scored
  again on the comparison's test items. Protocol, sampler, runner and scorer frozen on 2026-09-30
  at 09:24, before any external model answered a training item; the 6,000 raw training answers
  (the three models refused the same 49 general items that have no text), the results, and
  judgly's train-split readout of the sampled items, so that the results can be rebuilt from the
  repository. Four fitted temperatures reached the search's upper bound (54.6).
- `make compare-score` also rebuilds the control's `result.json` and `result.txt` byte for byte
  (`reproduce.py score --part control`; scipy 1.18.1 is now pinned there), with a test.
- `compare-control-by-type.json`: the control's point values by question type.

### Changed

- `compare-tiers` shows each Ollama model both as served and with the equal calibration;
  `make compare-figures` fits the control's temperatures again and checks every number of its
  `result.json` before drawing.
- README restructured and shortened: what judgly is (Gemma 4 12B or Qwen3-4B, unchanged, run
  through llama.cpp, with a thin layer on top), a short "Results at a glance", and one comparison
  section with the models as served and with the equal calibration. The 0.1.0 H2 results
  (tables and the reliability and selective-accuracy figures) moved to `docs/calibration.md`, the
  reliability figure of the comparison to `docs/methods.md`, and the quoted figures of other
  systems to a shorter table (their calibration figures to `docs/methods.md`).
- The comparison's caveats: the statement that no temperature had been fitted for the external
  models is replaced by the control's results. With the same temperature fitted on the same data,
  the external models' ECE came close to judgly's defaults on most test sets, but stayed above
  them on general final, and the temperature made Nimble 9B and Tev1 4B worse calibrated on
  typed-decisions.

## 0.2.0 - 2026-09-30

### Added

- A second calibration option in every built-in pack, the per-type temperature: one temperature
  per question type (choice, yes/no, score), applied to the probabilities after they are
  averaged over the option orders, fitted on the same train items as H2
  (`heads/temperature.bin`, `heads/temperature-stance.bin`). The shipped values are the ones
  frozen before a pre-registered comparison with H2 on an untouched confirm tier, where the
  temperature met all three criteria for Gemma 4 12B stance and Qwen3-4B general and stance, and
  not for Gemma 4 12B general. `docs/calibration-options.md` describes the fit, the exploratory
  analyses that led to it (which read the final tier again), the confirmation and both options on
  every tier; `docs/results/calibration-study/` holds their scripts and outputs.
- `Engine.load(pack, calibration="default" | "h2" | "temperature" | "raw")` and
  `engine.calibration`; `Pack.heads(calibration=...)`, `Pack.defaults()`,
  `Pack.calibration_options()`.
- A temperature head type in the engine (head file type 3; `s1-train --head temperature`,
  `s1-eval --head`), with the head guardrails (model and template SHA-256, engine settings in
  the sidecar, temperature within [0.05, 100]).
- The confirm tier (five general families never used before and ClimateCheck for stance) in the
  pipeline and the calibration records; `make calibrate` fits and scores the temperature from
  cached features on the CPU; `docs/tools/confirmation_check.py`.
- The five confirm-tier sources (CLUTRR, CodeMMLU execution prediction, SpartQA-YN, IBM Argument
  Quality 30k, Humicroedit) in the data registry and `docs/licences.md`, evaluation only;
  ClimateCheck moved from the reserved tier to the stance confirm tier.
- A README per analysis in `docs/results/calibration-study/` (what it is, when it ran, what data
  it read, exploratory or confirmatory, how to rerun it) and `rerun.py`, which reruns the
  unchanged analysis scripts from a checkout and compares their output with the committed files.
- A section "The calibration comparison" in `docs/methods.md` (the exploratory analyses and the
  tiers they read, the protocol and its amendment, the confirm tier's construction, per-family
  and secondary results, and the negative results), temperature rows for every tier in the model
  cards, and a diagram of both options in `docs/calibration-options.md`.
- A comparison with three dedicated decision models served by Ollama 0.35.0 (Nimble 9B by Bespoke
  Labs, Tev1 4B and Tev1 0.8B by Together AI) on the same 17,482 items of judgly's confirm,
  final, bench and final-flagged tiers, scored by the same code; descriptive, with the protocol,
  runner and scorer frozen before any of these models answered a test item. The record
  (`docs/results/external-comparison/`: protocol, decision note, the 52,446 raw answers, results,
  single-request timings, environment and model digests), `make compare-score` (rebuilds every
  result from the committed answers, byte for byte) and `make compare-run` (asks the models
  again, never into the record); `make compare-figures` and two figures (accuracy against ECE on
  six test sets, reliability on the confirm tiers); a README section "Comparison with dedicated
  decision models" and a methods section with every number, interval and paired difference.
  Only the models' outputs are committed, not their weights.

### Changed

- The default calibration is now the temperature for Qwen3-4B (general and stance) and for Gemma
  4 12B stance; Gemma 4 12B general keeps H2. Use `calibration="h2"` for the 0.1.0 behaviour.
  The H2 head files, and every raw and H2 number and per-item dump, are unchanged. This is a
  trade-off, not a gain everywhere: with the new defaults top-answer accuracy is the raw
  readout's, but on the 0.1.0 fresh final tier stance ECE rises from 0.048 to 0.087 (Gemma 4 12B)
  and from 0.052 to 0.093 (Qwen3-4B), and Qwen3-4B general ECE from 0.030 to 0.034 (and on
  final-seen from 0.019 to 0.037, on the bench tier from 0.114 to 0.128), while on the untouched
  confirm tier the temperature was the better calibrated in these three cases
  (`docs/calibration-options.md`).
- `pack.json` schema 2 (per format, the options and the default); schema 1 packs still load.
  Calibration records are schema 3 (the `temperature` option, the condition and the confirm tier).
- The fresh final tier is no longer untouched for comparisons involving the temperature: after
  the 0.1.0 release run it was read again by the three exploratory analyses of the calibration
  study.

## 0.1.0 - 2026-09-28

First functional release. A weekend hobby project; Apple silicon (macOS 14 or later) wheels only.

### Added

- The engine as `libjudgly`: a JSON C API over
  llama.cpp (pinned commit) that answers choice, yes/no and score questions from one read of the
  text, with isolated branches, rotation averaging (score questions keep their natural level
  order) and an optional content-free pass (off in the release packs).
- The Python package: `Engine`, the question models `Choice`, `Binary` and `Score`, typed
  answers, `adecide` for asyncio, and model packs with pinned, SHA-256-checked model files
  (`gemma4-12b-q8`, `qwen3-4b-q8`).
- The data and head pipeline: a data registry with a licence policy checked against every
  dataset card, fit, dev, final, final-flagged, final-seen, bench and reserved tiers, a
  contamination checker, a self-test gate, resumable sharded extraction, H2 fitting, evaluation with bootstrap intervals, and
  calibration records. Fit guardrails: the head temperature is held within [0.05, 100], a
  question type whose head is input-independent or no better than raw on validation falls back
  to the identity (recorded), `scripts/check_heads.py` fails a pack build otherwise, and a head
  is refused under engine settings other than those it was fitted with.
- Documentation, examples, citation metadata, licence notices and a security policy.
- Calibration heads (H2) and calibration records for both built-in packs, general and stance
  formats, fitted on licence-checked data (`src/judgly/packs/*/heads`, `*/calibration`). The
  release numbers, per-item dumps and figures are in `docs/results` and `docs/assets/results`.
- libjudgly is built for macOS 14.0 in any build; the wheel is tagged `macosx_14_0_arm64` only
  when built with `MACOSX_DEPLOYMENT_TARGET=14.0` (as the release workflow does); a plain local
  build is tagged for the host macOS.
- Fit data include FEVER (SUPPORTS and REFUTES claims with their gold evidence), SNLI, WANLI and
  SciNLI for the stance head; HelpSteer2 helpfulness scores and LEDGAR contract clauses for the general
  head.
- Evaluation tiers: a fresh final tier (general: BBQ, BLiMP, Fig-QA, Circa, ETHICS deontology and
  justice, TabFact, ESCI, CEFR-SP; stance: Check-COVID), final-flagged (Stanford Politeness,
  HealthFC; reported apart, judging no bar), final-seen (an earlier held-out tier, a secondary
  evaluation) and bench (the 2,000 typed-decisions test decisions and the 231 public JevBench
  v1.4.2 items, also scored against their own gold). Intervals on final, final-flagged and bench
  resample groups of related items.

### Known limitations (measured on the fresh final tier, 95% bootstrap intervals)

- The general heads lower accuracy: Gemma 4 12B 0.760 [0.751, 0.770] raw, 0.737 [0.726, 0.746]
  with the head; Qwen3-4B 0.656 [0.644, 0.667] and 0.639 [0.628, 0.650] (n = 7,879), while ECE
  falls from 0.194 to 0.020 and from 0.268 to 0.030.
- Stance head (Check-COVID, n = 1,343): ECE 0.048 [0.034, 0.072] for Gemma 4 12B (bar 0.05 met
  narrowly) and 0.052 [0.034, 0.077] for Qwen3-4B (missed narrowly); both miss the dev-tier bar
  (0.122 [0.106, 0.143] and 0.209 [0.189, 0.232] against 0.08). On HealthFC (final-flagged,
  n = 749) the stance heads lower accuracy from 0.750 to 0.634 and from 0.718 to 0.541. Source:
  `docs/results/*/*/tables.md`.
- On typed-decisions (2,000 decisions) the Gemma 4 12B head reaches 0.700 [0.677, 0.721]
  accuracy, below the 0.727 its dataset card reports for Jev; Qwen3-4B reaches 0.591
  [0.567, 0.614] with ECE 0.126.
- The final-seen tier's families were read during development, so its figures are reported
  separately and may be somewhat optimistic.

## 0.0.1

- Name placeholder on PyPI. No functionality.
