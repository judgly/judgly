# The calibration study: exploratory analyses and the pre-registered confirmation

These are the files behind the second calibration option, the per-type temperature, exactly as
they were run on 2026-09-29, after the judgly 0.1.0 release: three exploratory analyses of the
cached readouts of the 0.1.0 runs and one pre-registered confirmation on untouched data. Each
directory has a README saying what the analysis is, when it ran, what data it read, whether it
was exploratory or confirmatory, what it found and how to rerun it. The write-up is in
[docs/calibration-options.md](../../calibration-options.md) and
[docs/methods.md](../../methods.md#the-calibration-comparison).

The scripts and their outputs are kept byte for byte as they were run (checked against the
originals when they were committed); they are the record, not part of the package. The frozen
files still have the SHA-256 recorded when they were frozen.

| directory | kind | what it did | fitted on | tiers it read |
|---|---|---|---|---|
| [`exploratory-0-combine`](exploratory-0-combine/README.md) | exploratory, no protocol | temperature-only variants and pooling Gemma 4 12B with Qwen3-4B; every variant reported | the fit tier's test split | test, dev, final, final-seen |
| [`exploratory-1-temperature`](exploratory-1-temperature/README.md) | exploratory, choice rule in the docstring before any result | one temperature (T1), one per question type (Ttype), per type with position biases (TBtype), against H2; rule: lowest validation log loss | the fit tier's train split | train, validation, dev, final, final-flagged, final-seen, bench |
| [`exploratory-2-post-processing`](exploratory-2-post-processing/README.md) | exploratory, protocol frozen first (`PROTOCOL.md`, `PROTOCOL.sha256`) | order-disagreement temperature, Platt, isotonic, histogram binning on top of Ttype, chosen on dev; the own-data temperature sub-study | the fit tier's train split | train, dev, final, final-flagged, final-seen, bench |
| [`confirmation`](confirmation/README.md) | confirmatory, pre-registered (`CONFIRM.md` with Amendment 1, `CONFIRM.sha256`) | Ttype with the temperatures frozen in `temperatures.json` against H2, three criteria fixed in advance | nothing refitted | confirm (read once by each model); dev for the scorer's smoke test |

Facts for the record:

- The fresh final tier and final-seen were read once by the 0.1.0 release run, as designed, and
  then again by each of the three exploratory analyses; final-flagged and bench by the release
  run and by analyses 1 and 2. For any comparison involving the temperature these tiers are not
  untouched.
- In exploratory analysis 1 the choice rule chose H2 in all four pack and format cases; in
  analysis 2 the frozen rule chose Ttype in one of four. Taking the temperature to a
  confirmation was a judgement made after seeing the held-out results, not the output of either
  rule, which is why it had to be confirmed on data nobody had read.
- In exploratory analysis 2 three bugs in the analysis script were fixed during the run; the
  frozen protocol was not changed.
- The confirm tier was built after every other tier (judgly commits `ad8c91a` and `dbf1a9f`) and
  frozen by SHA-256 (`data/tiers-confirm.sha256`). Each model read it once (`s1-features`, 10:38
  to 13:12 by `results-confirm/run.log`), starting after `CONFIRM.md` (with Amendment 1) and
  `temperatures.json` were frozen and about two minutes before `score_confirm.py` was frozen
  (10:40); no per-item output existed before 11:31. The frozen scorer produced the confirmatory
  result once (13:13). The same readouts were later scored again, deterministically, by `make
  calibrate`, `confirmation_check.py` and `rerun.py`.

## Rerunning

The scripts take the root of a judgly checkout on the command line and read uncompressed
per-item dumps under `<root>/results/`; they write their output next to themselves.
[`rerun.py`](rerun.py) runs them without editing them: it builds a scratch workspace (the
committed dumps gunzipped, links to `data/` and to the feature directories of a finished pack
run, and for analysis 1 the train and validation dumps written by `build/cli/s1-eval`), copies
the analysis directory there, runs the script with the arguments its docstring gives, and
compares every output with the committed file.

```bash
make data tools            # the tier files and the command-line tools
uv run --no-project --with numpy --with scipy --with scikit-learn \
    python docs/results/calibration-study/rerun.py all
```

`combine` and `confirm` need only the committed dumps and the tier files; `temperature` and
`tricks` also need the feature files of a finished pack run in `results/` (`--results` to point
elsewhere). Nothing runs the model. On 2026-09-29 all four reruns (Python 3.13.13, numpy 2.5.3,
scipy 1.18.1, scikit-learn 1.9.1) gave output byte-identical to the committed files.

`docs/tools/confirmation_check.py` checks that the frozen files still have their recorded
SHA-256, that the shipped temperature heads hold exactly the confirmed temperatures, and that the
engine's temperature output on the confirm and dev tiers equals the frozen scorer's.
