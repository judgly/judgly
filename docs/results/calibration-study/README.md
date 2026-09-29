# The calibration study: exploratory analyses and the pre-registered confirmation

These are the files behind the second calibration option, the per-type temperature, exactly as
they were run: three exploratory analyses of the cached readouts of the judgly 0.1.0 runs and
one pre-registered confirmation on untouched data. What they found, and how the numbers came to
be, is written up in [docs/calibration-options.md](../../calibration-options.md). The scripts
are kept as the record; they are not part of the package and are not maintained. Paths inside
them refer to the working directories they were run from (a checkout of judgly and a scratch
directory next to it).

| directory | kind | what it did | tiers it read |
|---|---|---|---|
| `exploratory-0-combine` | exploratory | temperature-only variants and a Gemma + Qwen combination, parameters fitted on the fit tier's test split | dev, final, final-seen |
| `exploratory-1-temperature` | exploratory | one temperature (T1), one per question type (Ttype), per type with position biases (TBtype), against the shipped H2 head; parameters fitted on the fit tier's train split; a choice rule fixed in the script's docstring before any result: lowest log loss on the validation split | train, validation, dev, final, final-flagged, final-seen, bench |
| `exploratory-2-post-processing` | exploratory, protocol frozen first (`PROTOCOL.md`, `PROTOCOL.sha256`) | order-disagreement temperature, Platt scaling, isotonic regression and histogram binning, each on top of Ttype, fitted on the train split and compared on dev; and a sub-study of a temperature fitted on a user's own items | dev, final, final-flagged, final-seen, bench |
| `confirmation` | confirmatory, pre-registered (`CONFIRM.md` with Amendment 1, `CONFIRM.sha256` with the freeze times) | Ttype with the temperatures frozen in `temperatures.json` against H2, on the confirm tier, judged by three criteria fixed in advance | confirm, once |

Each exploratory directory holds its script, its printed output (`.txt`, where there is one)
and its machine-readable output (`.json`). `combine.py` and `temperature.py` also read per-item
H2 dumps of the train and validation splits (`s1-eval --split train|validation --head h2.bin
--rotations --dump-items`), which are not committed.

Facts for the record:

- The fresh final tier (and final-flagged, final-seen and bench) was read once by the 0.1.0
  release run, as designed, and then again by each of the three exploratory analyses. For any
  comparison involving the temperature it is therefore not untouched.
- In `exploratory-1-temperature`, the choice rule chose H2 in all four pack and format cases
  (H2 has the lowest validation log loss, which is in-distribution). The temperature was taken
  further because of its results on the held-out tiers, which is why it had to be confirmed on
  data nobody had read.
- In `exploratory-2-post-processing`, three bugs in the analysis script were fixed during the run;
  the frozen protocol was not changed. The committed `tricks.py` is the version that produced the
  committed output. The temperatures in `confirmation/temperatures.json` are this script's Ttype
  fit, rounded to three decimals.
- The confirmation's data were built after every other tier (judgly commits ad8c91a and
  dbf1a9f), frozen by SHA-256 (`data/tiers-confirm.sha256`), and read once by each pack, after
  `CONFIRM.md`, `temperatures.json` and `score_confirm.py` were frozen. `result-confirm.txt` and
  `result-confirm.json` are the scorer's output of that one run.

`docs/tools/confirmation_check.py` checks that the frozen files still have their recorded
SHA-256, that the shipped temperature heads hold exactly the confirmed temperatures, and that the
engine's temperature output on the confirm and dev tiers equals the frozen scorer's.
