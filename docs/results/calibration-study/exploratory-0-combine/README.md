# Exploratory analysis 0: temperature-only variants and pooling Gemma 4 12B with Qwen3-4B

**Kind.** Exploratory. No protocol was written or frozen before it ran; every variant it
computed is reported below and in `combine.json`.

**When.** 2026-09-29, after the judgly 0.1.0 release (commit `034d5e6`, 2026-09-28). File times:
`combine.py` 08:38, `combine.json` 08:40 (UTC+8).

**Question.** Does a temperature alone calibrate as well as the shipped H2 head, and does
combining the two packs' probabilities (a log-linear pool, per item) help?

**Data it read** (per-item dumps of the 0.1.0 release runs, `results/<pack>/<format>/items-{raw,h2}-<tier>.tsv`,
the same files as the gzipped ones in `docs/results/`, and the tier files for groups):

| role | tier | general items | stance items |
|---|---|---|---|
| fitting the weights | the fit tier's **test** split (in-distribution) | 1,350 | 3,145 |
| reporting | dev | 4,500 | 2,100 |
| reporting | final (fresh; resampled by group) | 7,879 | 1,343 |
| reporting | final-seen | 10,300 | 2,100 |

The weights were fitted on the fit tier's *test* split, not on its train split: after this
analysis that split is no longer an unused in-distribution check for these variants. The fresh
final tier had been read once before, by the 0.1.0 release run; this analysis read it a second
time.

**Variants** (weights w >= 0 per source, by Nelder-Mead on the mean log loss; p proportional to
the product of p_source^w): per pack the raw readout, one temperature, one temperature per
question type and the shipped H2 head; Gemma + Qwen raw readouts pooled (one weight pair, or
one per question type); both packs' H2 outputs pooled; and a plain average of the two H2
outputs. Every variant is compared with Gemma 4 12B's H2 on the same items (paired bootstrap,
1,000 resamples, seed 20260929).

**What it found** (from `combine.json`; log loss, lower is better):

- A temperature per question type had a lower log loss than H2 in 7 of the 12 pack, format and
  tier cases, for example Gemma 4 12B
  general final 0.6025 (per type) against 0.6309 (H2), Qwen3-4B general final 0.7818 against
  0.7968; on stance final (Check-COVID) H2 was better in both packs (0.5085 against 0.5355;
  0.5945 against 0.6479). Values as stored in `combine.json` (four decimals).
- No pooled variant was more than 0.0022 more accurate than Gemma 4 12B's own raw readout (general
  dev), and in the other five format-and-tier cases every pooled variant was less accurate.
  The best pooled variant's log loss differed from the best Gemma-only variant's by at most 0.030 in the six
  format-and-tier cases, lower in four and higher in two; that comparison picks the best of
  several variants after seeing the results, and pooling needs both models at run time. Pooling
  was not pursued.
- The temperature-per-type idea was taken to exploratory analysis 1 and fitted there on the
  train split.

**Rerun from the repository** (needs the tier files from `make data`; no model, no features):

```bash
uv run --no-project --with numpy --with scipy --with scikit-learn \
    python docs/results/calibration-study/rerun.py combine
```

`rerun.py` gunzips the committed dumps into a scratch root, runs the unchanged `combine.py` there
(`python combine.py <root>`, as its docstring says) and compares `combine.json` with this one.
On 2026-09-29 the rerun was byte-identical.
