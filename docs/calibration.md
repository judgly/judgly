# Calibration

## What calibrated means

A probability is *calibrated* if, among all the answers given with probability about p, a
fraction of about p is correct. If judgly gives 1,000 answers at 0.8, about 800 should be right.

Calibration is not accuracy. A model that is right 60% of the time can be perfectly
calibrated: it says 0.6 when it is unsure, 0.95 when it is sure, and is right that often. What
calibration buys you is knowing *which* answers to trust. Raw probabilities from chat-tuned
models are usually overconfident. On the packs' fresh final tier (task families no head saw),
the raw letter probabilities had expected calibration errors (ECE) of 0.194 [0.186, 0.204]
(Gemma 4 12B) and 0.268 [0.259, 0.279] (Qwen3-4B) on general questions (n = 7,879), and the
fitted heads brought them to 0.020 [0.015, 0.031] and 0.030 [0.025, 0.040], while lowering
accuracy a little (0.760 to 0.737 and 0.656 to 0.639). On stance (Check-COVID, n = 1,343) they
brought ECE from 0.142 to 0.048 [0.034, 0.072] and from 0.187 to 0.052 [0.034, 0.077], around the
0.05 bar. Averaged over the eight general families instead of pooled, head ECE was 0.051 and
0.077; see [methods.md](methods.md#paired-differences-and-per-family-averages)
([README results](../README.md#results), [model cards](model-cards/gemma4-12b-q8.md)).

These are the numbers of the H2 heads released in 0.1.0. The packs now also ship a second
calibration option, one temperature per question type, and use it by default where a
pre-registered comparison on untouched data confirmed it (Qwen3-4B general and stance, Gemma 4
12B stance); Gemma 4 12B general keeps H2. [calibration-options.md](calibration-options.md) has
both options on every tier and how each number came to be.

## The numbers reported

- **Accuracy**: how often the top answer is correct.
- **Log loss**: the average of -log(probability given to the correct answer). Lower is better.
  It rewards being right *and* knowing when you are not, and it is the primary outcome of the
  evaluation because it is a proper scoring rule (Gneiting and Raftery 2007): it is minimised
  only by honest probabilities.
- **ECE (expected calibration error)**: sort answers into 10 equal-width bins by their top
  probability (0.0-0.1, ..., 0.9-1.0); in each bin, take the gap between mean confidence and
  observed accuracy; average the gaps weighted by bin size. 0 is perfect. ECE depends on the
  binning and is noisy on small samples, so it is always reported with an interval.
- **Reliability table**: the per-bin confidence and accuracy behind the ECE. The most direct
  picture of calibration.
- **Selective accuracy at t**: accept only answers whose top probability is above t (strictly
  greater); report the accuracy of those and the share of items answered. For example "0.88 /
  34% at 0.8" (Qwen3-4B, general, fresh final tier) means that 34% of items had a top
  probability above 0.8, and 88% of those were right.
- **Intervals**: 95% bootstrap intervals from 1,000 resamples of the items (or of groups of
  related items, such as claims on one abstract, where a tier records them). Comparisons between
  two heads or models are paired: the same items are resampled for both, which gives much
  tighter intervals on the difference. ECE intervals lean upward when the true ECE is near 0,
  because ECE cannot be negative.

## The heads

A head turns the model's scores for the answer letters into probabilities. judgly has three
levels:

| head | what it fits | parameters | needs labels |
|---|---|---|---|
| H0 | nothing: the raw letter probabilities, averaged over option orders | 0 | no |
| H1 | a temperature, a bias per letter and, only with the content-free pass (off in the built-in packs), how much of the content-free scores to subtract, per question type | 28 per type | a few hundred |
| H2 | H1 plus a small correction to each letter's output row of the model | 28 + 26 x hidden size per type | thousands |
| temperature | one temperature per question type, applied to the probabilities after they are averaged over the option orders | 1 per type | about 50 to 100 per type helped on average in an exploratory sub-study; more to check it |

H1 is contextual calibration (Zhao et al. 2021) with the amount of correction learned rather
than fixed, plus temperature scaling (Guo et al. 2017). H2 lets the correction depend on the
model's hidden state at the answer position. The built-in packs run without the content-free
pass, so their heads use no contextual-calibration term. Both are fitted by minimising log loss with
L-BFGS; H2's corrections are penalised towards zero, with the penalty chosen on validation
data. [methods.md](methods.md#the-heads) has the equations.

The packs ship two options per format, both fitted on the same public data (general and
stance): an H2 head (for score questions it is H1, fitted with every level weighted equally,
[methods.md](methods.md#the-heads)) and the per-type temperature. `Engine.load(pack)` uses the
pack's default per format; `calibration="h2"`, `"temperature"` or `"raw"` chooses one for every
format ([calibration-options.md](calibration-options.md)). The temperature never changes which
answer is on top, so its top-answer accuracy is the raw readout's; H2 can change it, for better
or worse.
Both are fitted on some task families and evaluated on others, so the published calibration
describes tasks they have not seen. Your task is also one they have not seen: check it.

## Check calibration on your own data

This is the step that matters most. Label a few hundred real cases from your workflow, in the
JSONL form below, and run [own_calibration.py](../examples/own_calibration.py):

```json
{"id": "t-001", "state": "My parcel has not arrived.", "instructions": "Which team should handle this?", "options": {"delivery": "Deliveries", "billing": "Payments"}, "label": "delivery"}
```

```bash
uv run python examples/own_calibration.py my_cases.jsonl --out predictions.tsv
```

It prints accuracy, log loss and ECE with bootstrap intervals, the reliability table, and
optionally one line per item. How many cases you need depends on the precision you want: a bin
with 100 items and 90% observed accuracy has a 95% interval of about 83% to 94%. A few hundred
cases give a usable picture; 40 do not.

If the reliability table is close to the diagonal, use the probabilities as they are. If not,
fit a head of your own (below), or at least choose thresholds on your data
([thresholds.py](../examples/thresholds.py)).

## Choosing a threshold

A common use is to answer automatically when the model is confident and send the rest to a
person. [thresholds.py](../examples/thresholds.py) tabulates, for a labelled set, the share
answered and the accuracy at each threshold, and picks the lowest threshold whose accuracy
meets a target with the lower end of its 95% Wilson interval. Choose the threshold on one set of
cases and check it on another.

## Fitting a head on your own data

With a source checkout and the command-line tools
([installation.md](installation.md#build-from-source)), you can fit an H1 head or a per-type
temperature on your own labelled cases. H1 needs a few hundred cases: calibration curves flatten
after a few hundred labelled items. A temperature has one number per question type. In an
exploratory analysis of held-out families, a temperature fitted on 50 to 100 of a family's own
items had, averaged over families, a lower log loss than the shipped one in 7 or 8 of the 8
pack, format and tier rows; per family it helped in 22 to 25 of the 36 family cases and not in
the other 11 to 14
([calibration-options.md](calibration-options.md#what-the-results-do-and-do-not-show)). This was
not confirmed, so check it on cases you held back.

1. Write your cases as JSONL with the fields `id`, `task`, `family`, `split`, `type`, `state`,
   `instructions`, `options` and `label`. Split them yourself into `train`, `validation` and
   `test` (for example 60/20/20, grouped so that near-duplicates stay together).
   [examples/data/support_tickets.jsonl](../examples/data/support_tickets.jsonl) shows the form.
2. Extract features with the pack's model, template and engine settings (the built-in packs use
   at most four rotations; score questions are read once, in their natural order, whatever the
   settings):

   ```bash
   build/cli/s1-features --model /models/gemma-4-12B-it-Q8_0.gguf \
       --template src/judgly/packs/gemma4-12b-q8/template.tpl \
       --examples my_cases.jsonl --out my_cases.feat --rotations --max-rotations 4
   ```

3. Fit H1 and evaluate it on your test split, against the raw probabilities:

   ```bash
   build/cli/s1-train --features my_cases.feat --head h1 --out my-h1.bin --rotations --max-rotations 4
   build/cli/s1-eval --features my_cases.feat --split test --rotations
   build/cli/s1-eval --features my_cases.feat --split test --rotations --head my-h1.bin
   ```

   For a temperature instead, use `--head temperature --out my-temperature.bin` (it takes neither
   `--limit-train` nor `--probe`) and pass that file to `s1-eval --head` in the same way.

   `s1-train` takes the same engine settings as `s1-features`, checks the features against them,
   and records them in `my-h1.bin.json`; the engine refuses the head under other settings. Its
   log and the sidecar say, per question type, whether the head was fitted or fell back to the
   identity because it was input-independent or no better than raw on validation.

4. Use it for your questions, and keep the pack's heads for everything else:

   ```python
   from judgly import Choice, Engine, Pack

   heads = {**Pack.find("gemma4-12b-q8").heads(), "tickets": "my-h1.bin"}
   engine = Engine.load("gemma4-12b-q8", heads=heads)
   engine.decide(state, {"team": Choice(format="tickets", instructions=..., options=...)})
   ```

A head is only valid for the model file, template and engine settings it was fitted under.
The head file records the SHA-256 of the model and of the template, and the engine refuses a head fitted with a different one.
