<p align="center"><img src="https://raw.githubusercontent.com/judgly/judgly/v0.2.0/docs/assets/banner.svg" alt="judgly: calibrated, deterministic judgments from an open language model" width="720"></p>

# judgly

judgly asks an open language model typed questions about a piece of text (pick one of these
options, yes or no, a rating from 1 to 5) and returns a probability for every allowed answer
instead of generated text, on your own Mac. A small head fitted on public data recalibrates
those probabilities, aiming for answers given with 0.8 to be right about 80% of the time. On
eight task families that no head was fitted on, the released heads (H2) brought the pooled
calibration error (ECE) from 0.194 to 0.020 (Gemma 4 12B) and from 0.268 to 0.030 (Qwen3-4B), at
a cost of about 2 accuracy points. A second, simpler option, one temperature per question type,
keeps the raw top-answer accuracy; it is the default where a pre-registered comparison on
untouched data confirmed it (Qwen3-4B, and stance questions in both packs), so the H2 numbers
above are not what those defaults give ([Two calibration options](https://github.com/judgly/judgly/blob/v0.2.0/README.md#two-calibration-options)).
Every accuracy and calibration number for judgly in this README can be recomputed from the
per-item results committed in docs/results/.

It is a weekend hobby project, built from well-known pieces (llama.cpp, an open model,
option-order averaging and a calibration head). It is not more accurate than Jev or the best
open alternatives ([How judgly compares](https://github.com/judgly/judgly/blob/v0.2.0/README.md#how-judgly-compares));
its strengths are calibration measured on task families no head was fitted on, an evaluation
that can be recomputed from committed per-item results, and licence-checked heads that run
locally. Its weaknesses are listed under
[Limitations](https://github.com/judgly/judgly/blob/v0.2.0/README.md#limitations). The idea of
typed, calibrated "System One" decisions comes from Jev, TypeSafe's commercial decision model;
judgly is an independent open take on that idea and is not affiliated with or endorsed by
TypeSafe, Google or Alibaba Cloud (the Qwen team). Full credits are at the end of this page.

**Status:** early (0.x). macOS on Apple Silicon only. The API may still change.

## How it works

<p align="center"><img src="https://raw.githubusercontent.com/judgly/judgly/v0.2.0/docs/assets/how-it-works.svg" alt="The state is read once and cached; each question branches from the cached state; the option-letter probabilities go through a small head to become calibrated probabilities." width="860"></p>

1. The model reads your text (the *state*) once, and llama.cpp caches it.
2. Every question branches from that cache on its own, so questions never see each other.
3. Each question lists its allowed answers as letters. judgly reads the model's probabilities
   for those letters at one position and generates no text. It asks each question in at most
   four cyclic orders of the options (all of them when there are four or fewer) and averages,
   which reduces the effect of option order; with more than four options some position bias
   can remain.
4. A small fitted *head* (a few hundred thousand numbers, against billions in the model) turns
   the letter scores into calibrated probabilities.

For a fixed model file, build and hardware, the same request gives the same numbers every
time. [docs/methods.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/methods.md)
describes the method and the evaluation in detail.

## Installation

Requirements:

- a Mac with Apple Silicon (M1 or later) and macOS 14 or later;
- Python 3.10 or newer;
- memory and disk for the model:

| pack | model | download (disk) | peak memory measured | Mac with |
|---|---|---|---|---|
| `gemma4-12b-q8` (default) | Google Gemma 4 12B instruction-tuned, 8-bit GGUF | 12.7 GB | about 25 GB (22.9 GiB) | 32 GB or more |
| `qwen3-4b-q8` | Qwen3-4B-Instruct-2507, 8-bit GGUF | 4.3 GB | about 9 to 10 GB | 16 GB or more |

Peak memory was measured during batch runs with the default engine settings; a single request
on a short text needs less.

With uv:

```bash
uv add judgly
```

With pip:

```bash
pip install judgly
```

The model file is downloaded from Hugging Face the first time a pack is loaded, and its
SHA-256 is checked. See [docs/installation.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/installation.md) for building from source,
using a model file you already have, and troubleshooting.

## Quickstart

The first `Engine.load("gemma4-12b-q8")` downloads 12.7 GB into the Hugging Face cache
(`~/.cache/huggingface/hub`) and hashes it once; later loads take a few seconds, and the first
call in a process also spends about 15 seconds compiling the Metal kernels. For a smaller
first try, use `"qwen3-4b-q8"` (4.3 GB). If you already have the model file, pass
`model_path="/path/to/file.gguf"` or set `JUDGLY_MODEL_DIR` to its folder.

```python
from judgly import Engine, Choice, Binary

with Engine.load("gemma4-12b-q8") as engine:
    d = engine.decide(
        "Claim: Coffee raises blood pressure.\n\nEvidence: In 40 adults, systolic pressure rose 8 mmHg after caffeine.",
        {"stance": Choice(format="stance", instructions="How does the evidence bear on the claim?",
                          options={"supports": "The evidence supports the claim",
                                   "contradicts": "The evidence contradicts the claim",
                                   "no_bearing": "The evidence has no bearing on the claim"}),
         "trial": Binary(instructions="Does the evidence describe an experiment?")})
    print(d["stance"].probs, d["trial"].p_true)
```

It prints one probability per stance option (they sum to 1) and the probability of "yes". With
Gemma 4 12B on an M3 Max, loading takes about 22 seconds with the file already downloaded.

> **About stance calibration.** The `stance` question above uses the pack's default stance
> calibration, the per-type temperature. On the fresh final tier (Check-COVID, n = 1,343) its ECE
> was 0.087 for Gemma 4 12B and 0.093 for Qwen3-4B, worse than the H2 stance heads' 0.048 and
> 0.052; on the untouched confirm tier (ClimateCheck, n = 1,780) it was 0.052 and 0.106, better
> than H2's 0.078 and 0.183 (see Results below). Stance calibration varies a lot between kinds of
> claims and evidence: treat its probabilities with more caution than the general ones.
> `heads=False` gives the raw letter probabilities, which rank the answers but are overconfident.

More examples are in [examples/](https://github.com/judgly/judgly/tree/v0.2.0/examples):
[stance_check.py](https://github.com/judgly/judgly/blob/v0.2.0/examples/stance_check.py),
[many_questions.py](https://github.com/judgly/judgly/blob/v0.2.0/examples/many_questions.py),
[async_usage.py](https://github.com/judgly/judgly/blob/v0.2.0/examples/async_usage.py),
[thresholds.py](https://github.com/judgly/judgly/blob/v0.2.0/examples/thresholds.py) (choosing a confidence threshold) and
[own_calibration.py](https://github.com/judgly/judgly/blob/v0.2.0/examples/own_calibration.py) (checking calibration on your own labelled
data). [docs/usage.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/usage.md) is the guide.

## Question types

| type | question | answer |
|---|---|---|
| choice | `Choice(instructions=..., options={key: text, ...})`, 2 to 26 options (a list of strings also works) | `probs` per key, `top` |
| yes or no | `Binary(instructions=...)` | `p_true`, `top` |
| score | `Score(instructions=..., levels=5)`, 2 to 9 levels | `probs` per level, `mean`, `top` |

A question may name a `format`, which selects a calibration head. `"stance"` (does the evidence
support the claim, contradict it, or neither?) has its own calibration. By default that is the
per-type temperature, whose ECE on the final tier (Check-COVID) was 0.087 (Gemma 4 12B) and 0.093
(Qwen3-4B), well above the 0.05 bar; the H2 stance heads were around the bar there (0.048 and
0.052) and missed the dev-tier bar (see Results below). A question with `format` unset, or with a format the pack has no head
for, uses the general head (the pack's heads entry `"*"`); "general" in this page means exactly
that. See [docs/question-formats.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/question-formats.md).

## Switching models

A *pack* ties together a model file (by Hugging Face repository, revision and SHA-256), its
prompt template and its fitted heads. A head only fits the exact model and template it was
fitted with.

```python
Engine.load("qwen3-4b-q8")          # a built-in pack
Engine.load("path/to/my-pack")      # a pack directory you built
```

[docs/model-packs.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/model-packs.md) explains how to build a pack for another model.

## Results

<!-- RESULTS:SUMMARY -->
These are the numbers of one run per pack on an Apple M3 Max, scored on the **fresh final
tier**: eight general task families and one stance family (Check-COVID) that no head was fitted
on, chosen and frozen before any head was scored on them, and not used to choose the model or its
settings. They come from the committed snapshot in
[docs/results/](https://github.com/judgly/judgly/tree/v0.2.0/docs/results) (calibration records,
per-item dumps and checksums); `make figures` checks every plotted number against it. Brackets are
95% percentile bootstrap intervals (1,000 resamples) of groups of related items (items that share a
claim, a table, a template or a query are resampled together); n is the number of questions. ECE
(expected calibration error) is the average gap between confidence and accuracy over ten bins; 0 is
perfect. The stance head's bar is a final-tier ECE below 0.05.

**These tables compare raw with H2**, the head released in 0.1.0 ("head" in the tables). H2 is
the default only for Gemma 4 12B general questions. Qwen3-4B (general and stance) and Gemma 4 12B
stance use the per-type temperature by default; its numbers on the same tier are under
[Two calibration options](https://github.com/judgly/judgly/blob/v0.2.0/README.md#two-calibration-options) below.

| pack | format (fresh final tier) | n | condition | accuracy | log loss | ECE |
|---|---|---|---|---|---|---|
| gemma4-12b-q8 | general (8 families) | 7,879 (6,803 groups) | raw | 0.760 [0.751, 0.770] | 1.553 [1.478, 1.636] | 0.194 [0.186, 0.204] |
| | | | head | 0.737 [0.726, 0.746] | 0.631 [0.613, 0.651] | **0.020** [0.015, 0.031] |
| gemma4-12b-q8 | stance (Check-COVID) | 1,343 (315 groups) | raw | 0.827 [0.806, 0.848] | 1.603 [1.370, 1.855] | 0.142 [0.123, 0.162] |
| | | | head | 0.812 [0.790, 0.836] | 0.509 [0.472, 0.550] | 0.048 [0.034, 0.072] (bar 0.05 met, narrowly) |
| qwen3-4b-q8 | general (8 families) | 7,879 (6,803 groups) | raw | 0.656 [0.644, 0.667] | 3.474 [3.330, 3.639] | 0.268 [0.259, 0.279] |
| | | | head | 0.639 [0.628, 0.650] | 0.797 [0.779, 0.815] | **0.030** [0.025, 0.040] |
| qwen3-4b-q8 | stance (Check-COVID) | 1,343 (315 groups) | raw | 0.778 [0.755, 0.801] | 3.185 [2.812, 3.589] | 0.187 [0.167, 0.212] |
| | | | head | 0.773 [0.749, 0.795] | 0.594 [0.562, 0.628] | 0.052 [0.034, 0.077] (bar 0.05 missed, narrowly) |

"raw" is the letter probabilities averaged over option orders (score questions are read once,
levels in their natural order), with no head; "head" is H2, the fitted head released in 0.1.0 (the packs now also ship the
per-type temperature, below). The
general families are social bias (BBQ), grammar (BLiMP), figurative language (Fig-QA), indirect
answers (Circa), ethics (ETHICS deontology and justice), tables (TabFact), search relevance (ESCI)
and sentence difficulty (CEFR-SP).

In short: the H2 heads cut the pooled ECE about ninefold on general questions (0.194 to 0.020 and
0.268 to 0.030) and about threefold on stance (0.142 to 0.048 and 0.187 to 0.052) and lowered log
loss, at some cost in accuracy on these fresh families. Paired over the same items, head minus raw
accuracy was -0.023 [-0.031, -0.016] (Gemma 4 12B) and -0.017 [-0.024, -0.011] (Qwen3-4B) on
general questions, and -0.015 [-0.030, -0.001] (Gemma 4 12B) and -0.005 [-0.024, 0.013]
(Qwen3-4B, within noise) on stance (`docs/tools/final_tier_stats.py`; items resampled within families, not by group). The
pooled ECE also hides differences between families: averaged over the eight general families,
head ECE was 0.051 [0.049, 0.063] (Gemma 4 12B) and 0.077 [0.071, 0.088] (Qwen3-4B), with single
families up to 0.114 [0.089, 0.142] (ethics, Qwen3-4B). ECE is never negative, so its percentile
intervals lean upward when the true value is near 0.

**Families seen during development (secondary).** An earlier held-out tier is scored as
*final-seen*. Its families were read while judgly was developed, so these numbers are
not a held-out result and are shown only for comparison; items are resampled as if independent.

| pack | format (final-seen tier) | n | condition | accuracy | ECE |
|---|---|---|---|---|---|
| gemma4-12b-q8 | general (15 tasks, 9 families) | 10,300 | raw | 0.714 [0.705, 0.723] | 0.180 [0.172, 0.189] |
| | | | head | 0.713 [0.703, 0.722] | 0.026 [0.021, 0.034] |
| gemma4-12b-q8 | stance (HealthVer) | 2,100 | raw | 0.632 [0.610, 0.652] | 0.320 [0.302, 0.342] |
| | | | head | 0.626 [0.605, 0.647] | 0.067 [0.051, 0.089] |
| qwen3-4b-q8 | general (15 tasks, 9 families) | 10,300 | raw | 0.649 [0.640, 0.658] | 0.231 [0.223, 0.240] |
| | | | head | 0.654 [0.645, 0.663] | 0.019 [0.016, 0.030] |
| qwen3-4b-q8 | stance (HealthVer) | 2,100 | raw | 0.571 [0.548, 0.592] | 0.354 [0.333, 0.377] |
| | | | head | 0.548 [0.527, 0.570] | 0.100 [0.084, 0.122] |

**External benchmarks.** Two public benchmarks for typed decisions, scored against their own
gold (evaluation only; no head saw them):

| pack | benchmark | items (cases) | condition | accuracy | Brier | ECE |
|---|---|---|---|---|---|---|
| gemma4-12b-q8 | typed-decisions | 2,000 (400) | raw | 0.702 [0.680, 0.723] | 0.359 [0.342, 0.378] | 0.251 [0.232, 0.274] |
| | | | head | 0.700 [0.677, 0.721] | 0.117 [0.109, 0.125] | 0.028 [0.019, 0.049] |
| gemma4-12b-q8 | JevBench, public items | 231 (195) | raw | 0.835 [0.784, 0.884] | 0.274 [0.193, 0.363] | 0.133 [0.093, 0.185] |
| | | | head | 0.840 [0.789, 0.890] | 0.207 [0.159, 0.256] | 0.054 [0.043, 0.106] |
| qwen3-4b-q8 | typed-decisions | 2,000 (400) | raw | 0.576 [0.549, 0.600] | 0.482 [0.457, 0.507] | 0.356 [0.334, 0.380] |
| | | | head | 0.591 [0.567, 0.614] | 0.200 [0.184, 0.217] | 0.126 [0.111, 0.147] |
| qwen3-4b-q8 | JevBench, public items | 231 (195) | raw | 0.693 [0.626, 0.751] | 0.495 [0.393, 0.607] | 0.251 [0.199, 0.314] |
| | | | head | 0.671 [0.605, 0.733] | 0.361 [0.307, 0.419] | 0.070 [0.049, 0.127] |

typed-decisions' gold is the averaged output of a teacher model, so its accuracy measures
agreement with that teacher. Brier here averages over all 2,000 decisions, score questions
included. The JevBench official scores also include sealed items, so the public-item number here
is not comparable with the JevBench leaderboard. Other systems' figures on these benchmarks are
in [How judgly compares](https://github.com/judgly/judgly/blob/v0.2.0/README.md#how-judgly-compares).

<!-- RESULTS:FIGURE-1 -->
<p align="center"><img src="https://raw.githubusercontent.com/judgly/judgly/v0.2.0/docs/assets/results/reliability.svg" alt="Reliability diagrams on the fresh final tier for both packs and both formats, raw against head" width="860"></p>

**Figure 1. Reliability on the fresh final tier.** *What it shows:* for each pack and format,
answers are grouped into ten bins by their top probability; each point is a bin's mean confidence
(x) against the share of its answers that were right (y), with 95% Wilson intervals, raw and with
the H2 head. *How to read it:* on the dotted diagonal, confidence equals accuracy; points below it
are overconfident. *What it says:* raw answers sit far below the diagonal at high confidence (ECE
0.142 to 0.268); with the heads the points lie close to it (general ECE 0.020 and 0.030, n = 7,879
each; stance 0.048 and 0.052, n = 1,343 each).

<!-- RESULTS:FIGURE-2 -->
<p align="center"><img src="https://raw.githubusercontent.com/judgly/judgly/v0.2.0/docs/assets/results/selective.svg" alt="Selective accuracy against share answered on the fresh final tier for both packs and both formats" width="860"></p>

**Figure 2. Answering only when confident.** *What it shows:* accuracy on the questions answered
(y) against the share answered (x), as the threshold on the top probability rises; markers are
the thresholds 0.5, 0.6, 0.7, 0.8, 0.9, 0.95 and 0.99; curves stop where fewer than 50
questions are left. *How to read it:* moving left trades coverage for accuracy. *What it
says:* with the Gemma 4 12B general H2 head, answering only above 0.9 kept 38% of the 7,879
questions at 0.95 accuracy, against 0.737 for all of them; on stance, above 0.9 kept 32% of 1,343
at 0.95. The Qwen3-4B general H2 head is less trustworthy at the top: above 0.95 it kept 11% at
0.90 accuracy. The raw curves cannot go far left, because without a head many answers get a top
probability above 0.999.

Per-family numbers, the flagged, final-seen, test and dev tiers, selective accuracy tables and the
determinism checks are in the model cards: [Gemma 4 12B](https://github.com/judgly/judgly/blob/v0.2.0/docs/model-cards/gemma4-12b-q8.md) and
[Qwen3-4B](https://github.com/judgly/judgly/blob/v0.2.0/docs/model-cards/qwen3-4b-q8.md). How the tiers were built and which bars were met
is in [docs/methods.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/methods.md#results); `docs/tools/final_tier_stats.py` recomputes
the per-family averages and paired differences above from the snapshot.

### Two calibration options

Each pack now ships a second calibration option next to H2: one temperature per question type,
applied to the probabilities after they are averaged over the option orders, fitted on the same
train items as H2 (6,300 general and 14,783 stance items). It never changes which answer is on top, so its top-answer accuracy is the raw accuracy. The
idea came from exploratory analyses of the 0.1.0 readouts, which read the final tier again, so
the two options were compared on a new tier that nothing had read: five general families never
used before (2,500 items) and ClimateCheck for stance (1,780 items), read once, with criteria
fixed in advance (paired temperature minus H2, 95% interval: accuracy above -0.01, Brier score
below +0.01, ECE below +0.02). The temperature met them in three of four cases:

| pack | format (confirm tier) | accuracy T / H2 | ECE T / H2 | Brier T / H2 | verdict | default |
|---|---|---|---|---|---|---|
| gemma4-12b-q8 | general (2,500) | 0.509 / 0.509 | 0.087 / 0.051 | 0.579 / 0.562 | not confirmed | H2 |
| gemma4-12b-q8 | stance (1,780) | 0.655 / 0.629 | 0.052 / 0.078 | 0.476 / 0.490 | confirmed | temperature |
| qwen3-4b-q8 | general (2,500) | 0.484 / 0.463 | 0.047 / 0.076 | 0.597 / 0.624 | confirmed | temperature |
| qwen3-4b-q8 | stance (1,780) | 0.568 / 0.526 | 0.106 / 0.183 | 0.581 / 0.612 | confirmed | temperature |

`Engine.load(pack)` uses the default shown; `calibration="h2"`, `"temperature"` or `"raw"`
chooses for every format. A confirmed case means "not worse than H2 beyond these margins on this
tier", not better everywhere: the confirm tier has a single stance source, Qwen3-4B's general
accuracy gain comes mostly from one of its five families, on the stance final tier (Check-COVID)
H2 was the better calibrated of the two in both packs by ECE and log loss (Brier favoured the
temperature for Gemma 4 12B), and for Qwen3-4B general H2 had the lower ECE on the dev, final,
final-seen and bench tiers. For Gemma 4 12B general, which keeps H2, H2 was better calibrated
than the temperature on the confirm tier.

The Results tables above are for H2. With the defaults that use the temperature, the numbers on
the fresh final tier, and on typed-decisions for Qwen3-4B, are:

| pack | format | tier | n | accuracy | log loss or Brier | ECE |
|---|---|---|---|---|---|---|
| gemma4-12b-q8 | stance | final (Check-COVID) | 1,343 (315 groups) | 0.827 [0.807, 0.848] | log loss 0.536 [0.504, 0.571] | 0.087 [0.068, 0.108] (bar 0.05 missed) |
| qwen3-4b-q8 | general | final (8 families) | 7,879 (6,803 groups) | 0.656 [0.646, 0.668] | log loss 0.783 [0.765, 0.802] | 0.034 [0.027, 0.042] |
| qwen3-4b-q8 | stance | final (Check-COVID) | 1,343 (315 groups) | 0.778 [0.754, 0.802] | log loss 0.650 [0.617, 0.682] | 0.093 [0.068, 0.116] (bar 0.05 missed) |
| qwen3-4b-q8 | general | typed-decisions (bench) | 2,000 (400 cases) | 0.576 [0.552, 0.600] | Brier 0.210 [0.194, 0.226] | 0.137 [0.123, 0.158] |

Both options on every tier, the paired intervals and how each number came to be are in
[Calibration options](https://github.com/judgly/judgly/blob/v0.2.0/docs/calibration-options.md).

The weak spots (where the heads cost accuracy, bars that were missed, and families that stay
poorly calibrated) are collected under
[Limitations](https://github.com/judgly/judgly/blob/v0.2.0/README.md#limitations), with the full
numbers in [docs/methods.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/methods.md#negative-and-null-results).

## How judgly compares

judgly is one of several open attempts at Jev-style typed decisions, and it is not the most
accurate. This section quotes other systems' own figures; three dedicated decision models that
run locally through Ollama were also run on judgly's test tiers, with the results under
[Comparison with dedicated decision models](https://github.com/judgly/judgly/blob/v0.2.0/README.md#comparison-with-dedicated-decision-models).
The only benchmark on which several other systems report is typed-decisions (2,000 decisions over
400 cases; gold is a teacher model's output, and the card gives the teacher's self-agreement,
measured on its 1,600 train and test cases, as 0.735):

| system | how it works | accuracy | who measured |
|---|---|---|---|
| meraGPT Decider 1 | hosted, proprietary | 0.768 | listed on the dataset card (measurer not stated) |
| Laya, checkpoint fine-tuned on this benchmark's train split | fine-tuned ModernBERT-large encoder | 0.766 | Laya's own model card |
| open-alternative-jev, Qwen3.6-27B (8-bit), two option orders / one | frozen LLM, letter logits, no training | 0.755 / 0.737 | the project itself |
| Jev 1.13.0 | hosted, commercial | 0.727 | the dataset card's authors, through TypeSafe's API |
| Featherless Simple Jev | hosted demo endpoint (model id `Qwen3.6-35B-A3B-classifier`) | 0.716 | listed on the dataset card (measurer not stated) |
| **judgly, Gemma 4 12B with H2** (its default for these questions) | frozen LLM, letter logits, fitted heads | **0.700** | this repository |
| ModernBERT-base specialist | frozen encoder, classifiers fitted on the train split of the same workflows | 0.646 | the dataset card's authors |
| open-alternative-jev, Qwen3.5-4B, two option orders / one | frozen LLM, letter logits, no training | 0.595 / 0.593 | the project itself |
| **judgly, Qwen3-4B with H2** | frozen LLM, letter logits, fitted heads | **0.591** | this repository |
| MiniLM-L6 specialist (22M) | frozen encoder, classifiers fitted on the train split of the same workflows | 0.587 | the dataset card's authors |
| **judgly, Qwen3-4B with its default, the temperature** | frozen LLM, letter logits, one temperature per question type | **0.576** | this repository |

Calibration is harder to compare. judgly's accuracy and ECE use the same definitions as the
third-party scorer from Luni/laya-jev-benchmark, which open-alternative-jev uses (by
open-alternative-jev's check, it reproduces Laya's published numbers to within 0.003). Applied to
judgly's per-item results, those definitions give judgly's values. By that measure open-alternative-jev's 27B reports ECE 0.020 with one option order and 0.0075
with two (judgly averages up to four orders, so the two-order figure is the closer comparison),
against 0.028 for judgly's Gemma H2; at about 4B, open-alternative-jev's Qwen3.5-4B reports ECE
0.118 with one order and 0.062 with two, against 0.126 for judgly's Qwen3-4B with H2 and 0.137
with its default, the temperature. The dataset card gives Jev ECE 0.144 but publishes
no scorer, and warns that on this benchmark ECE rewards a baseline that ignores the input (ECE
0.088), so Brier is the better guide. judgly's Brier counts every decision, as the card's rows
appear to (its Uniform row is reproduced only that way): 0.117 for the Gemma H2 against the
card's 0.148 for Jev and 0.052 for meraGPT Decider 1. That scorer leaves the 800 score questions
out of Brier; counted that way, judgly's Gemma H2 reaches 0.113, the same as
open-alternative-jev's 27B with one option order (its two-order Brier is not reported), and
judgly's Qwen3-4B H2 0.238, worse than open-alternative-jev's Qwen3.5-4B (0.164).

The closest design is **Cygnet** ([blockbrain-ai/cygnet-recipe](https://github.com/blockbrain-ai/cygnet-recipe),
MIT): the same Gemma-4-12B-it, frozen (bf16 served with vLLM there, 8-bit through llama.cpp
here), with its letter probabilities and one temperature fitted on 241 items its authors
generated. It reports 203 of the 231 public JevBench items (0.879, scored with JevBench's own
tool; one near-tie item can make it 204); judgly's Gemma H2 scores 0.840 on the same items with
judgly's scorer, which has not been checked against JevBench's tool. On the live JevBench board
(v1.5.0, as of 29 September 2026) Cygnet (73.7) and Winnow-12B Q8 (73.2) are joint leaders, in a
statistical tie, ahead of Jev 1.13.0 (72.1); the board calls 75 of its 88 neighbouring pairs
statistical ties. judgly is also not alone in measuring calibration on data its fit never saw:
Kev reports it on frozen, checksummed new sources and decider-4b on pre-registered held-out
tasks.

**judgly may suit you if** you want probabilities you can check, on your own Mac, offline, with
the same numbers for the same request; if you need to know where calibration has and has not
been measured; or if licence-checked heads and a reproducible record matter to you.

**Something else may suit you better if** you need the best accuracy per decision (meraGPT
Decider 1, fine-tuned models, larger open models, or Jev), low latency, hardware other than
Apple silicon, or reliable claim checking against scientific text. On latency: TypeSafe's
documentation says most Jev queries take about 100 ms; the typed-decisions card measured 710 ms
per five-decision case end to end, and the JevBench board 0.62 s (adjusted p50). Cygnet reports
0.05 to 0.07 s per decision (median, serial, standard tier) on 48 GB GPUs; the board measures it
at 0.23 s. judgly's release runs took 0.34 to 1.37 s per question depending on the tier (1.15 s
on the bench tier, typed-decisions and JevBench together; 0.83 s over all tiers) for Gemma 4 12B,
batched, with up to four option orders per question, on an M3 Max. That is throughput. A single
request took a median of 0.96 s (Gemma 4 12B) and 0.32 s (Qwen3-4B) on 100 confirm-tier items,
measured for the comparison below.

The figures for other systems are quoted from the cited sources (the projects' own pages, the
typed-decisions card and the JevBench board) and were not reproduced here; see
[docs/methods.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/methods.md#related-work) for
sources.

## Comparison with dedicated decision models

Ollama 0.35.0 serves three open models trained for this kind of typed decision: **Nimble 9B**
(`nimble:9b`, Bespoke Labs) and **Tev1 4B** and **Tev1 0.8B** (`tev1:4b`, `tev1:0.8b`, Together
AI). Each of them was asked every item of judgly's held-out test tiers (17,482 items) through
Ollama's `/v1/systemone` endpoint, with default tags and settings, as a user would run them, and
their probabilities were scored by the same code as judgly's. The protocol, the runner and the
scorer were frozen before any of these models answered a test item. It is a descriptive
comparison: no criterion was set and every number is reported. The record (the protocol, the
52,446 raw answers, the results, the environment and model digests) is in
[docs/results/external-comparison/](https://github.com/judgly/judgly/tree/v0.2.0/docs/results/external-comparison); `make compare-score`
rebuilds every result from the committed answers, and the 95% intervals, paired differences and
per-family numbers are in [docs/methods.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/methods.md#comparison-with-dedicated-decision-models).

judgly's rows are the pack defaults (Gemma 4 12B: H2 for general questions and the temperature
for stance; Qwen3-4B: the temperature for both), taken from the committed per-item dumps; judgly
was not run again. Accuracy is the top answer against the gold label, ECE uses the top label and
ten bins, and the Brier score is summed over the options against the one-hot label (0 is best, 2
the worst). The bench numbers are therefore scored a little differently from the benchmark
scoring in [Results](https://github.com/judgly/judgly/blob/v0.2.0/README.md#results) above: judgly Gemma 4 12B's JevBench accuracy is
0.844 here and 0.840 there, where score questions count by their rounded expected level, and
Brier there is against each benchmark's own soft gold. Each external model is scored on the
items it answered and judgly on every item. The two differ only on JevBench: Tev1 answered every
item except 36 JevBench items longer than its 2,050-token context, which Ollama refused, so the
JevBench table also gives judgly on Tev1's 195 items, and in Figure 3 the Tev1 and judgly points
in the JevBench panel are on different item sets.

On the confirm tiers, judgly's default rows are not untouched in one respect: each pack's default
was set by the pre-registered rule from judgly's one read of this same tier (the temperature
became the default where that read confirmed it). H2, the 0.1.0 default, was fixed before that
read, so its rows are given beside the defaults; for Gemma 4 12B on general questions, where the
temperature was not confirmed, the default is H2.

**confirm, general**

| system | items | accuracy | ECE | Brier |
|---|---|---|---|---|
| Nimble 9B | 2,500 | 0.481 | 0.229 | 0.702 |
| Tev1 4B | 2,500 | 0.479 | 0.142 | 0.681 |
| Tev1 0.8B | 2,500 | 0.378 | 0.154 | 0.701 |
| judgly Gemma 4 12B (default) | 2,500 | 0.509 | 0.051 | 0.562 |
| judgly Qwen3-4B (default) | 2,500 | 0.484 | 0.047 | 0.597 |
| judgly Qwen3-4B, H2 (0.1.0 default) | 2,500 | 0.463 | 0.076 | 0.624 |

**confirm, stance (ClimateCheck)**

| system | items | accuracy | ECE | Brier |
|---|---|---|---|---|
| Nimble 9B | 1,780 | 0.595 | 0.264 | 0.657 |
| Tev1 4B | 1,780 | 0.630 | 0.185 | 0.542 |
| Tev1 0.8B | 1,780 | 0.451 | 0.304 | 0.794 |
| judgly Gemma 4 12B (default) | 1,780 | 0.655 | 0.052 | 0.476 |
| judgly Qwen3-4B (default) | 1,780 | 0.568 | 0.106 | 0.581 |
| judgly Gemma 4 12B, H2 (0.1.0 default) | 1,780 | 0.629 | 0.078 | 0.490 |
| judgly Qwen3-4B, H2 (0.1.0 default) | 1,780 | 0.526 | 0.183 | 0.612 |

**final, general (8 families)**

| system | items | accuracy | ECE | Brier |
|---|---|---|---|---|
| Nimble 9B | 7,879 | 0.674 | 0.129 | 0.450 |
| Tev1 4B | 7,879 | 0.657 | 0.052 | 0.431 |
| Tev1 0.8B | 7,879 | 0.504 | 0.086 | 0.588 |
| judgly Gemma 4 12B (default) | 7,879 | 0.737 | 0.020 | 0.348 |
| judgly Qwen3-4B (default) | 7,879 | 0.656 | 0.034 | 0.440 |

**final, stance (Check-COVID)**

| system | items | accuracy | ECE | Brier |
|---|---|---|---|---|
| Nimble 9B | 1,343 | 0.817 | 0.113 | 0.312 |
| Tev1 4B | 1,343 | 0.826 | 0.041 | 0.264 |
| Tev1 0.8B | 1,343 | 0.577 | 0.111 | 0.556 |
| judgly Gemma 4 12B (default) | 1,343 | 0.827 | 0.087 | 0.280 |
| judgly Qwen3-4B (default) | 1,343 | 0.778 | 0.093 | 0.357 |

**typed-decisions**

| system | items | accuracy | ECE | Brier |
|---|---|---|---|---|
| Nimble 9B | 2,000 | 0.702 | 0.052 | 0.415 |
| Tev1 4B | 2,000 | 0.613 | 0.038 | 0.485 |
| Tev1 0.8B | 2,000 | 0.434 | 0.186 | 0.674 |
| judgly Gemma 4 12B (default) | 2,000 | 0.700 | 0.028 | 0.401 |
| judgly Qwen3-4B (default) | 2,000 | 0.576 | 0.137 | 0.578 |

**JevBench, public items**

| system | items | accuracy | ECE | Brier |
|---|---|---|---|---|
| Nimble 9B | 231 | 0.779 | 0.118 | 0.283 |
| Tev1 4B | 195 | 0.790 | 0.104 | 0.306 |
| Tev1 0.8B | 195 | 0.672 | 0.093 | 0.455 |
| judgly Gemma 4 12B (default) | 231 | 0.844 | 0.067 | 0.221 |
| judgly Qwen3-4B (default) | 231 | 0.693 | 0.084 | 0.384 |
| judgly Gemma 4 12B (default), on Tev1's items | 195 | 0.882 | 0.053 | 0.188 |
| judgly Qwen3-4B (default), on Tev1's items | 195 | 0.759 | 0.092 | 0.333 |

On the final-flagged tier (politeness and HealthFC, fresh families with known caveats), judgly's
defaults were the more accurate on general questions (0.512 and 0.492 against 0.339 to 0.421). On
stance (HealthFC) judgly Gemma 4 12B's default was the more accurate (0.750 against 0.474 to
0.689); judgly Qwen3-4B's (0.718) was ahead of Nimble 9B and Tev1 0.8B and level with Tev1 4B
(0.689; paired difference -0.029, interval -0.059 to +0.000)
([docs/methods.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/methods.md#comparison-with-dedicated-decision-models)).

<!-- RESULTS:FIGURE-3 -->
<p align="center"><img src="https://raw.githubusercontent.com/judgly/judgly/v0.2.0/docs/assets/results/compare-tiers.svg" alt="Accuracy against ECE on six test sets for Nimble 9B, Tev1 4B, Tev1 0.8B and judgly's two packs" width="860"></p>

**Figure 3. Accuracy and calibration on six test sets.** *What it shows:* each system's accuracy
(y) against its ECE (x), with 95% bootstrap intervals over groups of related items; judgly's
defaults are filled, judgly without calibration (raw) is hollow and faint. *How to read it:* up
and to the left is better; points whose intervals overlap are not clearly different (the paired
differences in docs/methods.md are the sharper test). In the JevBench panel, Tev1's points are on
the 195 items it answered and judgly's on all 231 (on Tev1's items judgly Gemma 4 12B's accuracy
is 0.882; see the JevBench table). On the confirm tiers judgly's defaults are the ones selected
from this same read (above). *What it says:* judgly Gemma 4 12B is the
most accurate or level with the most accurate on every set. With calibration, judgly's defaults
have the lowest ECE on the confirm and general final tiers; on stance final and typed-decisions
Tev1 4B is as well or better calibrated. Without its calibration judgly is among the worst
calibrated on every set (hollow points): the calibration, not the language model, is what moves
it to the left.

<!-- RESULTS:FIGURE-4 -->
<p align="center"><img src="https://raw.githubusercontent.com/judgly/judgly/v0.2.0/docs/assets/results/compare-reliability.svg" alt="Reliability diagrams on the two confirm tiers for Nimble 9B, Tev1 4B and judgly's two defaults" width="860"></p>

**Figure 4. Reliability on the confirm tiers.** *What it shows:* for ten bins of the top
probability, the mean confidence (x) against the share of correct answers (y), with 95% Wilson
intervals, on the general (2,500 items) and stance (1,780) confirm tiers. *How to read it:* on the
dotted diagonal, confidence equals accuracy; points below it are overconfident. On stance the
items come in 70 linked groups, so the Wilson intervals, which treat items as independent, are
too narrow. judgly's defaults are the ones selected from this same read (above); with H2, the
0.1.0 default, ECE was 0.076 (Qwen3-4B) on general and 0.078 (Gemma 4 12B) and 0.183 (Qwen3-4B)
on stance. *What it says:* Nimble 9B and Tev1 4B, as served, are overconfident on these tiers
(ECE 0.229 and 0.142 on general, 0.264 and 0.185 on stance). judgly's defaults lie closer to the
diagonal (0.051 and 0.047 on general, 0.052 and 0.106 on stance), but both are overconfident on
stance above 0.5 (Gemma 4 12B by 0.03 to 0.13 per bin, Qwen3-4B by 0.09 to 0.14), and Gemma 4
12B is overconfident on general questions in the 0.5 to 0.6 bin (by 0.12, 796 items).

**Single-request latency** (`final/timing.json`): 100 items (the first 50 of each confirm tier by
the SHA-256 of their id), one question per request, one request at a time, after one warm-up
request per system that was not counted (the warm-up item is the first of the 100 and was
timed again), on an Apple M3 Max (64 GB). The 95th percentile is the 96th of the 100 sorted times,
without interpolation.

| system | median (s) | 95th percentile (s) | mean (s) |
|---|---|---|---|
| Nimble 9B | 0.533 | 1.039 | 0.599 |
| Tev1 4B | 0.289 | 0.567 | 0.321 |
| Tev1 0.8B | 0.078 | 0.153 | 0.087 |
| judgly Gemma 4 12B (default) | 0.960 | 1.732 | 0.994 |
| judgly Qwen3-4B (default) | 0.321 | 0.557 | 0.333 |

These do not measure the same thing. judgly ran in-process through its Python API and asked
each question in up to four option orders; the Ollama models were asked over HTTP and read each
question once. The timings were taken after the run and were not part of the frozen protocol.

<!-- RESULTS:FIGURE-5 -->
<p align="center"><img src="https://raw.githubusercontent.com/judgly/judgly/v0.2.0/docs/assets/results/compare-timing.svg" alt="Median seconds per request with the 95th percentile, for Tev1 0.8B, Tev1 4B, judgly Qwen3-4B, Nimble 9B and judgly Gemma 4 12B" width="640"></p>

**Figure 5. Time per request.** The median (bar) and 95th percentile (whisker) from the table
above; judgly in blue, the other models in grey. The same caveats apply: judgly asks each
question in up to four option orders in-process, the others once over HTTP.

**How judgly was fitted, and the test sets.** judgly's language models are frozen and never
trained. Its calibration (the H2 heads and the per-type temperatures) was fitted on the train
split of its fit tier only: 6,300 general and 14,783 stance items from the sources listed in
[data/registry.yaml](https://github.com/judgly/judgly/blob/v0.2.0/data/registry.yaml). None of the test sets above was used for fitting.
The **confirm** tiers (five general families never used before, and ClimateCheck for stance) were
built after everything else; each judgly model read them once, for the pre-registered
confirmation of the temperature, and each pack's default was then set from that read by the
pre-registered rule (the temperature where it was confirmed, H2 otherwise); this comparison
rescored those committed answers. So nothing was fitted on the confirm tiers, but the default
rows there were chosen by their own result, and H2's rows are the comparator fixed beforehand. The
**final** tiers were fresh at the 0.1.0 release; they were read once by the release run and again
by the exploratory calibration analyses that led to the temperature option, so they are not
untouched for the temperature defaults. **typed-decisions** (2,000 decisions over 400 cases, whose
gold is a teacher model's output) and the 231 public **JevBench** items are external benchmarks;
judgly read them in the release run and in two of the exploratory analyses. The **final-flagged**
tiers were read as often as bench. The external models read every item once, in this comparison.

**What is known of the other models' training data.** As their providers document it, none of
the confirm, final or final-flagged items is in either model's fine-tuning data. Nimble is a LoRA
fine-tune of Qwen3.5-9B; the checkpoint Ollama serves appears to be release v3-12026, whose
training components include public banking77, multinli, boolq, ag_news, dbpedia and trec, as well
as local components that are not documented and cannot be checked. Nimble's schema configuration
lists a registered evaluation "jevbench-eval" (534 items; judgly uses the 231 public JevBench
items) and marks its own "bespoke-eval" set as "reporting_only"; whether JevBench items informed
training or model selection is not stated. Tev1 is a fine-tune of Qwen3.5-4B and
Qwen3.5-0.8B on the train splits of MultiNLI, BoolQ, Banking77, AG News and SST-5 plus synthetic
policy, routing and research data, and its authors state that no Jev data and no JevBench items
were used. typed-decisions is not named in either model's documented data.

**Why this comparison may favour judgly.**

- **The test sets are ours.** judgly's tiers were chosen and built by us, for judgly.
- **Calibration.** judgly's calibration was fitted by us on question types similar to those in
  the tiers; the external models were run as served, with no calibration fitted by us, and their
  providers do not present these probabilities as calibrated. Tev1's README says "Logprobs are
  model preferences, not calibrated confidence", and its model card says that calibration, among
  other things, has "not been comprehensively evaluated". Nimble's model card says "This
  checkpoint has not had a separate temperature fit" (its scorer defaults to T = 1.0; Bespoke
  Labs' hosted serving of an earlier revision divided by a fitted temperature, 2.179). Fitting a
  temperature for them would likely lower their ECE. We did not; it would need no new runs (a
  temperature cross-fitted on the committed answers would do), but it was not in the frozen
  protocol and has not been done. The comparison of judgly's calibrated defaults with the models
  as served therefore largely measures whether a calibration step was applied at all. Without
  calibration, judgly's ECE was higher than that of every external model on every set, with two
  exceptions: on stance final-flagged Nimble 9B's (0.404) was higher than both judgly models'
  (0.201 and 0.227), and on stance confirm Tev1 0.8B's (0.304) was about the same as judgly Gemma
  4 12B's (0.300).
- **Option orders.** judgly averages each question over up to four option orders, a test-time
  ensemble that is known to improve both accuracy and calibration; the external models read each
  question once, in one order.
- **Model size.** judgly's Gemma 4 12B has more parameters than any of the external models (9B,
  4B and 0.8B). The like-for-like comparison by size is judgly Qwen3-4B against Tev1 4B.
- **Tev1's prompt.** Ollama's `/v1/systemone` sends a `{"context", "schema"}` prompt, while Tev1
  was trained on `{"state", "question", "options"}`. Tev1 is measured as Ollama serves it, which
  may understate what it does with its own format. A run with its native format was considered
  and declined at 16:54, while the Tev1 4B run (started at 16:47) was in progress and before any
  of its answers were looked at.
- **Tev1's context.** 36 JevBench items were refused as too long; judgly and Nimble answered them.
- **Timing** was measured differently in kind (in-process against HTTP, up to four option orders
  against one).

**In short.** On these test sets judgly Gemma 4 12B, the largest model compared, was the most
accurate system almost everywhere: clearly so on general questions and JevBench, and level with
the best dedicated model on stance (Tev1 4B) and typed-decisions (Nimble 9B), where the paired
intervals include zero. Among the models of 4 to 9B, the dedicated models were level with judgly
Qwen3-4B's default on general questions (Nimble 9B slightly ahead on the general final tier);
against H2, the 0.1.0 default fixed before the confirm tier was read, Nimble 9B (0.481) and Tev1
4B (0.479) were ahead of Qwen3-4B (0.463) on general confirm in point value (the record has no
paired interval against H2). On the confirm and final stance tiers and on typed-decisions the
dedicated models were more accurate than judgly Qwen3-4B; on the flagged stance tier (HealthFC)
judgly Qwen3-4B was ahead of Nimble 9B and level with Tev1 4B. On JevBench Nimble 9B was ahead of
judgly Qwen3-4B and Tev1 4B level. judgly's defaults were the best calibrated on most sets, which
mostly shows that a calibration step was applied: without it, judgly was worse calibrated than
the dedicated models almost everywhere. The exceptions: on stance final Tev1 4B had the lowest
ECE (0.041 against 0.087 and 0.093); on typed-decisions Tev1 4B and Nimble 9B were better
calibrated than judgly Qwen3-4B and level with judgly Gemma 4 12B; on JevBench every interval
overlaps; on stance confirm Tev1 4B's ECE (0.185) was about the same as judgly Qwen3-4B's with
H2 (0.183); and on the flagged tiers Tev1 0.8B was level with judgly Gemma 4 12B. judgly Gemma 4
12B was the slowest per request (median 0.96 s); Tev1 0.8B was the fastest (0.08 s) but far
behind in accuracy. These are the numbers on judgly's own test sets, with its calibration fitted
by us and theirs as served, and with judgly averaging over up to four option orders; on other
data, or with a temperature fitted for the dedicated models, the picture may differ.

## Limitations

Please read these before using judgly for anything that matters.

- **Not the most accurate option.** On typed-decisions, judgly's Gemma 4 12B with H2 (0.700) is
  below Featherless Simple Jev (0.716), Jev (0.727), open-alternative-jev's 27B (0.737 to
  0.755), Laya fine-tuned on the benchmark (0.766) and meraGPT Decider 1 (0.768); at about 4B,
  judgly's Qwen3-4B (0.591 with H2, 0.576 with its default, the temperature) is at or below
  open-alternative-jev's untrained Qwen3.5-4B (0.593 to 0.595) in accuracy and worse calibrated
  (ECE 0.126 with H2 and 0.137 with the temperature, against 0.062 to 0.118)
  ([How judgly compares](https://github.com/judgly/judgly/blob/v0.2.0/README.md#how-judgly-compares)).
  On judgly's own test tiers, the dedicated decision models Nimble 9B and Tev1 4B, run as served,
  were more accurate than judgly's Qwen3-4B on the confirm and final stance tiers and on
  typed-decisions (on the flagged HealthFC stance tier judgly's Qwen3-4B was ahead of Nimble 9B
  and level with Tev1 4B)
  ([Comparison with dedicated decision models](https://github.com/judgly/judgly/blob/v0.2.0/README.md#comparison-with-dedicated-decision-models)).
- **Accuracy is modest, and calibration does not raise it.** On the fresh final tier, accuracy
  with H2 was 0.737 (Gemma 4 12B) and 0.639 (Qwen3-4B) on general questions and 0.812 and 0.773
  on stance, slightly below the raw readout (Results above); the temperature leaves the top
  answer, and so the accuracy, at the raw readout's (0.760, 0.656, 0.827, 0.778). Calibration
  tells you *when* to trust an answer; it does not make the model know more. The general H2
  heads cost 1.7 to 2.3 points there and changed accuracy by -2.2 to +1.5 points on
  typed-decisions, JevBench and the final-seen tier; the stance H2 heads cost 0.5 to 2.3 points
  on Check-COVID and HealthVer and 12 to 18 on HealthFC.
- **Calibration is uneven.** Answers the Gemma 4 12B H2 head gives at about 0.85 were right about
  85% of the time on the fresh final tier and more often on typed-decisions, but the Qwen3-4B H2
  head is overconfident on typed-decisions (answers at about 0.85 were right 61% of the time; ECE
  0.126, and 0.137 with Qwen3-4B's default, the temperature), and both stance H2 heads are
  overconfident on HealthVer and HealthFC. The pooled numbers hide weaker families (Results
  above), and some stay poorly calibrated: financial tweets with the Qwen3-4B H2 head (ECE
  0.253), WiC with either H2 head (0.325 Gemma 4 12B, 0.259 Qwen3-4B) and legal reasoning with
  the Gemma 4 12B H2 head (0.151), on the final-seen and dev tiers. The per-family tables of
  the calibration records (`tables.md` in
  [docs/results/](https://github.com/judgly/judgly/tree/v0.2.0/docs/results)) give the
  temperature's numbers for the same families.
- **Stance calibration misses its bar with the default option.** With the default stance
  calibration, the per-type temperature, ECE on Check-COVID was 0.087 (Gemma 4 12B) and 0.093
  (Qwen3-4B), well above the 0.05 bar. H2 met the bar only narrowly for Gemma 4 12B (0.048,
  interval up to 0.072) and missed it for Qwen3-4B (0.052); both H2 heads miss the dev-tier bar
  (0.122 and 0.209 against 0.08), worst on scientific abstracts; the temperature's dev-tier ECE
  was 0.069 and 0.109. On the untouched ClimateCheck tier the temperature was the better calibrated
  (0.052 and 0.106 against H2's 0.078 and 0.183). On HealthVer ECE was 0.067 and 0.100 with H2
  and 0.076 and 0.058 with the temperature; on HealthFC (reported apart because its evidence
  often states the verdict) 0.123 and 0.150 with H2 and 0.069 and 0.052 with the temperature.
- **Slow.** 0.34 to 1.37 s per question depending on the tier (1.15 s on the bench tier, 0.83 s
  over all tiers) for Gemma 4 12B in the batched release runs on an M3 Max, with up to four
  option orders per question (about 375 tokens per second read). Single requests took a median of
  0.96 s (Gemma 4 12B) and 0.32 s (Qwen3-4B) on 100 items, against 0.08 to 0.53 s for the Ollama
  decision models (judgly asks up to four option orders per question, in process; they read one,
  over HTTP)
  ([Comparison with dedicated decision models](https://github.com/judgly/judgly/blob/v0.2.0/README.md#comparison-with-dedicated-decision-models)).
- **Weak spots:** score questions such as sentence difficulty (CEFR-SP, six levels: 0.390 and
  0.253 accuracy with H2) and similarity ratings, word sense (the heads lowered WiC accuracy
  on the dev tier), legal reasoning (CaseHOLD near chance), stance on scientific abstracts and on
  health questions, and politeness ratings (a relative of a fit family; Gemma 4 12B H2 ECE 0.106)
  ([model cards](https://github.com/judgly/judgly/blob/v0.2.0/docs/model-cards/gemma4-12b-q8.md#known-weaknesses)).
- **Calibration was measured on public benchmarks.** On your task it may differ; check it on a
  few hundred labelled cases ([docs/calibration.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/calibration.md)).
- **macOS on Apple Silicon only** (Metal). Linux and CUDA builds are not set up.
- **English only.** All fitting and evaluation data is English.
- **One request at a time per engine.** The native handle is not re-entrant. `Engine`
  serialises calls with a lock, and each `Engine` holds its own copy of the model in memory.
- **Long texts are truncated** to 16,384 tokens by default; the response says so
  (`truncated`).
- **The model may have seen the benchmarks.** judgly holds whole task families out from the
  head, but it cannot hold anything out from the model's own training data
  ([docs/methods.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/methods.md#what-the-evaluation-can-and-cannot-show)).
- **Only the fresh final and final-flagged tiers are held out from development.** The
  final-seen numbers are on families read during development and may be optimistic. One smoke
  run with a throwaway head scored 173 items drawn from the fresh tiers (80 general, 93 stance);
  the tiers were not changed because of it
  ([docs/methods.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/methods.md#what-the-evaluation-can-and-cannot-show)).

## Reproducing the numbers

Everything that produced the release numbers is in this repository: the data registry
with pinned dataset revisions and licences, the tier builder, the contamination checker, the
resumable pipeline and the self-tests. [docs/reproduce.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/reproduce.md) lists the exact
commands, the hardware and where the outputs go. The outputs behind every accuracy and calibration
number above are committed in [docs/results/](https://github.com/judgly/judgly/tree/v0.2.0/docs/results), with SHA-256 checksums of the large inputs that are
not committed; the speed figures come from the run logs, which are not committed, except the
single-request timings of the comparison with dedicated decision models, which are in
[docs/results/external-comparison/](https://github.com/judgly/judgly/tree/v0.2.0/docs/results/external-comparison).

## Documentation

- [Installation](https://github.com/judgly/judgly/blob/v0.2.0/docs/installation.md)
- [Usage](https://github.com/judgly/judgly/blob/v0.2.0/docs/usage.md)
- [Question formats](https://github.com/judgly/judgly/blob/v0.2.0/docs/question-formats.md)
- [Model packs](https://github.com/judgly/judgly/blob/v0.2.0/docs/model-packs.md)
- [Calibration](https://github.com/judgly/judgly/blob/v0.2.0/docs/calibration.md): what ECE and selective accuracy mean, and how to check or
  fit calibration on your own data
- [Calibration options](https://github.com/judgly/judgly/blob/v0.2.0/docs/calibration-options.md): the H2 head and the
  per-type temperature, how they were compared, and every number of both
- [Methods](https://github.com/judgly/judgly/blob/v0.2.0/docs/methods.md): the method and the evaluation design
- [Reproduce](https://github.com/judgly/judgly/blob/v0.2.0/docs/reproduce.md)
- [FAQ](https://github.com/judgly/judgly/blob/v0.2.0/docs/faq.md)
- Model cards: [Gemma 4 12B](https://github.com/judgly/judgly/blob/v0.2.0/docs/model-cards/gemma4-12b-q8.md),
  [Qwen3-4B](https://github.com/judgly/judgly/blob/v0.2.0/docs/model-cards/qwen3-4b-q8.md)
- [Licences](https://github.com/judgly/judgly/blob/v0.2.0/docs/licences.md): the code, heads, results, models and datasets
- [Security](https://github.com/judgly/judgly/blob/v0.2.0/SECURITY.md)

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](https://github.com/judgly/judgly/blob/v0.2.0/CONTRIBUTING.md) for building,
testing and the rules for changes, [CODE_OF_CONDUCT.md](https://github.com/judgly/judgly/blob/v0.2.0/CODE_OF_CONDUCT.md),
[CHANGELOG.md](https://github.com/judgly/judgly/blob/v0.2.0/CHANGELOG.md) for what changed, and [SECURITY.md](https://github.com/judgly/judgly/blob/v0.2.0/SECURITY.md) for reporting a
vulnerability privately.

## Credits and acknowledgements

judgly is built on other people's work. Thank you to:

- **TypeSafe and Jev.** Jev, TypeSafe's commercial decision model, gave me the idea: typed
  questions answered with calibrated probabilities read at one position, with each question
  isolated and branched from one reading of the text. TypeSafe's launch post (15 September
  2026) and Archer Hume's black-box study of Jev (17 September 2026) shaped the isolation,
  option-order and irrelevant-option probes in `s1-probe` (cited by date; links to follow). judgly is not affiliated with TypeSafe and uses none of their code or weights.
- **llama.cpp and ggml** by Georgi Gerganov and the ggml authors (MIT), which run the models.
  libjudgly also compiles the llamafile sgemm kernels (Mozilla Foundation, MIT) and the YaRN
  RoPE code (Jeffrey Quesnelle and Bowen Peng, MIT) that ship inside ggml.
- **Gemma 4** by Google (Apache-2.0), the default model, and ggml-org for the GGUF conversion.
- **Qwen3** by the Qwen team at Alibaba Cloud (Apache-2.0), the model of the smaller pack, and
  Unsloth for the GGUF conversion.
- **yyjson** by YaoYuan ([ibireme/yyjson](https://github.com/ibireme/yyjson), MIT), for JSON
  in C.
- **open-alternative-jev** by ikermoel
  ([ikermoel/open-alternative-jev](https://github.com/ikermoel/open-alternative-jev),
  Apache-2.0), the closest open design: a frozen model whose option-letter logits are read
  at fixed positions, with an optional temperature and no other training. It reports 0.737
  accuracy on typed-decisions with Qwen3.6-27B (self-measured; see
  [docs/methods.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/methods.md#related-work)).
- **Cygnet** by blockbrain-ai
  ([blockbrain-ai/cygnet-recipe](https://github.com/blockbrain-ai/cygnet-recipe), MIT), the
  closest design: a frozen Gemma-4-12B-it with one temperature, joint leader of the live
  JevBench board as of 29 September 2026.
- **Kev** by jaredpalmer ([jaredpalmer/kev](https://github.com/jaredpalmer/kev), Apache-2.0)
  and **decider-4b** by Mapika ([Mapika/decider-4b](https://huggingface.co/Mapika/decider-4b),
  Apache-2.0),
  open decision models that also report calibration on data their fit never saw.
- **meraGPT Decider 1**, **Featherless Simple Jev** and the authors of the typed-decisions
  benchmark (LocalLLaMA), whose leaderboard gives the meraGPT, Featherless, Jev and specialist
  figures above.
- **Laya** by convaiinnovations
  ([convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya), package version
  0.3.20, Apache-2.0), an open decision model built on ModernBERT (Warner et al. 2024) and
  aimed at the same kind of typed decisions.
- **CLM-v0.1-8B** by Contrastive-LM
  ([Contrastive-LM/CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B),
  Apache-2.0), a related and much larger effort: two small projection heads on a frozen
  Qwen3-8B that score each candidate answer against the state by contrastive similarity,
  trained on about 91 million examples and aimed at agent decisions and verification. Its model
  card reports parity with Jev on computer-use, gaming and tool-calling tasks; I have not run
  it on judgly's benchmark (see [docs/methods.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/methods.md#related-work)).
- **Nimble** by Bespoke Labs and Maheswaran Sathiamoorthy
  ([bespokelabs/Bespoke-Nimble-9B](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B),
  weights Apache-2.0; [bespokelabsai/nimble](https://github.com/bespokelabsai/nimble); the
  revision Ollama serves appears to be release v3-12026) and **Tev1** by Together AI
  ([togethercomputer/Tev1-4B-experimental](https://huggingface.co/togethercomputer/Tev1-4B-experimental),
  [togethercomputer/Tev1-0.8B-experimental](https://huggingface.co/togethercomputer/Tev1-0.8B-experimental);
  code MIT, copyright the open-jev contributors, at
  [togethercomputer/tev1](https://github.com/togethercomputer/tev1); the weights' licence
  described as being finalized; introduced in Together AI's blog post by Hassan El Mghari), open
  decision models that were run on judgly's test
  tiers for the comparison above, and **Ollama** ([ollama/ollama](https://github.com/ollama/ollama),
  MIT), which served them. Their weights are not redistributed here; only their answers
  (probabilities) are committed, in docs/results/external-comparison. Both models are fine-tunes
  of Qwen3.5 by the Qwen team.
- **pydantic** and **huggingface_hub**, which the Python package uses, and **uv** and
  **scikit-build-core**, which build it. The pipeline and tests also use Hugging Face
  **datasets**, **NumPy**, **PyYAML** and **pytest**, and the native build uses **CMake** and
  **Ninja**. The **Hugging Face Hub** hosts the models and most of the datasets.
- **The datasets** the heads are fitted and evaluated on, and their authors. Each one is listed
  with its licence, pinned revision and citation in
  [docs/reproduce.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/reproduce.md#data-sources).
- **The papers behind the method**: reading and calibrating answer-letter probabilities on
  multiple-choice questions (Kadavath et al. 2022), option-order bias and averaging over
  option orders (Zheng et al. 2024; Pezeshkpour and Hruschka 2024), contextual calibration
  (Zhao et al. 2021), calibration of modern neural networks and temperature scaling (Guo et al.
  2017), proper scoring rules (Gneiting and Raftery 2007), and the observation in the GPT-4
  technical report (OpenAI 2023) that post-training makes models overconfident. See
  [docs/methods.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/methods.md#references).

The full licence notices are in [NOTICE](https://github.com/judgly/judgly/blob/v0.2.0/NOTICE) and [LICENSES/](https://github.com/judgly/judgly/tree/v0.2.0/LICENSES).

## Citation

If judgly is useful in your work, the metadata in [CITATION.cff](https://github.com/judgly/judgly/blob/v0.2.0/CITATION.cff) gives a
citation (GitHub shows it under "Cite this repository"). Please also cite llama.cpp, the model
you used, and the datasets behind any number you quote.

## Licence

The code is under the Apache License 2.0 ([LICENSE](https://github.com/judgly/judgly/blob/v0.2.0/LICENSE), [NOTICE](https://github.com/judgly/judgly/blob/v0.2.0/NOTICE)). Model weights
are not included; they are downloaded under their own licences. The general head is under
Apache-2.0. The stance head is under CC-BY-SA-4.0 because it is fitted on share-alike data
(MNLI, VitaminC, FEVER, SNLI and SciNLI). The results snapshot and figures are under CC-BY-4.0; see
[docs/licences.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/licences.md).
