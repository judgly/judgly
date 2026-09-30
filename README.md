<p align="center"><img src="https://raw.githubusercontent.com/judgly/judgly/v0.2.0/docs/assets/banner.svg" alt="judgly: calibrated, deterministic judgments from an open language model" width="720"></p>

# judgly

judgly asks an open language model typed questions about a piece of text (pick one of these
options, yes or no, a rating from 1 to 5) and returns a probability for every allowed answer,
on your own Mac.

judgly is not a new model, and no language model was trained for it. It runs Google's Gemma 4
12B or Alibaba's Qwen3-4B, unchanged apart from the 8-bit quantization of the published GGUF
files, locally through llama.cpp, and adds a thin layer on top:
typed questions, reading the probabilities of the answer letters instead of generating text,
averaging them over up to four orders of the options, and a calibration step. The calibration
step is temperature scaling (Guo et al. 2017), with one temperature per question type, or, for
Gemma 4 12B on general questions, a small fitted head (H2); both were fitted on public data. The
idea of typed, calibrated "System One" decisions comes from Jev, TypeSafe's commercial decision
model; judgly is an independent open take on that idea, built from well-known pieces, and is not
affiliated with or endorsed by TypeSafe, Google or Alibaba Cloud (the Qwen team). Full credits
are at the end of this page.

**Status:** early (0.x). macOS on Apple Silicon only. The API may still change.

## Results at a glance

<p align="center"><img src="https://raw.githubusercontent.com/judgly/judgly/v0.2.0/docs/assets/results/compare-tiers.svg" alt="Accuracy against ECE on six test sets for Nimble 9B, Tev1 4B and Tev1 0.8B, as served and with the same calibration as judgly, and judgly's two packs" width="860"></p>

judgly and three open decision models served by Ollama (Nimble 9B, Tev1 4B and Tev1 0.8B) on
the same held-out items of judgly's test sets: accuracy (up is better) against calibration error
(ECE, left is better), with 95% intervals. Grey hollow points are the Ollama models as served;
filled grey points are the same models given judgly's kind of calibration step (temperature
scaling, Guo et al. 2017: one temperature per question type, fitted by the same code on the same
sample of 1,500 general and 500 stance items from judgly's training split). Blue is judgly with
its default calibration as shipped, pale blue judgly without it. judgly's blue points were not
fitted on that sample: its shipped calibration was fitted by its own trainer on the whole train
split (6,300 general and 14,783 stance items, 4 to 30 times the sample), and for Gemma 4 12B on
general questions it is H2, not the temperature. judgly's temperature refitted on the sample,
which shares the sample and the code with the grey points, is in the tables below, not in the
figure.

- **Accuracy comes from the base model.** judgly's calibration does not raise it. judgly Gemma
  4 12B, the largest model compared, was the most accurate or level with the most accurate on
  every set: that is Gemma 4 12B's accuracy. At similar size, the trained decision models matched
  or beat judgly Qwen3-4B on several sets (stance, typed-decisions, and Nimble 9B on JevBench).
- **judgly's calibration lead came from its calibration step.** Against the models as served,
  judgly's defaults had the lowest calibration error on most sets, but not all: as served, Tev1 4B
  had a lower ECE than both judgly defaults on stance final (0.041 against 0.087 and 0.093), and
  Nimble 9B and Tev1 4B were better calibrated than judgly Qwen3-4B on typed-decisions. Given the
  same temperature fitted on the same data, Nimble 9B and Tev1 land in the same range as judgly on
  most sets, sometimes better and sometimes worse. judgly's own temperature, refitted on that same
  sample, had a higher ECE than all three calibrated external models on stance final and on general
  final-flagged.
- **The temperature usually carries over to unseen kinds of questions, but not always.** It made
  Nimble 9B and Tev1 4B worse calibrated on typed-decisions (ECE and Brier score), raised their
  Brier scores on JevBench although their ECE fell there, and made Tev1 4B slightly worse on the
  final tiers, where it was already well calibrated as served. Some of the ECE gains are nearly
  flat answers at the search's upper bound for the temperature (below).
- **judgly Gemma 4 12B is the slowest per request** (below; judgly asks each question in up to
  four option orders, in process; the Ollama models read it once, over HTTP).

These are judgly's own test sets, with the caveats listed under
[Comparison with dedicated decision models](https://github.com/judgly/judgly/blob/v0.2.0/README.md#comparison-with-dedicated-decision-models),
which has the tables. Every accuracy and calibration number for judgly on this page can be
recomputed from the per-item results committed in docs/results/.

<p align="center"><img src="https://raw.githubusercontent.com/judgly/judgly/v0.2.0/docs/assets/results/compare-timing.svg" alt="Median seconds per request with the 95th percentile, for Tev1 0.8B, Tev1 4B, judgly Qwen3-4B, Nimble 9B and judgly Gemma 4 12B" width="640"></p>

Median time per request (bar) and 95th percentile (whisker) on 100 items, one request at a time,
on an Apple M3 Max: Tev1 0.8B 0.08 s, Tev1 4B 0.29 s, judgly Qwen3-4B 0.32 s, Nimble 9B 0.53 s,
judgly Gemma 4 12B 0.96 s.

## How it works

<p align="center"><img src="https://raw.githubusercontent.com/judgly/judgly/v0.2.0/docs/assets/how-it-works.svg" alt="The state is read once and cached; each question branches from the cached state; the option-letter probabilities go through a calibration step (a temperature per question type, or a small head) to become calibrated probabilities." width="860"></p>

1. The model reads your text (the *state*) once, and llama.cpp caches it.
2. Every question branches from that cache on its own, so questions never see each other.
3. Each question lists its allowed answers as letters. judgly reads the model's probabilities
   for those letters at one position and generates no text. It asks each question in at most
   four cyclic orders of the options (all of them when there are four or fewer) and averages,
   which reduces the effect of option order; with more than four options some position bias
   can remain.
4. A calibration step turns the letter probabilities into calibrated ones: by default one
   temperature per question type (temperature scaling, Guo et al. 2017), and for Gemma 4 12B on
   general questions a small fitted *head*, H2 (a few hundred thousand numbers, against billions
   in the model). Neither changes the model.

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
> than H2's 0.078 and 0.183 ([docs/calibration.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/calibration.md#results-of-the-two-options)). Stance calibration varies a lot between kinds of
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
0.052) and missed the dev-tier bar ([docs/calibration.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/calibration.md#results-of-the-two-options)). A question with `format` unset, or with a format the pack has no head
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

## Calibration options

Each pack ships two calibration options per format, both fitted on the train split of judgly's
fit tier (6,300 general and 14,783 stance items from public, licence-checked sources): **H2**, a
small head released in 0.1.0 that can also use the model's hidden state at the answer position, and the
**per-type temperature**, one temperature per question type (temperature scaling as introduced
by Guo et al. 2017, applied to the probabilities after they are averaged over the option orders).
The temperature never changes which answer is on top; H2 can. The defaults were set by a
pre-registered comparison of the two on a tier nothing had read before:

| pack | general questions | stance questions |
|---|---|---|
| `gemma4-12b-q8` | H2 (the temperature was not confirmed) | temperature (confirmed) |
| `qwen3-4b-q8` | temperature (confirmed) | temperature (confirmed) |

`Engine.load(pack)` uses these defaults; `calibration="h2"`, `"temperature"` or `"raw"` chooses
one for every format. [docs/calibration.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/calibration.md#results-of-the-two-options)
has the results of both options (including the 0.1.0 H2 results on the fresh final tier), and
[docs/calibration-options.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/calibration-options.md)
how they were fitted and compared, with every number.

## Comparison with dedicated decision models

Ollama 0.35.0 serves three open models trained for this kind of typed decision: **Nimble 9B**
(`nimble:9b`, Bespoke Labs, a fine-tune of Qwen3.5-9B) and **Tev1 4B** and **Tev1 0.8B**
(`tev1:4b`, `tev1:0.8b`, Together AI, fine-tunes of Qwen3.5-4B and Qwen3.5-0.8B). Each was asked
every item of judgly's held-out test sets (17,482 items) through Ollama's `/v1/systemone`
endpoint, with default tags and settings, as a user would run them, and their probabilities were
scored by the same code as judgly's. The protocol, runner and scorer were frozen before any of
these models answered a test item. judgly was not run again: its rows are the pack defaults from
the committed per-item dumps. The comparison is descriptive: no criterion was set, and every
number is reported. The record is in
[docs/results/external-comparison/](https://github.com/judgly/judgly/tree/v0.2.0/docs/results/external-comparison),
and `make compare-score` rebuilds every result from the committed answers, byte for byte.

**The same calibration for every system.** As served, the Ollama models have no calibration step,
while judgly's defaults do. A control with its own protocol
([calibrated/](https://github.com/judgly/judgly/tree/v0.2.0/docs/results/external-comparison/calibrated))
therefore gave each of them judgly's kind of temperature step: one temperature per question type
(temperature scaling, Guo et al. 2017), fitted by the same code on the same sample of judgly's
training split (1,500 general and 500 stance items; the models refused 49 general items that
have no text, so theirs were fitted on 1,451). The protocol's files were hashed at 09:24 by the
author's own record, the same minute the run log starts; they were first committed together with
the results, so nothing in the repository timestamps the freeze independently. judgly's H2
cannot be fitted for the external models, because Ollama does not expose their internal state.
judgly's own temperature was refitted on the same sample by the same code (rows "temperature
refitted on the sample" below). Those rows, not judgly's defaults, are the like-for-like
comparison: judgly's defaults were fitted by its own trainer on the whole train split (6,300
general and 14,783 stance items), and for Gemma 4 12B general questions they are H2. Refitting on
the sample raised judgly's own ECE on general final from 0.017 (Gemma 4 12B's shipped
temperature; its default, H2, gave 0.020) and 0.034 (Qwen3-4B) to 0.038 and 0.044, and on stance
final from 0.087 and 0.093 to 0.114 and 0.111. On that footing (point values; the record has
paired intervals only against judgly's defaults), the calibrated external models' ECE was below
both refitted judgly packs on stance final and general final-flagged, and for Tev1 4B and Tev1
0.8B on stance final-flagged; above both for Tev1 4B on stance confirm and general final, Tev1
0.8B on stance confirm, and Nimble 9B on typed-decisions and stance final-flagged; and otherwise
between the two, or within 0.002 of them (Tev1 on JevBench). Four fitted temperatures (Tev1 4B
score, Tev1 0.8B yes/no and score, and judgly Qwen3-4B score when refitted on the sample) reached
the search's upper bound, 54.6; at that temperature the answers of that type are almost flat, so
a low ECE for them means uninformative answers, not good ones.

**The tables.** Accuracy is the top answer against the gold label (a temperature does not change
it), ECE uses the top label and ten bins, and the Brier score is summed over the options against
the one-hot label (0 is best, 2 the worst). judgly's "default" rows are its defaults as shipped,
which already include its calibration step; its "temperature refitted on the sample" rows are its
equal calibration (their accuracy differs from the default's only where the default is H2, which
can change the top answer: Gemma 4 12B on general questions). Each external model is scored on
the items it answered and judgly on every item; they differ only on JevBench, where Tev1 refused 36 items longer than its 2,050-token
context. typed-decisions and JevBench are the two sources of the bench tier, which the records
score together; `make compare-figures` splits them. Scored against the gold label, the bench
numbers differ slightly from the benchmarks' own scoring in docs/calibration.md (judgly Gemma 4
12B's JevBench accuracy is 0.844 here and 0.840 there). The 95% intervals, paired differences, the final-flagged tiers and judgly's other
calibrations are in
[docs/methods.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/methods.md#comparison-with-dedicated-decision-models).

**confirm, general (5 families new to judgly)**

| system | items | accuracy | ECE as served | ECE, equal calibration | Brier as served | Brier, equal calibration |
|---|---|---|---|---|---|---|
| Nimble 9B | 2,500 | 0.481 | 0.229 | 0.059 | 0.702 | 0.607 |
| Tev1 4B | 2,500 | 0.479 | 0.142 | 0.065 | 0.681 | 0.620 |
| Tev1 0.8B | 2,500 | 0.378 | 0.154 | 0.043 | 0.701 | 0.678 |
| judgly Gemma 4 12B, default | 2,500 | 0.509 | 0.051 | — | 0.562 | — |
| judgly Qwen3-4B, default | 2,500 | 0.484 | 0.047 | — | 0.597 | — |
| judgly Gemma 4 12B, temperature refitted on the sample | 2,500 | 0.509 | — | 0.072 | — | 0.578 |
| judgly Qwen3-4B, temperature refitted on the sample | 2,500 | 0.484 | — | 0.020 | — | 0.596 |

**confirm, stance (ClimateCheck)**

| system | items | accuracy | ECE as served | ECE, equal calibration | Brier as served | Brier, equal calibration |
|---|---|---|---|---|---|---|
| Nimble 9B | 1,780 | 0.595 | 0.264 | 0.068 | 0.657 | 0.545 |
| Tev1 4B | 1,780 | 0.630 | 0.185 | 0.104 | 0.542 | 0.495 |
| Tev1 0.8B | 1,780 | 0.451 | 0.304 | 0.204 | 0.794 | 0.703 |
| judgly Gemma 4 12B, default | 1,780 | 0.655 | 0.052 | — | 0.476 | — |
| judgly Qwen3-4B, default | 1,780 | 0.568 | 0.106 | — | 0.581 | — |
| judgly Gemma 4 12B, temperature refitted on the sample | 1,780 | 0.655 | — | 0.044 | — | 0.474 |
| judgly Qwen3-4B, temperature refitted on the sample | 1,780 | 0.568 | — | 0.092 | — | 0.576 |

**final, general (8 families)**

| system | items | accuracy | ECE as served | ECE, equal calibration | Brier as served | Brier, equal calibration |
|---|---|---|---|---|---|---|
| Nimble 9B | 7,879 | 0.674 | 0.129 | 0.043 | 0.450 | 0.429 |
| Tev1 4B | 7,879 | 0.657 | 0.052 | 0.070 | 0.431 | 0.442 |
| Tev1 0.8B | 7,879 | 0.504 | 0.086 | 0.040 | 0.588 | 0.570 |
| judgly Gemma 4 12B, default | 7,879 | 0.737 | 0.020 | — | 0.348 | — |
| judgly Qwen3-4B, default | 7,879 | 0.656 | 0.034 | — | 0.440 | — |
| judgly Gemma 4 12B, temperature refitted on the sample | 7,879 | 0.760 | — | 0.038 | — | 0.326 |
| judgly Qwen3-4B, temperature refitted on the sample | 7,879 | 0.656 | — | 0.044 | — | 0.448 |

**final, stance (Check-COVID)**

| system | items | accuracy | ECE as served | ECE, equal calibration | Brier as served | Brier, equal calibration |
|---|---|---|---|---|---|---|
| Nimble 9B | 1,343 | 0.817 | 0.113 | 0.061 | 0.312 | 0.297 |
| Tev1 4B | 1,343 | 0.826 | 0.041 | 0.048 | 0.264 | 0.265 |
| Tev1 0.8B | 1,343 | 0.577 | 0.111 | 0.027 | 0.556 | 0.536 |
| judgly Gemma 4 12B, default | 1,343 | 0.827 | 0.087 | — | 0.280 | — |
| judgly Qwen3-4B, default | 1,343 | 0.778 | 0.093 | — | 0.357 | — |
| judgly Gemma 4 12B, temperature refitted on the sample | 1,343 | 0.827 | — | 0.114 | — | 0.288 |
| judgly Qwen3-4B, temperature refitted on the sample | 1,343 | 0.778 | — | 0.111 | — | 0.362 |

**typed-decisions**

| system | items | accuracy | ECE as served | ECE, equal calibration | Brier as served | Brier, equal calibration |
|---|---|---|---|---|---|---|
| Nimble 9B | 2,000 | 0.702 | 0.052 | 0.181 | 0.415 | 0.500 |
| Tev1 4B | 2,000 | 0.613 | 0.038 | 0.127 | 0.485 | 0.550 |
| Tev1 0.8B | 2,000 | 0.434 | 0.186 | 0.074 | 0.674 | 0.647 |
| judgly Gemma 4 12B, default | 2,000 | 0.700 | 0.028 | — | 0.401 | — |
| judgly Qwen3-4B, default | 2,000 | 0.576 | 0.137 | — | 0.578 | — |
| judgly Gemma 4 12B, temperature refitted on the sample | 2,000 | 0.702 | — | 0.044 | — | 0.407 |
| judgly Qwen3-4B, temperature refitted on the sample | 2,000 | 0.576 | — | 0.169 | — | 0.610 |

**JevBench, public items**

| system | items | accuracy | ECE as served | ECE, equal calibration | Brier as served | Brier, equal calibration |
|---|---|---|---|---|---|---|
| Nimble 9B | 231 | 0.779 | 0.118 | 0.072 | 0.283 | 0.299 |
| Tev1 4B | 195 | 0.790 | 0.104 | 0.099 | 0.306 | 0.322 |
| Tev1 0.8B | 195 | 0.672 | 0.093 | 0.079 | 0.455 | 0.451 |
| judgly Gemma 4 12B, default | 231 | 0.844 | 0.067 | — | 0.221 | — |
| judgly Qwen3-4B, default | 231 | 0.693 | 0.084 | — | 0.384 | — |
| judgly Gemma 4 12B, temperature refitted on the sample | 231 | 0.836 | — | 0.105 | — | 0.225 |
| judgly Qwen3-4B, temperature refitted on the sample | 231 | 0.693 | — | 0.049 | — | 0.406 |
| judgly Gemma 4 12B, default, on Tev1's items | 195 | 0.882 | 0.053 | — | 0.188 | — |
| judgly Qwen3-4B, default, on Tev1's items | 195 | 0.759 | 0.092 | — | 0.333 | — |
| judgly Gemma 4 12B, temperature refitted on the sample, on Tev1's items | 195 | 0.867 | — | 0.098 | — | 0.196 |
| judgly Qwen3-4B, temperature refitted on the sample, on Tev1's items | 195 | 0.759 | — | 0.081 | — | 0.358 |

On the final-flagged tiers (politeness and HealthFC, fresh families with known caveats), judgly's
defaults were the more accurate on general questions (0.512 and 0.492 against 0.339 to 0.421). On
stance (HealthFC) judgly Gemma 4 12B's default was the more accurate (0.750 against 0.474 to
0.689); judgly Qwen3-4B's (0.718) was ahead of Nimble 9B and Tev1 0.8B and level with Tev1 4B
(0.689; paired difference -0.029, interval -0.059 to +0.000). With the equal calibration, ECE on
general final-flagged was 0.042 (Nimble 9B), 0.082 (Tev1 4B) and 0.004 (Tev1 0.8B, nearly flat
answers at the temperature bound, mean top probability 0.335), against 0.106 and 0.037 for
judgly's defaults and 0.107 and 0.104 for its refitted temperature; on stance final-flagged it
was 0.215, 0.062 and 0.039, against 0.069 and 0.052 for judgly's defaults and 0.093 and 0.065
refitted. Nimble 9B on HealthFC is the largest calibration gap that the equal calibration left
(+0.146 [+0.094, +0.181] against judgly Gemma 4 12B's default).

**How judgly was fitted, and the test sets.** judgly's language models are never trained. Its
calibration (H2 and the per-type temperatures) was fitted on the train split of its fit tier
only ([data/registry.yaml](https://github.com/judgly/judgly/blob/v0.2.0/data/registry.yaml) lists
the sources); none of the test sets was used for fitting. The **confirm** tiers (five general
families never used before, and ClimateCheck for stance) were built last; each judgly model read
them once, for the pre-registered confirmation of the temperature, and each pack's default was
set from that read (the temperature where it was confirmed, H2 otherwise). Nothing was fitted on
them, but judgly's default rows there were chosen by their own result; H2, fixed beforehand, had
ECE 0.076 (Qwen3-4B) on general confirm and 0.078 (Gemma 4 12B) and 0.183 (Qwen3-4B) on stance
confirm. The **final** tiers were fresh at the 0.1.0 release and were read again by the
exploratory analyses that led to the temperature option. **typed-decisions** (2,000 decisions over
400 cases, whose gold is a teacher model's output) and the 231 public **JevBench** items are
external benchmarks that judgly read in its release run and in two exploratory analyses. The
external models read every item once, in this comparison.

**What is known of the other models' training data.** As their providers document it, none of
the confirm, final or final-flagged items is in either model's fine-tuning data. Nimble is a LoRA
fine-tune of Qwen3.5-9B; the checkpoint Ollama serves appears to be release v3-12026, whose
training components include public banking77, multinli, boolq, ag_news, dbpedia and trec, as well
as local components that are not documented and cannot be checked. Nimble's schema configuration
lists a registered evaluation "jevbench-eval" (534 items; judgly uses the 231 public JevBench
items) and marks its own "bespoke-eval" set as "reporting_only"; whether JevBench items informed
training or model selection is not stated. Tev1 is a fine-tune of Qwen3.5-4B and Qwen3.5-0.8B on
the train splits of MultiNLI, BoolQ, Banking77, AG News and SST-5 plus synthetic policy, routing
and research data, and its authors state that no Jev data and no JevBench items were used.
typed-decisions is not named in either model's documented data. MultiNLI and Banking77 are also
among the sources of the control's training sample (112 and 18 items).

**Why this comparison may still favour judgly.**

- **The test sets are ours.** judgly's tiers were chosen and built by us, for judgly.
- **Calibration.** As served, the external models have no calibration step, and their providers
  do not present these probabilities as calibrated (Tev1's README: "Logprobs are model
  preferences, not calibrated confidence"; Nimble's model card: "This checkpoint has not had a
  separate temperature fit"). The control above gives them judgly's temperature step, fitted on
  judgly's training data; a calibration fitted on their own training data, or on data closer to
  the test sets, might do better or worse. judgly's H2 could not be given to them.
- **Option orders.** judgly averages each question over up to four option orders, a test-time
  ensemble known to improve both accuracy and calibration; the external models read each question
  once.
- **Model size.** judgly's Gemma 4 12B has more parameters than any of the external models (9B,
  4B and 0.8B). The like-for-like comparison by size is judgly Qwen3-4B against Tev1 4B.
- **Tev1's prompt.** Ollama's `/v1/systemone` sends a `{"context", "schema"}` prompt, while Tev1
  was trained on `{"state", "question", "options"}`, so Tev1 is measured as Ollama serves it,
  which may understate what it does with its own format. A run with its native format was
  considered and declined at 16:54, while the Tev1 4B run (started at 16:47) was in progress and
  before any of its answers were looked at.
- **Tev1's context.** 36 JevBench items were refused as too long; judgly and Nimble answered them.
- **Timing** was measured differently in kind (below).

**Single-request timing** (`final/timing.json`, taken after the comparison run and not part of
its frozen protocol): 100 items (the first 50 of each confirm tier by the SHA-256 of their id), one
question per request, one request at a time, after one uncounted warm-up request per system, on
an Apple M3 Max (64 GB). The 95th percentile is the 96th of the 100 sorted times.

| system | median (s) | 95th percentile (s) | mean (s) |
|---|---|---|---|
| Nimble 9B | 0.533 | 1.039 | 0.599 |
| Tev1 4B | 0.289 | 0.567 | 0.321 |
| Tev1 0.8B | 0.078 | 0.153 | 0.087 |
| judgly Gemma 4 12B (default) | 0.960 | 1.732 | 0.994 |
| judgly Qwen3-4B (default) | 0.321 | 0.557 | 0.333 |

judgly ran in-process through its Python API and asked each question in up to four option
orders; the Ollama models were asked over HTTP and read each question once.

**In short.** judgly Gemma 4 12B was clearly the most accurate on general questions and
JevBench and level with the best dedicated model on stance (Tev1 4B) and typed-decisions (Nimble
9B). The dedicated models of 4 to 9B were level with judgly Qwen3-4B on general questions (Nimble
9B slightly ahead on general final) and more accurate on the confirm and final stance tiers, on
typed-decisions and, for Nimble 9B, on JevBench. As served, Tev1 4B had a lower ECE than both
judgly defaults on stance final, and Nimble 9B and Tev1 4B were better calibrated than judgly
Qwen3-4B on typed-decisions. With the same temperature, the dedicated models' ECE was level with
judgly's defaults on general confirm. On stance final all three were better calibrated than both
judgly packs; on stance confirm Nimble 9B and Tev1 4B fell between judgly's two packs, and Tev1
0.8B was worse (0.204). On general final all three were above judgly Gemma 4 12B's default;
against judgly Qwen3-4B's, Tev1 4B was clearly above, Nimble 9B marginally
(+0.009 [+0.001, +0.022]) and Tev1 0.8B level (+0.006 [-0.004, +0.018]). judgly's own temperature, refitted on the
same sample, also rose above its shipped calibration there (0.038 and 0.044 against 0.017 for
Gemma 4 12B's shipped temperature and 0.034 for Qwen3-4B's), so much of that remaining gap comes
with the fitting sample, not with H2. Nimble 9B stayed far above both judgly defaults on stance
final-flagged (HealthFC, 0.215), and the temperature made Nimble 9B and Tev1 4B worse than as
served on typed-decisions. On other data, the picture may differ.

**Other systems' published figures.** The only benchmark on which several other systems report
is typed-decisions (gold is a teacher model's output; the card gives the teacher's
self-agreement as 0.735). These figures are quoted from their sources and were not reproduced
here ([docs/methods.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/methods.md#related-work)
has the sources, the calibration figures and the JevBench board):

| system | how it works | accuracy | who measured |
|---|---|---|---|
| meraGPT Decider 1 | hosted, proprietary | 0.768 | listed on the dataset card (measurer not stated) |
| Laya, checkpoint fine-tuned on this benchmark's train split | fine-tuned ModernBERT-large encoder | 0.766 | Laya's own model card |
| open-alternative-jev, Qwen3.6-27B (8-bit), two option orders / one | frozen LLM, letter logits, no training | 0.755 / 0.737 | the project itself |
| Jev 1.13.0 | hosted, commercial | 0.727 | the dataset card's authors, through TypeSafe's API |
| Featherless Simple Jev | hosted demo endpoint (model id `Qwen3.6-35B-A3B-classifier`) | 0.716 | listed on the dataset card (measurer not stated) |
| **judgly, Gemma 4 12B with H2** (its default for these questions) | frozen LLM, letter logits, fitted head | **0.700** | this repository |
| ModernBERT-base specialist | frozen encoder, classifiers fitted on the train split of the same workflows | 0.646 | the dataset card's authors |
| open-alternative-jev, Qwen3.5-4B, two option orders / one | frozen LLM, letter logits, no training | 0.595 / 0.593 | the project itself |
| **judgly, Qwen3-4B with H2** | frozen LLM, letter logits, fitted head | **0.591** | this repository |
| MiniLM-L6 specialist (22M) | frozen encoder, classifiers fitted on the train split of the same workflows | 0.587 | the dataset card's authors |
| **judgly, Qwen3-4B with its default, the temperature** | frozen LLM, letter logits, one temperature per question type | **0.576** | this repository |

The closest open design is **Cygnet**
([blockbrain-ai/cygnet-recipe](https://github.com/blockbrain-ai/cygnet-recipe), MIT): the same
Gemma-4-12B-it, frozen, with its letter probabilities and one temperature fitted on 241 items its
authors generated. It reports 203 of the 231 public JevBench items (0.879, with JevBench's own
tool); judgly Gemma 4 12B with H2 scores 0.840 on the same items with judgly's scorer, which has
not been checked against JevBench's tool.

**judgly may suit you if** you want probabilities you can check, on your own Mac, offline, with
the same numbers for the same request; if you need to know where calibration has and has not
been measured; or if licence-checked calibration and a reproducible record matter to you.
**Something else may suit you better if** you need the best accuracy per decision, low latency,
hardware other than Apple silicon, or reliable claim checking against scientific text.

## Limitations

Please read these before using judgly for anything that matters.

- **Not the most accurate option.** On typed-decisions, judgly's Gemma 4 12B with H2 (0.700) is
  below Featherless Simple Jev (0.716), Jev (0.727), open-alternative-jev's 27B (0.737 to
  0.755), Laya fine-tuned on the benchmark (0.766) and meraGPT Decider 1 (0.768); at about 4B,
  judgly's Qwen3-4B (0.591 with H2, 0.576 with its default, the temperature) is at or below
  open-alternative-jev's untrained Qwen3.5-4B (0.593 to 0.595) in accuracy and worse calibrated
  (ECE 0.126 with H2 and 0.137 with the temperature, against 0.062 to 0.118)
  ([other systems' published figures](https://github.com/judgly/judgly/blob/v0.2.0/docs/methods.md#related-work)).
  On judgly's own test tiers, the dedicated decision models Nimble 9B and Tev1 4B, run as served,
  were more accurate than judgly's Qwen3-4B on the confirm and final stance tiers and on
  typed-decisions, and Nimble 9B also on JevBench (on the flagged HealthFC stance tier judgly's
  Qwen3-4B was ahead of Nimble 9B and level with Tev1 4B)
  ([Comparison with dedicated decision models](https://github.com/judgly/judgly/blob/v0.2.0/README.md#comparison-with-dedicated-decision-models)).
- **judgly's calibration lead is its calibration step, not its models.** Against the dedicated
  decision models as served, judgly's defaults were the best calibrated on most test sets, but
  Tev1 4B was better calibrated than both judgly defaults on stance final, and Nimble 9B and Tev1 4B than judgly
  Qwen3-4B on typed-decisions; given the same temperature fitted on the same data, those models
  came close to judgly on most sets, and on stance final and general final-flagged they were better
  calibrated than judgly's own temperature refitted on the same sample
  ([Comparison with dedicated decision models](https://github.com/judgly/judgly/blob/v0.2.0/README.md#comparison-with-dedicated-decision-models)).
  The step is temperature scaling (Guo et al. 2017), or H2 for Gemma 4 12B general questions;
  anyone can apply the former to another model's probabilities.
- **A temperature fitted on one kind of question may not suit another.** judgly's temperatures
  were fitted on its training sources; on typed-decisions the same kind of temperature made two
  of the dedicated models worse calibrated (ECE and Brier score), and judgly Qwen3-4B's default is
  poorly calibrated there too (ECE 0.137; 0.169 refitted on the control's sample). On JevBench it
  lowered those two models' ECE but raised their Brier scores (Nimble 9B 0.283 to 0.299, Tev1 4B
  0.306 to 0.322). Check calibration on your own questions (below).
- **Accuracy is modest, and calibration does not raise it.** On the fresh final tier, accuracy
  with H2 was 0.737 (Gemma 4 12B) and 0.639 (Qwen3-4B) on general questions and 0.812 and 0.773
  on stance, slightly below the raw readout ([docs/calibration.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/calibration.md#results-of-the-two-options)); the temperature leaves the top
  answer, and so the accuracy, at the raw readout's (0.760, 0.656, 0.827, 0.778). Calibration
  tells you *when* to trust an answer; it does not make the model know more. The general H2
  heads cost 1.7 to 2.3 points there and changed accuracy by -2.2 to +1.5 points on
  typed-decisions, JevBench and the final-seen tier; the stance H2 heads cost 0.5 to 2.3 points
  on Check-COVID and HealthVer and 12 to 18 on HealthFC.
- **Calibration is uneven.** Answers the Gemma 4 12B H2 head gives at about 0.85 were right about
  85% of the time on the fresh final tier and more often on typed-decisions, but the Qwen3-4B H2
  head is overconfident on typed-decisions (answers at about 0.85 were right 61% of the time; ECE
  0.126, and 0.137 with Qwen3-4B's default, the temperature), and both stance H2 heads are
  overconfident on HealthVer and HealthFC. The pooled numbers hide weaker families 
  ([docs/calibration.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/calibration.md#results-of-the-two-options)), and some stay poorly calibrated: financial tweets with the Qwen3-4B H2 head (ECE
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
[docs/results/external-comparison/](https://github.com/judgly/judgly/tree/v0.2.0/docs/results/external-comparison)
with the comparison's record and its equal-calibration control (`make compare-score`,
`make compare-figures`).

## Documentation

- [Installation](https://github.com/judgly/judgly/blob/v0.2.0/docs/installation.md)
- [Usage](https://github.com/judgly/judgly/blob/v0.2.0/docs/usage.md)
- [Question formats](https://github.com/judgly/judgly/blob/v0.2.0/docs/question-formats.md)
- [Model packs](https://github.com/judgly/judgly/blob/v0.2.0/docs/model-packs.md)
- [Calibration](https://github.com/judgly/judgly/blob/v0.2.0/docs/calibration.md): what ECE and selective accuracy mean, the results of
  both calibration options, and how to check or fit calibration on your own data
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
  Apache-2.0), a close open design: a frozen model whose option-letter logits are read
  at fixed positions, with an optional temperature and no other training. It reports 0.737
  accuracy on typed-decisions with Qwen3.6-27B (self-measured; see
  [docs/methods.md](https://github.com/judgly/judgly/blob/v0.2.0/docs/methods.md#related-work)).
- **Cygnet** by blockbrain-ai
  ([blockbrain-ai/cygnet-recipe](https://github.com/blockbrain-ai/cygnet-recipe), MIT), the
  closest open design (the same model): a frozen Gemma-4-12B-it with one temperature, joint
  leader of the live JevBench board as of 29 September 2026.
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
  tiers for the comparison above (and on a sample of its training split for the equal-calibration
  control), and **Ollama** ([ollama/ollama](https://github.com/ollama/ollama),
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
  2017; judgly's per-type temperature is their method, applied per question type), proper
  scoring rules (Gneiting and Raftery 2007), and the observation in the GPT-4
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
