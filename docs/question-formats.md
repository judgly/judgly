# Question formats

judgly answers three types of question. Each is a pydantic model in `judgly` and a JSON object
in the C API.

## Choice

```python
Choice(instructions="Which team should handle this message?",
       options={"shipping": "Late, lost or damaged deliveries",
                "returns": "Returns and refunds",
                "tech": "Products that do not work"})
```

- 2 to 26 options. The model sees them as lettered options (A, B, C, ...) with their texts.
- Keys are for you; texts are for the model. Write texts that a careful reader could decide
  between.
- A list of strings, `options=["kitchen", "garage", "bedroom"]`, uses each string as key and
  text.
- Answer: `probs` (a probability per key, summing to 1) and `top`.

Cyclic orders of the options are asked and the results averaged: with the built-in packs' setting
(`max_rotations: 4`) all of them when there are four options or fewer, and four evenly spaced
ones otherwise, so with more than four options some position bias can remain. A question with K
options costs min(K, `max_rotations`) short branches.

## Binary (yes or no)

```python
Binary(instructions="Does the customer ask for a refund?")
```

- Asked as a two-option choice between true and false, in both orders. This is always on.
- Answer: `p_true` and `top` (`True` or `False`).
- Phrase it as a question or a statement to check. Both work; keep one style within a
  workflow, because the head's calibration was measured on a mix.

## Score

```python
Score(instructions="How positive is this review, from 1 (very negative) to 5 (very positive)?",
      levels=5)
```

- 2 to 9 levels. Name the ends of the scale in the instructions.
- Answer: `probs` (a list, level 1 first), `mean` (the expected level) and `top` (the most
  likely level).
- Levels keep their natural order: level 1 is always shown in slot A, level 2 in slot B, and
  so on, and a score question is asked once (`n_rotations` 1, `rotation_spread` 0), whatever
  the `rotations` setting. A scale shifted cyclically across the letter slots (A = 3, B = 4,
  ...) is read against its own order, and the head's per-slot bias could then not correct the
  model's preference for some levels. That preference is left to the head, which learns one
  bias per level with every level weighted equally in its fit, so it learns no prior over levels
  from the fit data (see [methods](methods.md#the-heads)).

## The `format` field and heads

Every question may carry `format="..."`. The engine uses the head fitted for that format if the
pack has one, and otherwise the pack's fallback head `"*"`. The answer reports which one it
used in `head_format`.

The built-in packs define two formats, and ship a head for each:

| format | for | fitted on | evaluated on |
|---|---|---|---|
| `"*"` (general; the default) | any choice, yes/no or score question | knowledge, reading, moderation, preference, emotion, intent, helpfulness (HelpSteer2), contracts (LEDGAR) and generated tasks | held-out families: sentiment, inference, similarity, paraphrase, word sense, commonsense (dev); social bias, grammar, figurative language, indirect answers, ethics, tables, search relevance, sentence difficulty (fresh final); politeness (final-flagged); topic, biomedical, legal, maths, finance, truthfulness, ratings and yes/no reading (final-seen, seen during development); typed-decisions and JevBench (bench) |
| `"stance"` | how a piece of evidence bears on a claim | MNLI, SNLI, WANLI, VitaminC, FEVER, SciNLI | ANLI, Climate-FEVER, COVID-Fact (dev); Check-COVID (fresh final); HealthFC (final-flagged); HealthVer (final-seen) |

### Stance

On the fresh final tier (Check-COVID, n = 1,343 pairs), the stance heads reached ECE 0.048
[0.034, 0.072] (Gemma 4 12B) and 0.052 [0.034, 0.077] (Qwen3-4B), around the bar of 0.05, but
**both missed the dev-tier bar** (ECE 0.122 and 0.209 against 0.08, worst on scientific
abstracts), and on HealthVer (final-seen) head ECE was 0.067 and 0.100
([README results](../README.md#results)). The general head (leave `format` unset) was not
scored on stance questions, so whether it would do better is not known.

The stance head is fitted on states of the form

```text
Claim: <the claim>

Evidence: <the passage>
```

with three options keyed `supports`, `contradicts` and `no_bearing`. Keep that state shape and
those keys. The option texts and the instruction may be worded differently: the fitting data
used several wordings and shuffled option orders, for example:

```python
Choice(format="stance",
       instructions="How does the evidence bear on the claim?",
       options={"supports": "The evidence supports the claim",
                "contradicts": "The evidence contradicts the claim",
                "no_bearing": "The evidence has no bearing on the claim"})
```

On the fresh final tier (Check-COVID, n = 1,343), both stance heads' most common error was
calling no-bearing evidence "contradicts" (95 and 116 of 442 no-bearing pairs, Gemma 4 12B and
Qwen3-4B); on HealthFC (final-flagged) the same error was far more common (217 and 275 of 422);
see the model cards ([Gemma 4 12B](model-cards/gemma4-12b-q8.md#results-stance-format),
[Qwen3-4B](model-cards/qwen3-4b-q8.md#results-stance-format)).

## Writing good questions

- **Ask about the state.** The model answers from the text plus what it already knows. If the
  answer is not in the text, say what to do (for example, add an option "not stated").
- **Make options exhaustive and exclusive.** Probabilities are spread over the options you
  give. If the true answer is missing, the model still has to pick one. Add "other" or "none of
  these" where it can happen.
- **Keep instructions short and concrete.** One question per question. Split compound
  questions ("Is it urgent and about billing?") into two.
- **Check `slot_mass`.** A low slot mass (well below 0.9) means the model did not want to
  answer with one of the letters. Rephrase.
- **Check `rotation_spread`.** A large spread means the answer depends on the option order.
  The averaged answer is still the best estimate, but treat it with care.
- **Calibrate on your own data** before trusting the probabilities in a new workflow
  ([calibration.md](calibration.md)).
