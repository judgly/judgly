# judgly

Calibrated, deterministic judgments from open language models.

**Status: early development.** This release (0.0.1) contains no functionality yet; it reserves
the name while the package is built. Do not depend on it.

## What judgly will do

judgly answers typed questions about a text (a choice among options, a yes or no, a rating on a
scale) with a probability for every allowed answer, without generating any text. A frozen open
language model reads the text once, every question is answered from that single reading and kept
isolated from the others, and a small calibration head, fitted on public data, turns the model's
scores into probabilities that match observed accuracy: when judgly says 0.8, it is right about
80% of the time. Answers are repeatable: the same request gives the same probabilities.

The default model is Google's Gemma 4 12B, run locally through llama.cpp.

## Planned interface (subject to change)

```python
from judgly import Engine, Choice

engine = Engine.load("gemma4-12b")
answer = engine.decide(
    state="Claim: ...\n\nEvidence: ...",
    questions={
        "stance": Choice(
            format="stance",
            instructions="How does the evidence bear on the claim?",
            options={"supports": "The evidence supports the claim",
                     "contradicts": "The evidence contradicts the claim",
                     "no_bearing": "The evidence has no bearing on the claim"},
        ),
    },
)
answer["stance"].probs   # {"supports": ..., "contradicts": ..., "no_bearing": ...}
```

## Acknowledgements

judgly runs on Google's Gemma 4 models, released under the Apache License 2.0, and on
llama.cpp and ggml by the ggml authors (MIT License). See NOTICE.

## Licence

Apache License 2.0. See LICENSE and NOTICE.
