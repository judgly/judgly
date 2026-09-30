# FAQ

**Is this a new method?**
No. judgly combines known pieces: reading answer-letter probabilities at one position
(Kadavath et al. 2022), averaging over option orders (Zheng et al. 2024; Pezeshkpour and
Hruschka 2024), contextual calibration (Zhao et al. 2021; available in the engine but off in the
release packs) and a small calibration
head in the spirit of temperature scaling (Guo et al. 2017), on top of llama.cpp and an open model. The
interface idea comes from TypeSafe's Jev. What judgly adds is a small open implementation and
an evaluation with documented leakage checks. See [methods.md](methods.md).

**Is judgly affiliated with TypeSafe, Google or the Qwen team?**
No. It is a personal hobby project. It uses Gemma and Qwen under their licences, and it does not
use any TypeSafe code or weights.

**Why not just ask the model and parse its answer?**
Generated answers give you one word, not a probability for each option, and chat-tuned models
are overconfident when asked for a confidence. Reading the letter probabilities costs no
generated tokens, gives a full distribution, and can be calibrated and measured.

**How accurate is it?**
Modest. On the fresh final tier (task families no head saw), accuracy with the head was 0.737
(Gemma 4 12B) and 0.639 (Qwen3-4B) on general questions (n = 7,879) and 0.812 and 0.773 on
stance (Check-COVID, n = 1,343), a little below the raw readout; see the
[model cards](model-cards/gemma4-12b-q8.md). With the Gemma 4 12B head it did best on social
bias questions (BBQ, 0.927, n = 1,000) and figurative language (Fig-QA, 0.882, n = 1,000), and
worst on sentence difficulty (CEFR-SP, six levels, 0.390, n = 874) and search relevance (ESCI,
0.569, n = 1,000). The point is that the probabilities tell you *when* an answer is likely
right. See the [results of the two calibration options](calibration.md#results-of-the-two-options).

**Can I trust a probability of 0.9?**
Not blindly. On the fresh final tier with the Gemma 4 12B general head, answers above 0.9 were
right 95% of the time (38% of 7,879 questions passed); with the Qwen3-4B general head, answers
above 0.95 were right only 90% of the time (11% passed), so that head is overconfident at the top
([model cards](model-cards/gemma4-12b-q8.md#selective-accuracy)). On other tiers it can be worse:
on HealthVer (final-seen), Gemma 4 12B stance answers above 0.9 were right 80% of the time. On your task, check with a few hundred
labelled cases ([own_calibration.py](../examples/own_calibration.py)) before you rely on it.

**Why was the stance head shipped if it missed a bar?**
It is still far better calibrated than the raw probabilities (fresh final-tier ECE 0.048 against
0.142 for Gemma 4 12B, 0.052 against 0.187 for Qwen3-4B), but both stance heads miss the dev-tier
bar, and the Qwen3-4B head also just misses the final-tier bar; this is stated wherever their
numbers appear. The heads are fitted only on data whose licences allow it; the stance head on
MultiNLI, SNLI, WANLI, VitaminC, FEVER and SciNLI. `heads=False` gives raw probabilities.

**Does it run on Linux, Windows or with CUDA?**
Not yet. The C code and llama.cpp are portable, but the build and the packaging are only set up
and tested for macOS on Apple Silicon.

**Can I use another model?**
Yes, by building a pack: a template, a pinned model file and a pipeline run to fit its heads.
See [model-packs.md](model-packs.md). A head fitted on one model does not work on another.

**Does it support languages other than English?**
The model may understand other languages, but all fitting and evaluation data is English, so
the calibration is only measured for English.

**Is it deterministic?**
For a fixed model file, template, heads, settings, build and hardware, yes: the same request
gives the same probabilities. Probabilities may differ slightly across hardware or llama.cpp
versions.

**Can the questions influence each other?**
Not beyond floating-point noise. Each question branches from the cached text on its own and
never sees the others; batching them together can change the order of floating-point sums, which
moved probabilities by at most 0.0002 (Gemma 4 12B) and 0.0014 (Qwen3-4B) in the release
self-tests. The self-test T5 bounds this at 0.01, and [many_questions.py](../examples/many_questions.py) shows it.

**How long can the text be?**
By default up to 16,384 tokens (half the 32,768-cell cache; the rest is for the question
branches). Longer texts are cut to their first 16,384 tokens and the answer says `truncated`.

**Where does my data go?**
Nowhere. Everything runs locally. The only network access is the one-time model download from
Hugging Face.

**How do I cite it?**
With [CITATION.cff](../CITATION.cff). Please also cite llama.cpp, the model, and the datasets
behind any number you quote.
