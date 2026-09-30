# FAQ

**Is this a new method?**
No. judgly combines known pieces: reading answer-letter probabilities at one position
(Kadavath et al. 2022), averaging over option orders (Zheng et al. 2024; Pezeshkpour and
Hruschka 2024), contextual calibration (Zhao et al. 2021; available in the engine but off in the
release packs) and, by default, temperature scaling (Guo et al. 2017) with one temperature per
question type; only Gemma 4 12B's general questions use a small fitted head (H2) by default. All
of this runs on top of llama.cpp and an open model (Gemma 4 12B or Qwen3-4B, unchanged). The
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
Modest. On the fresh final tier (task families no head saw), accuracy with the default
calibration was 0.737 (Gemma 4 12B, H2) and 0.656 (Qwen3-4B, temperature) on general questions
(n = 7,879) and 0.827 and 0.778 on stance (Check-COVID, n = 1,343, temperature). A temperature
leaves the raw readout's top answer, and so its accuracy, unchanged; H2 can change it and was a
little below the raw readout (Gemma 4 12B general 0.737 against 0.760; with H2 instead of the
defaults, Qwen3-4B general 0.639 and stance 0.812 and 0.773); see the
[model cards](model-cards/gemma4-12b-q8.md). With the Gemma 4 12B H2 head it did best on social
bias questions (BBQ, 0.927, n = 1,000) and figurative language (Fig-QA, 0.882, n = 1,000), and
worst on sentence difficulty (CEFR-SP, six levels, 0.390, n = 874) and search relevance (ESCI,
0.569, n = 1,000). The point is that the probabilities tell you *when* an answer is likely
right. See the [results of the two calibration options](calibration.md#results-of-the-two-options).

**Can I trust a probability of 0.9?**
Not blindly. On the fresh final tier with the default calibration for general questions, Gemma
4 12B's answers above 0.9 were right 95% of the time (38% of 7,879 questions passed; H2) and
Qwen3-4B's 91% (22% passed; temperature), and Qwen3-4B's answers above 0.95 were right only 91% of
the time (14% passed), so it is overconfident at the top (with Qwen3-4B's H2 head, not the
default, 90% at 0.95, 11% passed). With the default stance calibration, the temperature, no
stance answer reached 0.9 on Check-COVID or HealthVer; with the H2 stance head (not the default),
Gemma 4 12B's stance answers above 0.9 on HealthVer (final-seen) were right 80% of the time
([model cards](model-cards/gemma4-12b-q8.md#selective-accuracy)). On your task, check with a few
hundred labelled cases ([own_calibration.py](../examples/own_calibration.py)) before you rely on
it.

**Why is stance calibration shipped if it misses a bar?**
The default stance calibration is the per-type temperature. On the fresh final tier (Check-COVID)
its ECE was 0.087 (Gemma 4 12B) and 0.093 (Qwen3-4B), well above the 0.05 bar but better than the
raw probabilities (0.142 and 0.187); on the untouched confirm tier (ClimateCheck) it was 0.052 and
0.106. The H2 stance heads (not the default) reached 0.048 and 0.052 on Check-COVID but missed the
dev-tier bar (0.122 and 0.209 against 0.08) and did worse on ClimateCheck (0.078 and 0.183). This
is stated wherever these numbers appear. Both options are fitted only on data whose licences
allow it; for stance, MultiNLI, SNLI, WANLI, VitaminC, FEVER and SciNLI. `heads=False` gives raw
probabilities.

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
