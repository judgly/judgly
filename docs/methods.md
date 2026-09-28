# Methods

This page describes what judgly does and how it is evaluated, in the style of a methods
section. None of the parts is new. The contribution, if there is one, is putting them together
in a small, deterministic engine and reporting the result, including the parts that did not
work. The results are in [Results](#results), from the committed snapshot in
[results/](results/).

## Contents

- [Question and approach](#question-and-approach)
- [The engine](#the-engine)
- [The heads](#the-heads)
- [Engine self-tests](#engine-self-tests)
- [Data and tiers](#data-and-tiers)
- [Leakage and contamination checks](#leakage-and-contamination-checks)
- [Evaluation](#evaluation)
- [Bars](#bars)
- [Results](#results)
- [Negative and null results](#negative-and-null-results)
- [What the evaluation can and cannot show](#what-the-evaluation-can-and-cannot-show)
- [Related work](#related-work)
- [References](#references)

## Question and approach

Can a frozen open-weight language model, read at a single position and with no text
generated, give calibrated probabilities for typed decisions on kinds of task that the
calibration step has never seen?

The approach follows the design of Jev, TypeSafe's commercial decision model, as described in
its launch post: typed questions with a fixed set of answers, each question isolated from the
others and branched from one reading of the text, and probabilities trained to be calibrated
rather than decisive. judgly reimplements that interface on an open model. The calibration step
is a small head on top of the frozen model, not a trained model.

## The engine

**Prompt.** A request has a *state* (the text) and a set of questions. A model-specific
template wraps the state as a user turn after a short system instruction ("You are a decision
function ..."). Each question follows as its own suffix: the instruction, the options as
lettered lines (A, B, C, ...), and the start of the model's turn ending in `Answer:`.

**Branching.** The state is processed once as sequence 0 of llama.cpp's cache. Each question
suffix, and each option order of it, is a separate sequence that copies the state's cache
entries and is processed in one batched call. A branch attends to the state and to itself,
never to another branch. Large question lists are processed in groups.

**Readout.** At the last position of each branch the engine reads the model's final hidden
vector h and computes the logits of the K answer letters from the model's own output rows
(`dot(w[k], h)`, with Gemma's final logit soft-cap applied). A softmax over those K letters
only gives the raw answer distribution. The mass the model puts on the letters within the full
vocabulary is reported as `slot_mass`. Reading answer-letter probabilities of multiple-choice
questions, and asking whether they are calibrated, follows Kadavath et al. (2022).

**Rotation averaging.** A question with K options is asked in several cyclic rotations of the
options. With all K rotations every option appears once in every letter slot, so any fixed
preference for a position or a letter contributes equally to every option and cancels. The
release packs cap this at four evenly spaced rotations (`max_rotations: 4`, see
[Release settings](#release-settings)): for questions with four or fewer options that is all of
them, but with more than four options no option visits every slot, and some position bias can
remain. The distributions are mapped back from slots to option keys and averaged. Yes/no
questions are always asked in both orders. Score questions are never rotated: their levels keep
their natural order (level 1 in slot A, level 2 in slot B, ...), because an ordinal scale shifted
cyclically across the letter slots (A = 3, B = 4, ...) is read against its own order, and a
head's per-slot bias could then not correct the model's preference for some levels (see
[score heads](#the-heads)). The range of the top option's probability across rotations is
reported as `rotation_spread`. Language models are known to prefer some option positions or
letters, and averaging over option orders is a known remedy (Zheng et al. 2024; Pezeshkpour
and Hruschka 2024).

**Content-free pass (off in the release packs).** The engine can also ask each question
against the state "N/A"; the resulting letter logits (zc) measure the model's preference among
the options before it sees any evidence (contextual calibration, Zhao et al. 2021), and a head
can learn how much of them to subtract. The release packs do not use it (`content_free: false`,
see [Release settings](#release-settings)); their heads use z alone.

**Determinism.** For a fixed model file, template, heads, engine settings, build and hardware,
the same request gives the same probabilities. Asking the same question within different
sets of sibling questions moved its probabilities by at most 0.0002 (Gemma 4 12B) and 0.0014
(Qwen3-4B) in the release self-tests (T5,
[selftest.txt](results/gemma4-12b-q8/selftest.txt),
[selftest.txt](results/qwen3-4b-q8/selftest.txt)). Batch shape can change floating-point summation
order, which is why the self-tests compare batched against solo results.

The engine is C code on top of llama.cpp (pinned commit) and is exposed through a four-function
JSON C API (`csrc/judgly.h`) and a Python package that calls it through ctypes.

## The heads

Per rotation, the letter logits z (and zc) go through a head, then a softmax over the K
letters, then the slot-to-key mapping; the rotations are then averaged. The head is fitted per
question type (choice, yes/no, score) and per format.

- **H0**: no fitted parameters. u[k] = z[k] (optionally minus zc[k]).
- **H1**: u[k] = a z[k] - c zc[k] + b[k], with a = exp(theta) > 0 a learned inverse
  temperature, c the learned share of the content-free logits to subtract, and b one bias per
  letter slot. 28 parameters per type.
- **H2**: u[k] = a dot(w[k] + d[k], h) - c zc[k] + b[k]. The model's letter rows w[k] stay
  fixed. The corrections d[k] (26 x hidden size per type) start at zero, so an untrained H2 is
  H1, and are penalised with lambda * sum(d^2).

The loss is the mean cross-entropy (log loss) of the correct option. H1 is fitted by full-batch
L-BFGS. H2 is fitted from the H1 solution for each lambda in a short grid, with early stopping
on validation log loss; the lambda with the lowest validation log loss is kept. The test split
is not used for any choice.

**Score heads.** A level means something different in every score question (level 1 is "not
toxic" in one, "one star" in another), so a score head must not learn how often each level was
the answer in the fit data: its per-level biases would carry that frequency into every score
question. Score items are therefore weighted in the fit, in training and in validation, so that
every level carries the same total weight (w = n / (L n_level)). The score fit data are few
(1,051 training items: Civil Comments toxicity scores, most of them at level 1, HelpSteer2
helpfulness ratings and generated tasks), and the weights raise the rare
levels to the weight of the common ones, which the hidden-size-wide corrections of H2 overfit;
score heads are therefore H1 (temperature and per-level bias) in every H2 head file. Choice and
yes/no items need no weights, because their options are rotated through the letter slots. The analytic gradients are checked against numerical ones (self-test
T7, relative error below 1e-4; ported to the Python test suite as `tests/test_head_gradient.py`).

Features for fitting are extracted by the same engine code, with the same template, rotations
and content-free setting that the served engine uses, and a round-trip test (T8) checks that
probabilities from cached features equal those from the live engine. The trainer checks the
feature file against the declared engine settings (every item read in as many orders as the
engine reads it) and records the settings in the head's sidecar (`h2.bin.json`); the engine
refuses a head under other settings.

**Guardrails of the fit.** The temperature 1/a is held within [0.05, 100]: a fit that drives it
beyond a bound is refitted with it fixed there. A very large temperature means the head has all
but discarded the letter logits (a near 0), and because H2's corrections enter as a dot(d[k], h),
their gradients vanish with a as well; H2 therefore starts from the H1 solution with its
temperature capped at 20. After the fit, a question type falls back to the identity head (H0,
the raw readout) when its head is input-independent on the validation records (the largest
standard deviation of any option's probability across them below 0.001) or when its validation
log loss is not below the raw readout's; the sidecar and the calibration record state the
fallback and its reason. `scripts/check_heads.py` fails the pack build for a head that breaks
any of these without a recorded fallback.

## Engine self-tests

`s1-selftest` runs before every pipeline run (the *gate*). A pack is not built if a check fails,
unless the failure is explicitly accepted and recorded in the run's `selftest.txt`.

| test | checks | tolerance |
|---|---|---|
| T1 prompt integrity | detokenised prompt equals the intended text; a state containing special markers cannot forge a turn | exact |
| T2 rows against logits | the letter readout from the output rows equals llama.cpp's own logits, soft-cap included | 0.05 logit |
| T3 branch against recompute | branching from the cached state equals processing prefix and suffix from scratch | 0.02 probability |
| T4 packed against single | many examples per batch equals one per batch | 0.02 |
| T5 isolation | a question's answer does not change with its sibling questions | 0.01 |
| T6 independent reference | against a 32-bit CPU reference (or CPU against Metal for large models) | 0.05 |
| T7 gradient check | analytic against numerical head gradients | 1e-4 relative |
| T8 head round trip | cached features plus head equal the live engine | 0.02 |
| T9 rotation bookkeeping | slot-to-key mapping under rotation; probabilities sum to 1 | exact; 1e-5 |
| T10 sliding-window cache | branch results with and without the full sliding-window cache (Gemma) | 0.02 |

In the release self-tests, Gemma 4 12B at 8 bits passed with T3 0.0084 and T4 0.0106, and
Qwen3-4B with T3 0.0086 and T4 0.0065 ([selftest.txt](results/gemma4-12b-q8/selftest.txt),
[selftest.txt](results/qwen3-4b-q8/selftest.txt)). The pipeline runs the gate without the T6
reference file.

## Data and tiers

All data sources, their pinned revisions, licences and citations are listed in
[reproduce.md](reproduce.md#data-sources), generated from `data/registry.yaml`; the licence
evidence for each, and the candidates that are not used and why, are in
[licences.md](licences.md).

**Licence policy.** A head is fitted only on sources whose dataset card, at the pinned
revision, names a permissive licence (CC0, CC-BY, MIT, Apache-2.0, BSD), and on judgly's own
generated tasks. The stance head may also use share-alike sources, and is then released under
CC-BY-SA-4.0. Everything else (non-commercial, custom, "other", unknown, or no licence on the
card) is used for evaluation only. Where a card is wrong or incomplete, the registry records
how the source is treated and quotes the primary evidence: FEVER's card lists GPL-3.0 (the
licence of the code repository that published the evidence text) beside CC-BY-SA-3.0, and the
FEVER licence page gives CC-BY-SA-3.0 for the data (reading GPL-3.0 as the code's licence is an
inference, so only FEVER's SUPPORTS and REFUTES claims with its own gold evidence are fitted,
not the NOT ENOUGH INFO claims, whose evidence the release adds by retrieval); SciNLI's card
says Apache-2.0, and its authors release it under CC-BY-SA-4.0, so it is treated as share-alike.
Model-written data follow one rule: model-written text with human labels may be fitted (WANLI);
model-written labels never are (UltraFeedback and Prometheus are not used; typed-decisions and
JevBench are evaluation only). `scripts/verify_licences.py`
re-reads every card at its pinned revision and every quoted primary source, and checks the
registry against them. No dataset text is committed: the tiers are rebuilt from the pinned
sources and only their SHA-256 are committed.

**Families and tiers.** The unit of hold-out is the *family*, a kind of decision (for example
sentiment or topic classification), not the dataset, because two datasets of the same kind are
not independent tests. Each family belongs to exactly one tier, per format:

| tier | general format | stance format | used for |
|---|---|---|---|
| fit | knowledge, reading, moderation, preference, emotion, intent, generated, helpfulness (HelpSteer2), contracts (LEDGAR) | NLI (MNLI, SNLI, WANLI), claims (VitaminC, FEVER), scientific (SciNLI) | fitting (70/15/15 train, validation, test) |
| dev | sentiment, inference, similarity, paraphrase, word sense, commonsense | ANLI, Climate-FEVER, COVID-Fact | a held-out check during development (for stance, a dev ECE above 0.08 marks a head as doubtful) |
| final | social bias (BBQ), grammar (BLiMP), figurative language (Fig-QA), indirect answers (Circa), ethics (ETHICS deontology and justice), tables (TabFact), search relevance (ESCI), sentence difficulty (CEFR-SP) | COVID news claims (Check-COVID) | the reported numbers: fresh families, never used before this tier was frozen, read once after the heads are fitted |
| final-flagged | politeness (Stanford Politeness) | health questions (HealthFC) | fresh families read with final, but each with a recorded caveat: reported beside the final numbers, never pooled into them, judging no bar |
| final-seen | topic, biomedical, legal, maths, finance, truthfulness, MMLU-Pro, ratings, yes/no reading | HealthVer | a secondary evaluation, reported separately: an earlier held-out tier whose families were read during development |
| bench | typed-decisions, JevBench public items | none | external benchmarks, evaluation only, also scored as each defines |
| reserved | none | SciFact, ClimateCheck | nothing; kept unseen for later work (ClimateCheck must first drop the items that overlap Climate-FEVER, which dev holds) |

Tier sizes: general 9,000 fit (6,300 train, 1,350 validation, 1,350 test), 4,500 dev (250 per
task), 7,879 final (1,000 per task, 1,005 for BLiMP, 500 each for the two ETHICS subsets, 874
for CEFR-SP; 6,803 resampling groups), 1,000 final-flagged (Stanford Politeness), 10,300
final-seen and 2,231 bench items (all 2,000 typed-decisions test decisions and all 231
JevBench public items); stance 21,000 fit (4,500 each from MNLI and VitaminC, 4,000 from FEVER,
SUPPORTS and REFUTES only, 3,000 each from SNLI and WANLI, 2,000 from SciNLI), 2,100 dev, 1,343
final (Check-COVID, on 315 abstracts), 749 final-flagged (HealthFC) and 2,100 final-seen items. The fit tier is split by a hash of the
passage, or of a recorded group (the evidence page for FEVER and VitaminC, the prompt for
HelpSteer2), so items that share one stay in one split. Some sources are sampled evenly: by
label (the stance fit sources other than MNLI and VitaminC, HelpSteer2, Circa, ESCI, CEFR-SP,
Stanford Politeness), by paradigm (BLiMP, 15 pairs each) or by category and context (BBQ, one
item per template); ESCI keeps one product per query.

**The fresh final tier.** It was chosen before any head was scored on it, from datasets that no
experiment in this project had used, and it is frozen by its SHA-256 in `data/tiers.sha256`.
Every item carries a resampling group. Its families are unlike every fit family (no
MMLU-style knowledge, Wikipedia-sentence NLI, contracts, preference or rating, sentiment, topic,
biomedical questions, commonsense, reading or moderation). The QUICK pipeline does not score
the fresh tiers; one QUICK smoke run (Qwen3-4B, a throwaway head fitted on a few hundred items)
did score 173 items drawn from them (80 general, 93 stance) and printed their pooled accuracy
and ECE, and the tiers were not changed because of it. The stance final tier is health
claims, on claims and abstracts that no other tier holds. Check-COVID items show the sentences of the abstract that were
annotated as the claim's evidence, not the whole abstract (a not-enough-info item repeats a
supported or refuted claim with another sentence of the same abstract). "Untouched" means
untouched by this project: most of these datasets are years old and may be in the model's
pretraining data.

**The final-flagged tier.** Fresh families built and read like the final tier, but with a caveat
recorded in the registry (`caveat`), so their numbers are reported in a table of their own and
never pooled into the final numbers or the bars. Stanford Politeness asks for a 1-to-3 civility
rating of online requests, a close relative of the fit family moderation (Civil Comments'
1-to-5 toxicity rating); ETHICS commonsense was left out for the same reason. HealthFC claims
are yes/no questions (740 of 749), used verbatim ("supports" means the evidence answers yes),
with the English evidence sentences (never the explanation field); but those sentences are the
fact-checkers' own summary and often state the verdict ("we could not find any meaningful
scientific evidence", headings such as "Not checked"), so the no-bearing label can often be read
off the wording, the flaw for which PubHealth was not used.

**The final-seen tier.** An earlier held-out tier, built from its sources alone
and scored in every run, in its own table, as a secondary evaluation of families seen during
development. Two of its families are relatives of fit families: legal (CaseHOLD) of contracts
(LEDGAR), and ratings (Yelp) of helpfulness (HelpSteer2).

**The bench tier.** Two external benchmarks for typed decisions, evaluation only
([related work](#related-work)). typed-decisions: the 400 test cases (2,000 decisions) of
`LocalLLaMA/typed-decisions`, with each question's per-option descriptions kept (as options for
choice questions, written into the instructions for yes/no and score questions, which have no
description field in judgly), score levels shifted from 0-based to 1-based, and the teacher's
soft gold distribution kept with each item; its train split is never used. JevBench: the public
items of v1.4.2 (original 72, easy 48, hard 111), mapped the same way; the longest have states of
about 15,000 characters, and `s1-features` splits its blocks of examples so that each fits one
context of 32,768 tokens. Each benchmark
is scored as well against its own gold (see [Evaluation](#evaluation)).

**Generated tasks.** A part of the fit tier is generated by `scripts/prep_tiers.py` from a
fixed seed: arithmetic, dates, units, counting, ordering, parity and similar. These are free of
contamination by construction, and they give the fit tier bool and score items with a known
answer.

**Determinism of the data.** The tier builder is seeded, and a second build is byte-identical.
The SHA-256 of every tier file is committed (`data/tiers.sha256`) and checked by
`make verify-data`. The manifest records every item dropped and why.

## Leakage and contamination checks

The head must never see an item, or a near copy of it, that it is later scored on. Tiers are
built in the order reserved, final-seen, dev, final, final-flagged, bench, fit, and an item is dropped from a
later tier when it matches an earlier one:

- **exactly**: its normalised claim, passage, state or question equals one in an earlier tier.
  A passage embedded in a question (MMLU's "This question refers to the following
  information", then a passage, then the question) is compared on its own too, so that two
  items asking different questions about the same passage match;
- **by a shared sentence** (final, final-flagged and fit tiers): a passage shares a sentence of
  at least 60 normalised characters with an earlier tier, so that the same abstract under
  another title, or a quoted passage, is caught. A sentence that begins with a short
  "Title: " prefix (FEVER puts the page title before each page's evidence) is compared without
  it too, and Penn Treebank tokens of tokenised Wikipedia text (-LRB- and the like) are removed
  in normalisation;
- **by a shared passage** (final, final-flagged and fit tiers): a text shares at least five
  distinct word 8-grams with one text of the earlier tiers, not counting 8-grams found in more
  than three of their distinct texts (boilerplate such as "severe acute respiratory syndrome
  coronavirus 2"), so that a reworded or re-segmented copy of a passage is caught (a preprint
  and its published abstract, a Wikipedia paragraph and FEVER's sentences of it);
- **as a near duplicate** (final, final-flagged and fit tiers): a text's set of normalised words has a Jaccard
  similarity of at least 0.6 with a text of an earlier tier (both of at least six distinct
  words; candidates from MinHash banding, then the exact Jaccard).

final-seen and dev are built with the exact check only; the bench tier drops nothing for overlap (a benchmark is scored as published). Check-COVID items are checked with their whole
abstract, not only the sentences shown, so that every claim on an abstract that HealthVer or
COVID-Fact also uses is left out. FEVER claims whose evidence page is also an evidence article of
Climate-FEVER (dev) are left out (7,606 of the SUPPORTS and REFUTES claims). Exact duplicates
within a tier are dropped. In the current build this excluded, for example, 440 MMLU fit items
whose question or passage also appears in MMLU-Pro (final-seen) and 168 more that share a
sentence or a passage with it (or with MedQA) or nearly duplicate it; 158 Check-COVID claims (72
on an abstract also in HealthVer or COVID-Fact, 26 sharing a sentence and 59 a passage with
them, the latter including COVID boilerplate beyond the three-text cut, 1 near duplicate); 4
FEVER claims whose Wikipedia evidence ANLI (dev) quotes; one HealthFC question (vitamin C and
COVID-19, a near duplicate of a HealthVer claim); and 4 COVID-Fact items overlapping HealthVer or
SciFact.

`scripts/check_contamination.py` then recomputes everything from the tier files alone and fails
if it finds a reserved item in any tier; a final-seen item in fit or dev; a fit item in dev; a
fresh final or final-flagged item that equals, shares a sentence or a passage with or nearly
duplicates a fit, dev, final-seen or reserved text (final-flagged also a final text); a fit item
that shares a sentence or a passage with or nearly duplicates a dev, final-seen or reserved
text; a bench item that matches the fit tier in any of these ways; a family in two tiers; a
source used in a tier its licence does not allow; or duplicate ids. It also reports, without
failing, bench items matching other evaluation tiers, the fresh tiers of each format against
the fit tier of the other format, and dev and final-seen items that overlap a reserved source
(in the current build 8 dev items share a sentence, 52 texts a passage and 17 are near
duplicates, nearly all Climate-FEVER against ClimateCheck; Climate-FEVER is a dev-tier source,
so ClimateCheck must drop them before it becomes a final tier). The test suite plants each kind of
leak and checks that the checker catches it. The checker runs at the start of every pipeline
run, and its output is saved with the results.

Known limits of the check: texts shorter than 24 normalised characters are not compared
exactly, and texts of fewer than six distinct words not as near duplicates; a near duplicate is
a lexical match, so a paraphrase with other words is not found; formats are checked separately
for failures; and within the fit tier, texts shared between splits are reported, not failed
(in the current build: general 19 between train and test and 18 between train and validation,
stance 13 and 14).

## Evaluation

Each pack is evaluated per format on the fit tier's test split (in-distribution), dev, final
(fresh held-out families; the reported result), final-flagged (fresh families with a caveat,
reported beside final), final-seen (an earlier held-out tier, reported separately as seen
during development) and, for the general format, bench. The QUICK pipeline does not score final or final-flagged. Two conditions are scored on
each: raw (H0 with the served engine settings) and H2.

- **Metrics**: accuracy, log loss (primary), Brier score, ECE over 10 equal-width bins of the
  top probability, the reliability table, selective accuracy and share answered at thresholds
  0.5 to 0.99, per-family metrics, and for stance the confusion matrix.
- **Intervals**: every metric carries a 95% percentile bootstrap interval from 1,000 resamples
  (fixed seed): of items on test, dev and final-seen; of groups on final, final-flagged and
  bench, whose items carry a `group` (the Check-COVID abstract, which holds a claim, its
  negation and its not-enough-info variant; the TabFact table; the BBQ template; the Circa
  question; the ESCI query; the typed-decisions case), for accuracy, log loss, Brier score and
  ECE overall and per family. The tables give the number of groups next to the items. Percentile intervals for ECE lean upward when the true ECE is near
  0, because ECE cannot be negative. Comparisons between head and raw are paired: both
  conditions are resampled on the same items, and the difference is reported with its interval
  (see [Results](#paired-differences-and-per-family-averages)); an interval that includes zero is
  reported as within noise.
- **Per-item dumps**: every evaluation writes one line per item (id, correct option,
  probability of every option), so that any number can be recomputed and any two runs compared
  item by item. The stance confusion matrices are the exception: the dumps index options by
  their position in each item and do not name them, so the matrices come from `record.json`,
  which the calibration record builds with the tier files.
- **Calibration record**: `scripts/calibration_record.py` recomputes the metrics from the
  per-item dumps, stops if they disagree with the evaluator's, and writes `record.json` and
  `tables.md` with the sources and licences behind each head. final-seen has its own table,
  headed as a secondary evaluation of previously seen families.
- **Bench scoring**: besides the metrics above (against the argmax of each item's gold), each
  benchmark is scored against its own gold as it defines: accuracy (for JevBench score
  questions, the rounded expected level), the multi-class Brier score (the sum over options of
  the squared difference from the gold distribution: typed-decisions' soft teacher
  distribution, JevBench's expected label or its published gold probabilities), the KL
  divergence from the gold to the prediction, ECE (top label, 10 bins) and, for score
  questions, the mean absolute difference between predicted and gold expected level, with 95%
  intervals from resampling cases. typed-decisions' gold is a teacher model's output, so its
  numbers measure agreement with that teacher (its card gives a majority baseline of 0.520, a
  ceiling from its latent factors of 0.704 and the teacher's self-agreement of 0.735). The
  JevBench number covers the 231 public items only and is not comparable with the JevBench
  leaderboard, which also scores sealed items.

The limits of the final tier are in
[What the evaluation can and cannot show](#what-the-evaluation-can-and-cannot-show).

## Bars

The release packs are judged against these bars:

- **Engine**: every self-test within tolerance (above). A model that fails T3 or T4 is not
  shipped as a pack.
- **Contamination**: the checker exits 0 on the tiers used.
- **General head**: H2 must improve held-out (dev and final) log loss over raw. If it improved
  the in-distribution test split but not the held-out families, it would be learning
  task-specific features and would not be shipped as a general head.
- **Stance head**: final-tier ECE below 0.05, and dev-tier ECE not above 0.08.

The bars are judged on the fresh final tier; final-flagged, final-seen and bench numbers are
reported beside them and judge nothing.

<!-- RESULTS:BARS -->
Which bars the packs met is in [Results](#bars-met-and-missed): the engine, contamination and
general-head bars were met; the stance final-tier bar was met narrowly by Gemma 4 12B and missed
narrowly by Qwen3-4B, and **the stance dev bar was missed by both packs**.

## Results

All numbers in this section are from [results/](results/): the calibration records
(`record.json`), their tables (`tables.md`), the per-item dumps and the self-test output of the
runs, one run per pack on an Apple M3 Max. Brackets are 95% percentile bootstrap intervals (of
groups on final, final-flagged and bench, of items elsewhere; see [Evaluation](#evaluation)); n is
the number of questions. The model cards give the full tables:
[gemma4-12b-q8](model-cards/gemma4-12b-q8.md), [qwen3-4b-q8](model-cards/qwen3-4b-q8.md).

### Release settings

The packs, and the runs that fitted and scored them, use these settings (the `engine` block of
each `pack.json`; features for fitting are extracted with the same settings):

- **at most four cyclic option orders** per question, evenly spaced, averaged (all orders when
  a question has four options or fewer);
- **no content-free pass** (`content_free: false`); the heads use the letter logits z alone;
- **tier sizes**: general fit 9,000 items (6,300 train, 1,350 validation, 1,350 test), dev 4,500,
  final 7,879, final-flagged 1,000, final-seen 10,300 and bench 2,231 (2,000 typed-decisions and
  231 JevBench items); stance fit 21,000 (14,783 train, 3,072 validation, 3,145 test), dev 2,100,
  final 1,343, final-flagged 749 and final-seen 2,100. Calibration curves flatten after a few
  hundred labelled items, so a fit tier of a few thousand items is enough for heads of this size.

### Tiers and contamination

The tier design is described [above](#data-and-tiers). The general fresh final tier has 7,879
questions from 9 tasks in 8 families (social bias, grammar, figurative language, indirect
answers, ethics, tables, search relevance, sentence difficulty), in 6,803 resampling groups; the
stance fresh final tier has 1,343 Check-COVID pairs on 315 abstracts. The contamination checker
exited 0 on the tiers before the runs (a pipeline run stops otherwise).

### Headline: fresh final tier

| pack | format | n | accuracy raw → head | log loss raw → head | ECE raw → head |
|---|---|---|---|---|---|
| gemma4-12b-q8 | general | 7,879 | 0.760 [0.751, 0.770] → 0.737 [0.726, 0.746] | 1.553 [1.478, 1.636] → 0.631 [0.613, 0.651] | 0.194 [0.186, 0.204] → 0.020 [0.015, 0.031] |
| gemma4-12b-q8 | stance | 1,343 | 0.827 [0.806, 0.848] → 0.812 [0.790, 0.836] | 1.603 [1.370, 1.855] → 0.509 [0.472, 0.550] | 0.142 [0.123, 0.162] → 0.048 [0.034, 0.072] |
| qwen3-4b-q8 | general | 7,879 | 0.656 [0.644, 0.667] → 0.639 [0.628, 0.650] | 3.474 [3.330, 3.639] → 0.797 [0.779, 0.815] | 0.268 [0.259, 0.279] → 0.030 [0.025, 0.040] |
| qwen3-4b-q8 | stance | 1,343 | 0.778 [0.755, 0.801] → 0.773 [0.749, 0.795] | 3.185 [2.812, 3.589] → 0.594 [0.562, 0.628] | 0.187 [0.167, 0.212] → 0.052 [0.034, 0.077] |

The heads lower log loss and ECE a great deal in every case. They also lower accuracy: clearly
for the general heads, a little for the stance heads (below).

### Paired differences and per-family averages

Head minus raw on the same fresh final-tier items, with 95% paired bootstrap intervals (items
resampled within each family, not by group, 1,000 resamples; `docs/tools/final_tier_stats.py`
recomputes them from the snapshot):

| pack | format | accuracy | log loss | ECE |
|---|---|---|---|---|
| gemma4-12b-q8 | general | -0.023 [-0.031, -0.016] | -0.922 [-0.979, -0.865] | -0.174 [-0.182, -0.162] |
| gemma4-12b-q8 | stance | -0.015 [-0.030, -0.001] | -1.094 [-1.287, -0.909] | -0.094 [-0.123, -0.059] |
| qwen3-4b-q8 | general | -0.017 [-0.024, -0.011] | -2.672 [-2.795, -2.558] | -0.239 [-0.247, -0.223] |
| qwen3-4b-q8 | stance | -0.005 [-0.024, 0.013], within noise | -2.590 [-2.962, -2.238] | -0.134 [-0.171, -0.096] |

The pooled ECE of the general format mixes eight families, and errors in opposite directions
can cancel. The per-family average ECE is larger:

| pack | pooled ECE raw → head | per-family average ECE raw → head |
|---|---|---|
| gemma4-12b-q8 | 0.194 → 0.020 | 0.204 [0.197, 0.214] → 0.051 [0.049, 0.063] |
| qwen3-4b-q8 | 0.268 → 0.030 | 0.277 [0.269, 0.288] → 0.077 [0.071, 0.088] |

The stance final tier is one family, so its pooled and per-family ECE are the same.

### Final-flagged: fresh families with a caveat

Reported beside the final tier, never pooled into it, judging no bar ([the caveats](#data-and-tiers)):

| pack | family | n | accuracy raw → head | ECE raw → head |
|---|---|---|---|---|
| gemma4-12b-q8 | politeness (Stanford Politeness) | 1,000 | 0.519 [0.490, 0.549] → 0.512 [0.481, 0.544] | 0.423 [0.392, 0.452] → 0.106 [0.080, 0.138] |
| gemma4-12b-q8 | stance (HealthFC) | 749 | 0.750 [0.722, 0.782] → 0.634 [0.597, 0.668] | 0.201 [0.174, 0.229] → 0.123 [0.093, 0.158] |
| qwen3-4b-q8 | politeness (Stanford Politeness) | 1,000 | 0.492 [0.462, 0.523] → 0.487 [0.456, 0.519] | 0.442 [0.409, 0.472] → 0.046 [0.030, 0.080] |
| qwen3-4b-q8 | stance (HealthFC) | 749 | 0.718 [0.684, 0.752] → 0.541 [0.506, 0.575] | 0.227 [0.196, 0.260] → 0.150 [0.130, 0.192] |

### Final-seen: families seen during development (secondary)

An earlier held-out tier whose families were read while judgly was developed. These
numbers are not a held-out result; they are reported for comparison only, and their intervals
resample items as if independent.

| pack | format | n | accuracy raw → head | ECE raw → head |
|---|---|---|---|---|
| gemma4-12b-q8 | general | 10,300 | 0.714 [0.705, 0.723] → 0.713 [0.703, 0.722] | 0.180 [0.172, 0.189] → 0.026 [0.021, 0.034] |
| gemma4-12b-q8 | stance (HealthVer) | 2,100 | 0.632 [0.610, 0.652] → 0.626 [0.605, 0.647] | 0.320 [0.302, 0.342] → 0.067 [0.051, 0.089] |
| qwen3-4b-q8 | general | 10,300 | 0.649 [0.640, 0.658] → 0.654 [0.645, 0.663] | 0.231 [0.223, 0.240] → 0.019 [0.016, 0.030] |
| qwen3-4b-q8 | stance (HealthVer) | 2,100 | 0.571 [0.548, 0.592] → 0.548 [0.527, 0.570] | 0.354 [0.333, 0.377] → 0.100 [0.084, 0.122] |

### Bench: external benchmarks

Each benchmark scored against its own gold ([Evaluation](#evaluation)); intervals resample cases.

| pack | benchmark | items (cases) | accuracy raw → head | Brier raw → head | ECE raw → head |
|---|---|---|---|---|---|
| gemma4-12b-q8 | typed-decisions | 2,000 (400) | 0.702 [0.680, 0.723] → 0.700 [0.677, 0.721] | 0.359 [0.342, 0.378] → 0.117 [0.109, 0.125] | 0.251 [0.232, 0.274] → 0.028 [0.019, 0.049] |
| gemma4-12b-q8 | JevBench, public items | 231 (195) | 0.835 [0.784, 0.884] → 0.840 [0.789, 0.890] | 0.274 [0.193, 0.363] → 0.207 [0.159, 0.256] | 0.133 [0.093, 0.185] → 0.054 [0.043, 0.106] |
| qwen3-4b-q8 | typed-decisions | 2,000 (400) | 0.576 [0.549, 0.600] → 0.591 [0.567, 0.614] | 0.482 [0.457, 0.507] → 0.200 [0.184, 0.217] | 0.356 [0.334, 0.380] → 0.126 [0.111, 0.147] |
| qwen3-4b-q8 | JevBench, public items | 231 (195) | 0.693 [0.626, 0.751] → 0.671 [0.605, 0.733] | 0.495 [0.393, 0.607] → 0.361 [0.307, 0.419] | 0.251 [0.199, 0.314] → 0.070 [0.049, 0.127] |

On typed-decisions, whose gold is a teacher model's output, Gemma 4 12B with its head is at 0.700
(raw 0.702) and Qwen3-4B at 0.591. The dataset card reports meraGPT Decider 1 at 0.768 and Jev
1.13.0 at 0.727 (Jev measured by the card's authors through TypeSafe's API), Laya's own model card
reports 0.766 for a checkpoint fine-tuned on the benchmark's train split, and open-alternative-jev
reports 0.737 (one option order) and 0.755 (two) with Qwen3.6-27B; judgly is below all of them.
At about 4B, open-alternative-jev's untrained Qwen3.5-4B reports 0.593 / 0.595 accuracy, ECE 0.118 /
0.062 (one / two orders) and Brier 0.164 (score questions left out), against judgly's Qwen3-4B
head at 0.591, ECE 0.126 and Brier 0.238 counted the same way: at that size judgly's head does
no better.
judgly's accuracy and ECE follow the same definitions as the third-party scorer from
Luni/laya-jev-benchmark, which open-alternative-jev uses (by open-alternative-jev's check, it
reproduces Laya's published numbers to within 0.003) (argmax against the card's `label`; top-label confidence in ten equal-width bins over all
2,000 decisions): applied to judgly's per-item dumps, those definitions give the same values. Brier does
not: judgly averages it over all 2,000 decisions, as the card's own rows appear to (its Uniform
row is reproduced exactly only that way), while that scorer leaves the 800 score questions out;
counted that way, the Gemma 4 12B head's Brier is 0.113 (raw 0.302) and the Qwen3-4B head's 0.238.
The card gives Jev ECE 0.144 without publishing its scorer, and notes that on this benchmark ECE
favours a baseline that ignores the input. The JevBench numbers cover the public items only and
are not comparable with the leaderboard ([related work](#related-work)).

### Bars: met and missed

| bar | gemma4-12b-q8 | qwen3-4b-q8 |
|---|---|---|
| engine: self-tests within tolerance (T6 not run) | met | met |
| contamination checker exits 0 | met | met |
| general head: lower held-out log loss than raw on dev and final | met (dev 1.687 → 0.654, final 1.553 → 0.631) | met (dev 3.287 → 0.636, final 3.474 → 0.797) |
| stance head: final ECE below 0.05 | met, narrowly (0.048 [0.034, 0.072]) | **missed**, narrowly (0.052 [0.034, 0.077]) |
| stance head: dev ECE not above 0.08 | **missed** (0.122 [0.106, 0.143]) | **missed** (0.209 [0.189, 0.232]) |

The Gemma 4 12B stance head meets the final bar on its point estimate only: the interval reaches
well above 0.05. Both stance heads ship, because they are far better calibrated than raw (ECE
0.142 → 0.048 and 0.187 → 0.052), and they are marked as having missed the dev bar in the README
and the model cards.

## Negative and null results

These are the results that did not meet their bar or went against the head.

- **The general heads cost accuracy on the fresh final tier** (paired -0.023 [-0.031, -0.016]
  and -0.017 [-0.024, -0.011], [above](#paired-differences-and-per-family-averages)). The largest
  drops were on grammar (BLiMP: Gemma 4 12B 0.848 → 0.767, Qwen3-4B 0.717 → 0.635) and ethics
  (Gemma 4 12B 0.794 → 0.733). A head of this kind is meant to change how far to trust an answer,
  not which answer is on top; on these families it changed some top answers, more often for the
  worse. On the general final-seen and bench tiers accuracy moved by less than 0.025 either way.
- **Stance.** Both stance heads miss the dev bar, worst on scientific abstracts (dev_scientific
  ECE 0.228 [0.197, 0.264] and 0.273 [0.241, 0.311]), and only Gemma 4 12B meets the final bar,
  narrowly ([above](#bars-met-and-missed)). On the final tier the most common error with the head
  is calling no-bearing evidence "contradicts" (95 and 116 of 442 no-bearing pairs). On HealthFC
  (final-flagged) the stance heads lowered accuracy from 0.750 to 0.634 and from 0.718 to 0.541,
  mostly the same error (217 and 275 of 422 no-bearing pairs called "contradicts"); on HealthVer
  (final-seen) head ECE was 0.067 [0.051, 0.089] and 0.100 [0.084, 0.122]. The general head was
  not scored on the stance tiers, so whether it would do better on stance is not known.
- **Calibration varies by family.** The per-family average ECE on the fresh final tier was 0.051
  and 0.077 with the heads, against 0.020 and 0.030 pooled. The worst families were social bias
  (0.073 [0.057, 0.088]) and ethics (0.067 [0.057, 0.098]) for Gemma 4 12B, and ethics (0.114
  [0.089, 0.142]) and figurative language (0.098 [0.077, 0.124]) for Qwen3-4B.
- **Score questions.** Sentence difficulty (CEFR-SP, six levels, n = 874) reached 0.390
  [0.357, 0.422] (Gemma 4 12B, raw 0.383) and 0.253 [0.223, 0.283] (Qwen3-4B, raw 0.287) with the
  head; the head fixes its calibration (ECE 0.535 → 0.040 and 0.626 → 0.084), not its accuracy.
  Sentence similarity (STS-B, dev tier, n = 250) reached 0.536 [0.476, 0.592] and 0.444 [0.384,
  0.504], with ECE 0.094 [0.061, 0.159] and 0.135 [0.083, 0.195]. Politeness (final-flagged,
  three levels) stayed near 0.5 accuracy.
- **Word sense.** The heads lowered accuracy on WiC (dev tier, n = 250): Gemma 4 12B 0.700
  [0.640, 0.756] → 0.580 [0.516, 0.640], Qwen3-4B 0.628 [0.568, 0.684] → 0.564 [0.504, 0.628].
  Head ECE on WiC stayed high: 0.325 [0.268, 0.383] and 0.259 [0.200, 0.316].
- **Legal.** CaseHOLD (final-seen, five options, 0.2 by chance, n = 700) stayed near chance:
  0.254 [0.221, 0.289] (Gemma 4 12B) and 0.300 [0.267, 0.337] (Qwen3-4B) with the head; the
  Gemma 4 12B head left ECE at 0.151 [0.118, 0.187].
- **Financial tweets.** On the finance family (final-seen, n = 700) the Qwen3-4B head left ECE
  at 0.253 [0.223, 0.294], the worst per-family head ECE in any tier.
- **Dev calibration is worse than final.** With the head, general dev-tier ECE was 0.080
  [0.070, 0.092] (Gemma 4 12B) and 0.055 [0.049, 0.070] (Qwen3-4B), against 0.020 and 0.030 on
  the fresh final tier. The final-tier figure is the one to quote, but it depends on which
  families are in the tier.
- **typed-decisions.** Gemma 4 12B reached 0.700 [0.677, 0.721] accuracy with the head (raw
  0.702), below Jev's 0.727 as reported on the dataset card; the head did not raise accuracy.
  Qwen3-4B reached 0.591 [0.567, 0.614], with head ECE 0.126 [0.111, 0.147] on 2,000
  decisions, far from the final-tier figure.

## What the evaluation can and cannot show

- **The model is not held out.** The model was trained by others on data that may include
  these public benchmarks. Holding out families from the head says whether the *calibration*
  generalises; it cannot say whether the model's *accuracy* is inflated by contamination. The
  generated tasks are the only part free of this by construction.
- **Calibration transfers imperfectly.** The final tier is a set of families no head saw, but
  your task is another one. The reported ECE is evidence that the head generalises across these
  families, not a guarantee for yours.
- **Only the fresh final and final-flagged tiers are held out from development.** The final-seen tier is made of public benchmarks that were also used to choose the model and its
  settings, so it is not an independent test set and its figures may be somewhat optimistic;
  it is reported separately for that reason. The fresh final tier was frozen before any head
  was scored on it and is read once per release run; a tier that has been read is spent, and
  the next untouched one (ClimateCheck, reserved) is kept for later.
- **Clustered items.** The final-seen stance tier's 2,100 HealthVer pairs share 289
  abstracts (tier file `data/tiers/stance/final-seen.jsonl`, SHA-256 `315d67a0...`, listed in
  [INPUTS.sha256](results/INPUTS.sha256); built by `make data`, not committed). Check-COVID claims come in variants of one news item that share an abstract, and
  typed-decisions asks five questions of each case. On final, final-flagged and bench the
  intervals resample groups; on final-seen (whose items carry no group) they resample items as if independent, so they may be too narrow. The paired
  differences of `docs/tools/final_tier_stats.py` still resample items within families.
- **Evaluation-only licences.** Several evaluation sets are non-commercial or have unconfirmed
  licences (HealthVer, COVID-Fact, SciFact, ANLI, HealthFC, CEFR-SP and others). They are used only to measure,
  never to fit.
- **Hardware.** All numbers come from an Apple M3 Max with Metal. Other hardware may give
  slightly different probabilities. The self-test T6 (a CPU reference against Metal, tolerance
  0.05) is not run by judgly's pipeline, so the size of that difference is not checked here.

## Related work

judgly is one of several attempts to get typed, calibrated decisions from a language model
without generating text. **Jev** (TypeSafe, commercial, closed) is the model whose public
description gave the idea: typed questions, probabilities for every allowed answer, questions
isolated and branched from one reading of the state. The closest open design is
**open-alternative-jev** ([ikermoel/open-alternative-jev](https://github.com/ikermoel/open-alternative-jev),
Apache-2.0): it also keeps the model frozen, reads the option-letter logits at fixed positions
and normalises them over the options only, with no training beyond an optional single-scalar
temperature. It reports 0.737 accuracy on typed-decisions with Qwen3.6-27B (8-bit), with ECE
0.020 and Brier 0.113 (score questions left out of Brier), and 0.755 with ECE 0.0075 when two
option orders are averaged; these figures are the author's own, on a model more than twice the
size of Gemma 4 12B, and were not reproduced here. **Cygnet**
([blockbrain-ai/cygnet-recipe](https://github.com/blockbrain-ai/cygnet-recipe), MIT) is closer
still: the same frozen Gemma-4-12B-it (bf16, served with vLLM), one answer letter read from the
top log-probabilities and renormalised over the options, and a single temperature (T = 3.4)
fitted on 241 items its authors generated, never on JevBench. It reports 203 of the 231 public
JevBench items (0.879), scored with JevBench's own tool; one near-tie item can make it 204.
judgly's Gemma head scores 0.840 on the same items with judgly's scorer, which rounds the expected
level on score questions and has not been checked against JevBench's tool. Cygnet reports 0.05
to 0.07 s per decision (median, serial, standard tier) on 48 GB GPUs; the board measures it at
0.23 s. It is joint leader of the live JevBench board with Winnow-12B Q8, in a statistical tie
(v1.5.0 as of 29 September 2026: Cygnet 73.7, Winnow-12B Q8 73.2, Jev 1.13.0 72.1; the board calls
75 of its 88 neighbouring pairs statistical ties). judgly differs from
both mainly in fitting small per-type heads (slot biases and a state-dependent correction) on
licence-checked data and in evaluating them on many fresh task families. It is not the only
project that measures calibration on data its fit never saw: **Kev**
([jaredpalmer/kev](https://github.com/jaredpalmer/kev), Apache-2.0; a LoRA adapter and pointer
head on a fixed Qwen base) reports ECE on frozen, checksummed new sources (for Kev-9B, 0.106 raw and 0.042 with a
temperature fitted on in-distribution development data), and **decider-4b**
([Mapika/decider-4b](https://huggingface.co/Mapika/decider-4b), Apache-2.0)
reports ECE on 28 held-out tasks and on held-out generated families under pre-registered rules,
including a missed bar.
**Laya** (convaiinnovations,
Apache-2.0) is an open ModernBERT-large encoder (Warner et al. 2024) with a decision head, built
for a similar purpose. Laya has not been run on judgly's tiers. Its model card reports 0.766
accuracy and ECE 0.213 on typed-decisions for a checkpoint fine-tuned on that benchmark's train
split (the base checkpoint: 0.362), so that figure is not comparable with zero-shot systems.
**CLM-v0.1-8B** ([Contrastive-LM/CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B),
Apache-2.0) keeps Qwen3-8B frozen, as judgly keeps its model frozen, but scores options
differently: it embeds the state and each candidate separately and compares them with two
small projection heads trained contrastively (bidirectional InfoNCE) on about 60 million
question-answer pairs, 30 million synthetic hard negatives and 1 million agent trajectories.
Scoring candidates one at a time lets it rank around a thousand of them, where judgly shows
every option in the prompt, reads the answer letters (at most 26) and averages over option
orders. Its model card reports parity with Jev on computer-use, gaming and tool-calling tasks,
and results as a fine-tuned verifier on DeepSWE and Terminal-Bench; it runs on vLLM. These
figures are the authors' own and were not reproduced here, and CLM has not been evaluated on
judgly's tiers, so no comparison of accuracy or calibration between the two can be made from
this repository. The two differ in aim as well as design: CLM targets choosing among many agent
actions and was trained at scale; judgly fits only a small calibration head on a few thousand
labelled items and concentrates on measuring calibration on task families the head never saw.

Two public benchmarks target typed decisions directly, and judgly scores both in its bench tier
(evaluation only). **typed-decisions**
([LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions),
Apache-2.0) has 400 synthetic test cases in four workflows with five typed questions each; its
gold is the mean of three samples from an unnamed teacher model of "roughly 4B-class
capability", so a score measures agreement with that teacher, not correctness. Its card reports
Jev 1.13.0 at 0.727 accuracy, Brier 0.148 and ECE 0.144 (measured by the card's authors through
TypeSafe's API on 2026-09-18), meraGPT Decider 1 at 0.768 (Brier 0.052) and Featherless Simple
Jev at 0.716; the card does not say
who measured the meraGPT and Featherless rows. On speed, TypeSafe's documentation says most
Jev queries complete in about 100 ms ([docs.typesafe.ai](https://docs.typesafe.ai/concepts/how-to-build-with-system-one))
and mentions "real-time speeds (150ms)" ([use-case map](https://docs.typesafe.ai/concepts/use-case-map));
the typed-decisions card measured 710 ms per five-decision case end to end, and the JevBench board
0.62 s (adjusted p50). judgly's release runs took 0.34 to 1.37 s per question by tier (0.83 s over all
tiers; 1.15 s on the bench tier) for Gemma 4 12B, batched with up to four option orders per
question on an M3 Max, from the shard logs; that is throughput, not single-request latency. **JevBench** ([fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench),
MIT) is an operator-run leaderboard; its v1.5.0 score is a harmonic mean of intelligence,
calibration, speed and cost over 904 open and 720 sealed decisions. judgly runs the 231 public
items of v1.4.2, so its bench number is not comparable with that leaderboard. A frequently quoted
comparison of Kev-9B (0.852) with Jev (0.857) is on neither benchmark and on different item sets:
0.852 is Kev-9B's test score and Jev was run only on Kev's development items, where Kev-9B scores
0.822.

## References

- Kadavath et al. 2022. Language Models (Mostly) Know What They Know. arXiv:2207.05221.
- Zheng, Zhou, Meng, Zhou and Huang 2024. Large Language Models Are Not Robust Multiple Choice
  Selectors. ICLR. arXiv:2309.03882.
- Pezeshkpour and Hruschka 2024. Large Language Models Sensitivity to The Order of Options in
  Multiple-Choice Questions. Findings of NAACL. arXiv:2308.11483.
- Warner et al. 2024. Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast,
  Memory Efficient, and Long Context Finetuning and Inference (ModernBERT). arXiv:2412.13663.
- Guo, Pleiss, Sun and Weinberger 2017. On Calibration of Modern Neural Networks. ICML.
- Gneiting and Raftery 2007. Strictly Proper Scoring Rules, Prediction, and Estimation. JASA.
- Zhao, Wallace, Feng, Klein and Singh 2021. Calibrate Before Use: Improving Few-Shot
  Performance of Language Models. ICML.
- OpenAI 2023. GPT-4 Technical Report. arXiv:2303.08774.
- Nocedal 1980. Updating Quasi-Newton Matrices with Limited Storage. Mathematics of
  Computation (L-BFGS).
- Wilson 1927. Probable Inference, the Law of Succession, and Statistical Inference. JASA
  (the interval used in `examples/thresholds.py`).
- Efron and Tibshirani 1993. An Introduction to the Bootstrap. Chapman and Hall.
- TypeSafe, launch post for Jev, 15 September 2026; Archer Hume, black-box study of Jev, 17
  September 2026 (cited by date; links to follow). The source of the interface idea and of the isolation, option-order and
  irrelevant-option probes.
- Gerganov and the ggml authors. llama.cpp. https://github.com/ggml-org/llama.cpp
- Google 2026. Gemma 4 model card. https://huggingface.co/google/gemma-4-12B-it
- Qwen Team (Yang et al.) 2025. Qwen3 Technical Report. arXiv:2505.09388.
- YaoYuan. yyjson. https://github.com/ibireme/yyjson
- The dataset papers are cited per source in [reproduce.md](reproduce.md#data-sources).
