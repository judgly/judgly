# Model card: qwen3-4b-q8

This card follows the outline of Mitchell et al. (2019), "Model Cards for Model Reporting". It
describes the judgly pack `qwen3-4b-q8`: the Qwen3-4B model file, the prompt template and the two fitted
calibration heads (general and stance). judgly is a weekend hobby project; please read the numbers
below with that in mind. Every number on this page is copied from the committed snapshot in
[docs/results/qwen3-4b-q8/](../results/qwen3-4b-q8/) (`record.json`, `tables.md`, `selftest.txt`), where
the per-item dumps let anyone recompute them (the stance confusion counts come from
`record.json`: the dumps do not name the options). Brackets are 95% percentile bootstrap
intervals (1,000 resamples): of items on the test, dev and final-seen tiers, and of whole
groups of related items (a shared abstract, table, template, query or case) on the final,
final-flagged and bench tiers. The per-family intervals are separate resamples, so a family
that makes up a whole tier (the stance final tier) has slightly different brackets from the
tier's own row.

**In short.** On the fresh final tier of general questions (n = 7,879 questions from eight task
families that no head saw and that were not used during development; one smoke
run with a throwaway head scored 173 items drawn from the fresh tiers, 80 general and 93 stance), the general head brought ECE from 0.268 [0.259, 0.279]
to 0.030 [0.025, 0.040], and lowered accuracy from 0.656 to 0.639 (paired difference
-0.017 [-0.024, -0.011]). On stance (Check-COVID, n = 1,343 claim and evidence pairs) the
stance head brought ECE from 0.187 [0.167, 0.212] to 0.052 [0.034, 0.077], which **misses the bar of 0.05**, narrowly; accuracy went from
0.778 to 0.773 (paired difference -0.005 [-0.024, 0.013], within noise).
This is the small, fast pack; the default pack, [gemma4-12b-q8](gemma4-12b-q8.md), is more accurate.

## Contents

- [Model details](#model-details)
- [Intended use](#intended-use)
- [Out-of-scope uses](#out-of-scope-uses)
- [Fit data](#fit-data)
- [Evaluation data and tiers](#evaluation-data-and-tiers)
- [Metrics](#metrics)
- [Results: general format](#results-general-format)
- [Results: stance format](#results-stance-format)
- [Selective accuracy](#selective-accuracy)
- [Known weaknesses](#known-weaknesses)
- [Determinism checks](#determinism-checks)
- [Licences](#licences)
- [Citation](#citation)

## Model details

The language model is not part of judgly. It is downloaded from Hugging Face on first use and
checked against its SHA-256.

| field | value |
|---|---|
| base model | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| GGUF repository | [unsloth/Qwen3-4B-Instruct-2507-GGUF](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF) |
| revision | `a06e946bb6b655725eafa393f4a9745d460374c9` |
| file | `Qwen3-4B-Instruct-2507-Q8_0.gguf` (4,280,405,600 bytes) |
| SHA-256 | `391c1e410fd9f4cf2de2b510273b56a84c19ce18f4fa3bfb3774031dac4ef068` |
| licence | Apache-2.0 (Qwen team, Alibaba Cloud); GGUF conversion by Unsloth |
| prompt template SHA-256 | `f509b74bf401323f0d83922fb799a197bf17e6537c7108e29bcf2ea7daa613e9` |
| engine settings | `{"rotations": true, "content_free": false, "max_rotations": 4}` |
| llama.cpp commit | `6f41ac59e0a49a00483a316a22ada6b04edd2950` |
| judgly version of the runs | `v0.1.0` (release tag). The records say `judgly_dirty: true` (written from a working tree with uncommitted changes) and name the version, not a commit hash, so they do not pin the exact code state |
| hardware of the runs | Apple M3 Max, Metal |

The heads are H2 heads ([methods.md](../methods.md#the-heads)): a learned temperature, one bias
per letter slot and a small penalised correction to the model's own letter rows, fitted per
question type; for score questions there is no row correction (H1), and every level is weighted
equally in the fit. A head fits only this exact model file and template.

| head | file | SHA-256 | licence |
|---|---|---|---|
| general (every question without `format="stance"`) | `heads/h2.bin` | `9da59ab5a64c7f3782853bee054a2be43173d519217e20e8b284bfe30a6c2828` | Apache-2.0 |
| stance (`format="stance"`) | `heads/h2-stance.bin` | `f85c4fe3e9e0bf78af143a55b460cb220d1293168d23c9501bdee8f471b99c0c` | CC-BY-SA-4.0 |

## Intended use

- Typed decisions about a short text (pick an option, yes or no, a rating) where a probability
  per answer is more useful than generated text, for example routing, triage, labelling for
  later human review, or filtering with a confidence threshold.
- Research and hobby use on a Mac with Apple Silicon, in English.
- Use with a check on your own labelled cases: the calibration was measured on public benchmark
  families, and yours is another one ([calibration.md](../calibration.md)).

## Out-of-scope uses

- Decisions about people (health, credit, employment, legal, education, policing) without a
  human making the decision. The accuracy below is far from what such uses need.
- Medical, legal or scientific judgements treated as authoritative. The stance head is
  weakest on scientific abstracts (see [Known weaknesses](#known-weaknesses)).
- Languages other than English, and texts longer than half the context (they are truncated).
- Any other model file, quantisation or template with these heads.

## Fit data

The heads were fitted only on sources whose dataset card names a licence that allows it
([methods.md](../methods.md#data-and-tiers), [licences.md](../licences.md)). Items are the
training-split counts recorded in `record.json`. Each fit pool was split 70/15/15 into
training, validation and the in-distribution test split (general: 6,300 training items and
1,350 test items; stance: 14,783 training items and 3,145 test items).

**General head** (Apache-2.0):

| source | licence on the card | training items | citation |
|---|---|---|---|
| banking77 | cc-by-4.0 | 485 | Casanueva et al. 2020, Efficient Intent Detection with Dual Sentence Encoders. NLP4ConvAI workshop. |
| civil_comments | cc0-1.0 | 485 | Borkan et al. 2019, Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification. WWW Companion. |
| civil_comments_score | cc0-1.0 | 485 | Borkan et al. 2019 (as civil_comments). |
| commonsense_qa | mit | 485 | Talmor et al. 2019, CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge. NAACL. |
| cosmos_qa | cc-by-4.0 | 485 | Huang et al. 2019, Cosmos QA: Machine Reading Comprehension with Contextual Commonsense Reasoning. EMNLP. |
| generated | apache-2.0 | 485 | judgly (this repository). |
| go_emotions | apache-2.0 | 485 | Demszky et al. 2020, GoEmotions: A Dataset of Fine-Grained Emotions. ACL. |
| helpsteer2 | cc-by-4.0 | 485 | Wang et al. 2024, HelpSteer2: Open-source dataset for training top-performing reward models. arXiv:2406.08673. |
| hh_rlhf | mit | 484 | Bai et al. 2022, Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. arXiv:2204.05862. |
| ledgar | cc-by-4.0 | 484 | Tuggener et al. 2020, LEDGAR: A Large-Scale Multi-label Corpus for Text Classification of Legal Provisions in Contracts. LREC; Chalkidis et al. 2022, LexGLUE. ACL. |
| mmlu | mit | 484 | Hendrycks et al. 2021, Measuring Massive Multitask Language Understanding. ICLR. |
| qasc | cc-by-4.0 | 484 | Khot et al. 2020, QASC: A Dataset for Question Answering via Sentence Composition. AAAI. |
| strategyqa | mit | 484 | Geva et al. 2021, Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies. TACL. |

**Stance head** (CC-BY-SA-4.0, because MNLI, VitaminC, FEVER, SNLI and SciNLI are share-alike):

| source | licence on the card | training items | citation |
|---|---|---|---|
| fever | cc-by-sa-3.0, gpl-3.0 | 2,800 | Thorne et al. 2018, FEVER: a Large-scale Dataset for Fact Extraction and VERification. NAACL; evidence release: Atanasova, Wright and Augenstein 2020, Generating Label Cohesive and Well-Formed Adversarial Claims. EMNLP. |
| mnli | cc-by-3.0, cc-by-sa-3.0, mit, other | 3,182 | Williams, Nangia and Bowman 2018, A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference. NAACL. |
| scinli | apache-2.0 | 1,417 | Sadat and Caragea 2022, SciNLI: A Corpus for Natural Language Inference on Scientific Text. ACL. |
| snli | cc-by-sa-4.0 | 2,099 | Bowman et al. 2015, A large annotated corpus for learning natural language inference. EMNLP. |
| vitaminc | cc-by-sa-3.0 | 3,159 | Schuster, Fisch and Barzilay 2021, Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence. NAACL. |
| wanli | cc-by-4.0 | 2,126 | Liu et al. 2022, WANLI: Worker and AI Collaboration for Natural Language Inference Dataset Creation. Findings of EMNLP. |

FEVER is fitted with its SUPPORTS and REFUTES claims and their gold evidence only; SciNLI with
its entailment and neutral pairs ([methods.md](../methods.md#data-and-tiers)).

## Evaluation data and tiers

Families are held out whole ([methods.md](../methods.md#data-and-tiers)), and the contamination
checker removes any evaluation text from the fit tier.

| tier | what it tests | general sources (items) | stance sources (items) |
|---|---|---|---|
| test | in-distribution: held-out items of the fit families | banking77 (104), civil_comments (104), civil_comments_score (104), commonsense_qa (104), cosmos_qa (104), generated (104), go_emotions (104), helpsteer2 (104), hh_rlhf (104), ledgar (104), mmlu (104), qasc (103), strategyqa (103) | fever (600), mnli (661), scinli (314), snli (458), vitaminc (661), wanli (451) |
| dev | held-out families, a check scored during development | anli_general (250), cb (250), copa (250), hellaswag (250), imdb (250), mrpc (250), piqa (250), qqp (250), rotten_tomatoes (250), rte (250), siqa (250), sst2 (250), stsb (250), swag (250), tweet_irony (250), tweet_sentiment (250), wic (250), winogrande (250) | anli (700), climate_fever (700), covidfact (700) |
| final | fresh held-out families, frozen before any head was scored on them: the reported numbers | bbq (1,000), blimp (1,005), cefr_sp (874), circa (1,000), esci (1,000), ethics_deontology (500), ethics_justice (500), fig_qa (1,000), tabfact (1,000) | check_covid (1,343) |
| final-flagged | fresh families with a recorded caveat, reported beside final, never pooled, judging no bar | politeness (1,000) | healthfc (749) |
| final-seen | secondary: an earlier held-out tier, families seen during development | ag_news (700), aqua_rat (500), boolq (700), casehold (700), dbpedia (700), fin_tweets (700), med_qa (700), medmcqa (700), mmlu_pro (700), newsgroups (700), pubmedqa (700), trec (700), truthful_qa (700), yahoo (700), yelp (700) | healthver (2,100) |
| bench | external benchmarks, also scored against their own gold | jevbench (231), typed_decisions (2,000) | none |

No head saw an item of the dev, final, final-flagged, final-seen or bench tiers. Several
evaluation sources are non-commercial or have unconfirmed licences; they were used only to
measure, never to fit. Their licences and citations are in
[reproduce.md](../reproduce.md#data-sources). The limits of the tiers as held-out sets are in
[methods.md](../methods.md#what-the-evaluation-can-and-cannot-show).

## Metrics

Accuracy of the top answer; log loss (the primary outcome, a proper scoring rule); the Brier
score; ECE over ten equal-width bins of the top probability; selective accuracy (accuracy and
share answered when only answers above a threshold are kept). "raw" is the letter
probabilities averaged over at most four option orders with no head; "h2" is the shipped head.
The bench tier is also scored as each benchmark defines it, against its own gold (Brier score
and KL divergence against the gold distribution, and for score questions the mean absolute
error of the expected level). Definitions are in [calibration.md](../calibration.md#the-numbers-reported)
and [methods.md](../methods.md#evaluation).

## Results: general format

| tier | items | condition | accuracy | log loss | Brier | ECE |
|---|---|---|---|---|---|---|
| test | 1,350 | raw | 0.672 [0.647, 0.697] | 4.085 [3.691, 4.482] | 0.586 [0.541, 0.629] | 0.245 [0.223, 0.269] |
| test | 1,350 | h2 | 0.697 [0.675, 0.723] | 0.776 [0.727, 0.819] | 0.389 [0.365, 0.411] | 0.040 [0.031, 0.064] |
| dev | 4,500 | raw | 0.730 [0.718, 0.743] | 3.287 [3.076, 3.473] | 0.492 [0.468, 0.514] | 0.220 [0.208, 0.234] |
| dev | 4,500 | h2 | 0.731 [0.718, 0.745] | 0.636 [0.611, 0.659] | 0.380 [0.364, 0.395] | 0.055 [0.049, 0.070] |
| final | 7,879 (6,803 groups) | raw | 0.656 [0.644, 0.667] | 3.474 [3.330, 3.639] | 0.602 [0.583, 0.622] | 0.268 [0.259, 0.279] |
| final | 7,879 (6,803 groups) | h2 | 0.639 [0.628, 0.650] | 0.797 [0.779, 0.815] | 0.450 [0.440, 0.460] | 0.030 [0.025, 0.040] |

Paired over the same fresh final-tier items (`docs/tools/final_tier_stats.py`, items resampled
within families), head minus raw was -0.017 [-0.024, -0.011] in accuracy and -2.672 [-2.795, -2.558]
in log loss. Averaged over the eight families instead of pooled, ECE was 0.277 [0.269, 0.288]
raw and 0.077 [0.071, 0.088] with the head.

Fresh final tier by family (the tier to quote):

| family | items | raw accuracy | h2 accuracy | raw ECE | h2 ECE |
|---|---|---|---|---|---|
| difficulty | 874 (874 groups) | 0.287 [0.256, 0.317] | 0.253 [0.223, 0.283] | 0.626 [0.597, 0.659] | 0.084 [0.055, 0.115] |
| ethics | 1,000 (1,000 groups) | 0.682 [0.653, 0.710] | 0.667 [0.637, 0.696] | 0.266 [0.241, 0.297] | 0.114 [0.089, 0.142] |
| figurative | 1,000 (1,000 groups) | 0.802 [0.778, 0.826] | 0.809 [0.783, 0.835] | 0.157 [0.138, 0.181] | 0.098 [0.077, 0.124] |
| grammar | 1,005 (1,005 groups) | 0.717 [0.689, 0.744] | 0.635 [0.606, 0.666] | 0.132 [0.111, 0.159] | 0.066 [0.049, 0.092] |
| pragmatics | 1,000 (863 groups) | 0.685 [0.657, 0.715] | 0.680 [0.652, 0.709] | 0.222 [0.196, 0.248] | 0.072 [0.055, 0.098] |
| relevance | 1,000 (1,000 groups) | 0.442 [0.409, 0.471] | 0.447 [0.416, 0.477] | 0.502 [0.472, 0.535] | 0.064 [0.034, 0.094] |
| social_bias | 1,000 (317 groups) | 0.866 [0.842, 0.890] | 0.873 [0.850, 0.895] | 0.095 [0.076, 0.118] | 0.076 [0.058, 0.097] |
| tables | 1,000 (744 groups) | 0.722 [0.693, 0.749] | 0.699 [0.672, 0.726] | 0.219 [0.193, 0.247] | 0.041 [0.024, 0.066] |

![Per-family accuracy and ECE on the fresh final tier](../assets/results/families-qwen3-4b-q8.svg)

*Figure: fresh final tier by task family (both formats), accuracy and ECE, raw and with the
head, with 95% bootstrap intervals ([captions](../assets/results/CAPTIONS.md)).*

**Final-flagged (reported beside final, never pooled, judging no bar).** Stanford Politeness:
A relative of the fit family moderation (civil_comments_score rates online-discussion comments for toxicity on a 1-to-5 scale; this rates Wikipedia talk-page and StackExchange requests for politeness on a 1-to-3 scale), so it is not an unrelated family.

| tier | items | condition | accuracy | log loss | Brier | ECE |
|---|---|---|---|---|---|---|
| final-flagged | 1,000 (1,000 groups) | raw | 0.492 [0.462, 0.523] | 5.688 [5.214, 6.145] | 0.935 [0.877, 0.991] | 0.442 [0.409, 0.472] |
| final-flagged | 1,000 (1,000 groups) | h2 | 0.487 [0.456, 0.519] | 1.032 [1.002, 1.064] | 0.620 [0.601, 0.640] | 0.046 [0.030, 0.080] |

**Final-seen (secondary: families seen during development).** An earlier held-out tier.
Its families were read while judgly was developed, so these numbers are not a held-out result.

| tier | items | condition | accuracy | log loss | Brier | ECE |
|---|---|---|---|---|---|---|
| final-seen | 10,300 | raw | 0.649 [0.640, 0.658] | 3.559 [3.426, 3.687] | 0.569 [0.554, 0.584] | 0.231 [0.223, 0.240] |
| final-seen | 10,300 | h2 | 0.654 [0.645, 0.663] | 0.932 [0.914, 0.950] | 0.457 [0.448, 0.465] | 0.019 [0.016, 0.030] |

| family | items | raw accuracy | h2 accuracy | raw ECE | h2 ECE |
|---|---|---|---|---|---|
| biomedical | 2,100 | 0.640 [0.621, 0.660] | 0.644 [0.625, 0.664] | 0.247 [0.227, 0.265] | 0.035 [0.025, 0.057] |
| finance | 700 | 0.473 [0.437, 0.506] | 0.481 [0.446, 0.521] | 0.477 [0.444, 0.517] | 0.253 [0.223, 0.294] |
| knowledge_pro | 700 | 0.446 [0.410, 0.486] | 0.471 [0.436, 0.509] | 0.239 [0.206, 0.271] | 0.088 [0.064, 0.125] |
| legal | 700 | 0.280 [0.247, 0.313] | 0.300 [0.267, 0.337] | 0.393 [0.358, 0.429] | 0.088 [0.060, 0.121] |
| math | 500 | 0.442 [0.396, 0.486] | 0.452 [0.410, 0.496] | 0.206 [0.170, 0.249] | 0.085 [0.063, 0.132] |
| rating | 700 | 0.490 [0.454, 0.526] | 0.529 [0.490, 0.564] | 0.485 [0.451, 0.522] | 0.093 [0.066, 0.133] |
| topic | 3,500 | 0.822 [0.809, 0.835] | 0.817 [0.804, 0.829] | 0.126 [0.115, 0.138] | 0.043 [0.036, 0.056] |
| truthfulness | 700 | 0.654 [0.617, 0.690] | 0.634 [0.599, 0.671] | 0.203 [0.174, 0.234] | 0.048 [0.034, 0.080] |
| yesno | 700 | 0.861 [0.837, 0.887] | 0.867 [0.843, 0.891] | 0.091 [0.075, 0.119] | 0.078 [0.058, 0.100] |

**Bench (external benchmarks), scored against their own gold.** typed-decisions' gold is a
teacher model's output, so its numbers measure agreement with that teacher; the JevBench number
covers the 231 public items only and is not comparable with the JevBench leaderboard, which
also scores sealed items ([methods.md](../methods.md#data-and-tiers)).

| benchmark | items (cases) | condition | accuracy | Brier | KL | ECE | score MAE |
|---|---|---|---|---|---|---|---|
| jevbench | 231 (195) | raw | 0.693 [0.626, 0.751] | 0.495 [0.393, 0.607] | 2.704 [1.988, 3.504] | 0.251 [0.199, 0.314] | 0.433 [0.154, 0.881] |
| typed_decisions | 2,000 (400) | raw | 0.576 [0.549, 0.600] | 0.482 [0.457, 0.507] | 5.368 [5.130, 5.604] | 0.356 [0.334, 0.380] | 0.656 [0.619, 0.695] |
| jevbench | 231 (195) | h2 | 0.671 [0.605, 0.733] | 0.361 [0.307, 0.419] | 0.617 [0.533, 0.707] | 0.070 [0.049, 0.127] | 0.637 [0.461, 0.821] |
| typed_decisions | 2,000 (400) | h2 | 0.591 [0.567, 0.614] | 0.200 [0.184, 0.217] | 0.377 [0.352, 0.404] | 0.126 [0.111, 0.147] | 0.524 [0.498, 0.552] |

Dev tier by family:

| family | items | raw accuracy | h2 accuracy | raw ECE | h2 ECE |
|---|---|---|---|---|---|
| commonsense | 1,500 | 0.822 [0.801, 0.842] | 0.821 [0.801, 0.840] | 0.101 [0.088, 0.122] | 0.057 [0.042, 0.075] |
| inference | 750 | 0.639 [0.604, 0.673] | 0.631 [0.596, 0.665] | 0.313 [0.281, 0.349] | 0.144 [0.116, 0.180] |
| paraphrase | 500 | 0.766 [0.730, 0.802] | 0.754 [0.718, 0.788] | 0.206 [0.175, 0.248] | 0.080 [0.047, 0.115] |
| sentiment | 1,250 | 0.744 [0.721, 0.770] | 0.764 [0.739, 0.788] | 0.230 [0.206, 0.253] | 0.095 [0.076, 0.117] |
| similarity | 250 | 0.416 [0.356, 0.480] | 0.444 [0.384, 0.504] | 0.549 [0.488, 0.613] | 0.135 [0.083, 0.195] |
| word_sense | 250 | 0.628 [0.568, 0.684] | 0.564 [0.504, 0.628] | 0.353 [0.296, 0.412] | 0.259 [0.200, 0.316] |

Test tier (in-distribution) by family:

| family | items | raw accuracy | h2 accuracy | raw ECE | h2 ECE |
|---|---|---|---|---|---|
| contracts | 104 | 0.952 [0.913, 0.990] | 0.952 [0.904, 0.990] | 0.043 [0.013, 0.088] | 0.021 [0.015, 0.067] |
| emotion | 104 | 0.452 [0.356, 0.548] | 0.433 [0.337, 0.529] | 0.416 [0.332, 0.517] | 0.113 [0.080, 0.201] |
| generated | 104 | 0.606 [0.510, 0.702] | 0.740 [0.644, 0.817] | 0.246 [0.167, 0.329] | 0.110 [0.085, 0.179] |
| helpfulness | 104 | 0.471 [0.385, 0.558] | 0.452 [0.356, 0.548] | 0.495 [0.409, 0.584] | 0.176 [0.103, 0.276] |
| intent | 104 | 0.885 [0.817, 0.942] | 0.875 [0.808, 0.933] | 0.085 [0.048, 0.154] | 0.106 [0.092, 0.162] |
| knowledge | 414 | 0.713 [0.667, 0.754] | 0.696 [0.650, 0.737] | 0.186 [0.156, 0.233] | 0.039 [0.031, 0.087] |
| moderation | 208 | 0.510 [0.442, 0.577] | 0.663 [0.596, 0.726] | 0.430 [0.368, 0.501] | 0.099 [0.068, 0.140] |
| preference | 104 | 0.625 [0.529, 0.712] | 0.615 [0.519, 0.712] | 0.329 [0.251, 0.429] | 0.076 [0.045, 0.175] |
| reading | 104 | 0.875 [0.808, 0.933] | 0.885 [0.817, 0.942] | 0.090 [0.049, 0.155] | 0.100 [0.077, 0.159] |

## Results: stance format

| tier | items | condition | accuracy | log loss | Brier | ECE |
|---|---|---|---|---|---|---|
| test | 3,145 | raw | 0.680 [0.664, 0.697] | 4.786 [4.502, 5.102] | 0.580 [0.550, 0.611] | 0.276 [0.261, 0.294] |
| test | 3,145 | h2 | 0.755 [0.739, 0.770] | 0.624 [0.598, 0.651] | 0.351 [0.335, 0.367] | 0.017 [0.012, 0.033] |
| dev | 2,100 | raw | 0.549 [0.527, 0.570] | 7.057 [6.665, 7.485] | 0.829 [0.788, 0.870] | 0.396 [0.375, 0.419] |
| dev | 2,100 | h2 | 0.536 [0.514, 0.560] | 1.194 [1.145, 1.247] | 0.658 [0.630, 0.687] | 0.209 [0.189, 0.232] |
| final | 1,343 (315 groups) | raw | 0.778 [0.755, 0.801] | 3.185 [2.812, 3.589] | 0.410 [0.371, 0.453] | 0.187 [0.167, 0.212] |
| final | 1,343 (315 groups) | h2 | 0.773 [0.749, 0.795] | 0.594 [0.562, 0.628] | 0.337 [0.316, 0.359] | 0.052 [0.034, 0.077] |

Paired over the same fresh final-tier items, head minus raw was -0.005 [-0.024, 0.013] in
accuracy and -2.590 [-2.962, -2.238] in log loss. The final tier is one family (Check-COVID,
1,343 pairs on 315 abstracts).

Dev tier by family:

| family | items | raw accuracy | h2 accuracy | raw ECE | h2 ECE |
|---|---|---|---|---|---|
| dev_claims | 700 | 0.633 [0.597, 0.667] | 0.569 [0.530, 0.606] | 0.304 [0.271, 0.341] | 0.133 [0.105, 0.173] |
| dev_nli | 700 | 0.466 [0.430, 0.501] | 0.456 [0.419, 0.494] | 0.453 [0.418, 0.491] | 0.226 [0.191, 0.263] |
| dev_scientific | 700 | 0.549 [0.510, 0.584] | 0.583 [0.547, 0.621] | 0.431 [0.397, 0.470] | 0.273 [0.241, 0.311] |

Test tier (in-distribution) by family:

| family | items | raw accuracy | h2 accuracy | raw ECE | h2 ECE |
|---|---|---|---|---|---|
| fit_claims | 1,261 | 0.776 [0.752, 0.798] | 0.816 [0.793, 0.837] | 0.183 [0.163, 0.207] | 0.028 [0.021, 0.050] |
| fit_nli | 1,570 | 0.604 [0.582, 0.629] | 0.727 [0.704, 0.748] | 0.352 [0.330, 0.376] | 0.035 [0.023, 0.059] |
| fit_scientific | 314 | 0.678 [0.627, 0.733] | 0.646 [0.596, 0.701] | 0.272 [0.221, 0.322] | 0.052 [0.034, 0.112] |

Fresh final tier (Check-COVID) with the head, gold label (rows) by predicted label (columns):

| gold | contradicts | no_bearing | supports |
|---|---|---|---|
| contradicts | 390 | 18 | 43 |
| no_bearing | 116 | 260 | 66 |
| supports | 30 | 32 | 388 |

**Final-flagged (reported beside final, never pooled, judging no bar).** HealthFC:
The evidence sentences are the fact-checkers' own summary and often state the verdict ("we could not find any meaningful scientific evidence", verdict headings such as "Not checked"), so the no-bearing label (the verdict "insufficient evidence") can often be read off the wording, the flaw that excluded PubHealth; and 740 of the 749 claims are yes/no questions, not claims.

| tier | items | condition | accuracy | log loss | Brier | ECE |
|---|---|---|---|---|---|---|
| final-flagged | 749 (749 groups) | raw | 0.718 [0.684, 0.752] | 3.605 [3.081, 4.081] | 0.510 [0.450, 0.569] | 0.227 [0.196, 0.260] |
| final-flagged | 749 (749 groups) | h2 | 0.541 [0.506, 0.575] | 0.928 [0.877, 0.979] | 0.591 [0.556, 0.625] | 0.150 [0.130, 0.192] |

| gold | contradicts | no_bearing | supports |
|---|---|---|---|
| contradicts | 112 | 7 | 6 |
| no_bearing | 275 | 140 | 7 |
| supports | 18 | 31 | 153 |

**Final-seen (secondary: families seen during development).** HealthVer, an earlier held-out
stance tier. Its 2,100 pairs share 289 abstracts and carry no group, so these intervals resample pairs
as if independent and may be too narrow.

| tier | items | condition | accuracy | log loss | Brier | ECE |
|---|---|---|---|---|---|---|
| final-seen | 2,100 | raw | 0.571 [0.548, 0.592] | 5.891 [5.525, 6.277] | 0.766 [0.729, 0.807] | 0.354 [0.333, 0.377] |
| final-seen | 2,100 | h2 | 0.548 [0.527, 0.570] | 0.997 [0.966, 1.032] | 0.592 [0.572, 0.613] | 0.100 [0.084, 0.122] |

## Selective accuracy

Accuracy / share answered, with the head, when only answers whose top probability is above the
threshold are kept, and n, the number of answers kept (share times the tier's items in
`record.json`). Shares are rounded, so 0% can still hold a few answers; n/a means none
passed. Cells in italics rest on fewer than 50 answers and should not be read as
solid; the figures stop there for the same reason.

**General format:**

| tier | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|---|
| test | 0.83 / 69% (n = 938) | 0.89 / 59% (n = 792) | 0.92 / 50% (n = 670) | 0.96 / 39% (n = 530) | 0.99 / 29% (n = 393) | 0.99 / 19% (n = 263) | 1.00 / 4% (n = 57) |
| dev | 0.76 / 91% (n = 4,101) | 0.80 / 77% (n = 3,463) | 0.82 / 67% (n = 3,008) | 0.85 / 55% (n = 2,485) | 0.91 / 39% (n = 1,737) | 0.96 / 22% (n = 990) | *n/a / 0% (n = 0)* |
| final | 0.75 / 72% (n = 5,680) | 0.82 / 52% (n = 4,136) | 0.86 / 43% (n = 3,366) | 0.88 / 34% (n = 2,694) | 0.90 / 21% (n = 1,623) | 0.90 / 11% (n = 857) | *n/a / 0% (n = 0)* |
| final-flagged | 0.51 / 52% (n = 525) | 0.70 / 13% (n = 133) | *0.81 / 2% (n = 21)* | *n/a / 0% (n = 0)* | *n/a / 0% (n = 0)* | *n/a / 0% (n = 0)* | *n/a / 0% (n = 0)* |
| final-seen | 0.80 / 64% (n = 6,634) | 0.84 / 56% (n = 5,746) | 0.88 / 48% (n = 4,916) | 0.91 / 39% (n = 3,969) | 0.94 / 22% (n = 2,257) | 0.96 / 9% (n = 925) | *1.00 / 0% (n = 3)* |
| bench | 0.67 / 57% (n = 1,277) | 0.68 / 46% (n = 1,025) | 0.70 / 34% (n = 767) | 0.75 / 23% (n = 524) | 0.82 / 15% (n = 334) | 0.87 / 8% (n = 181) | *0.86 / 1% (n = 14)* |

**Stance format:**

| tier | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|---|
| test | 0.79 / 89% (n = 2,793) | 0.83 / 76% (n = 2,376) | 0.87 / 60% (n = 1,901) | 0.91 / 44% (n = 1,384) | 0.95 / 23% (n = 710) | 0.98 / 6% (n = 196) | *n/a / 0% (n = 0)* |
| dev | 0.56 / 87% (n = 1,834) | 0.58 / 74% (n = 1,563) | 0.61 / 62% (n = 1,299) | 0.64 / 48% (n = 1,007) | 0.71 / 27% (n = 576) | 0.88 / 7% (n = 141) | *n/a / 0% (n = 0)* |
| final | 0.81 / 87% (n = 1,165) | 0.86 / 71% (n = 959) | 0.88 / 58% (n = 773) | 0.91 / 40% (n = 542) | 0.94 / 17% (n = 226) | *0.96 / 4% (n = 48)* | *n/a / 0% (n = 0)* |
| final-flagged | 0.55 / 86% (n = 646) | 0.55 / 65% (n = 486) | 0.57 / 44% (n = 329) | 0.64 / 27% (n = 203) | *0.84 / 4% (n = 31)* | *n/a / 0% (n = 0)* | *n/a / 0% (n = 0)* |
| final-seen | 0.59 / 76% (n = 1,595) | 0.61 / 54% (n = 1,142) | 0.65 / 38% (n = 797) | 0.67 / 22% (n = 463) | 0.79 / 5% (n = 101) | *0.83 / 1% (n = 12)* | *n/a / 0% (n = 0)* |

## Known weaknesses

- **The general head costs accuracy on the fresh final tier.** Pooled, accuracy went from
  0.656 to 0.639 (paired difference -0.017 [-0.024, -0.011]). The largest drops were on grammar (0.717 to 0.635), difficulty (0.287 to 0.253), tables (0.722 to 0.699).
  A head is meant to change how far to trust the answers, not which answer is on top; on
  these families it also changed some top answers, more often for the worse.
- **Calibration varies by family.** Pooled final-tier ECE with the head was 0.030, but averaged
  over the eight families it was 0.077 [0.071, 0.088], with ethics at 0.114 [0.089, 0.142] and figurative at 0.098 [0.077, 0.124].
- **Sentence difficulty** (CEFR-SP, a six-level score question): accuracy 0.287 raw and
  0.253 [0.223, 0.283] with the head, n = 874. The head fixes the
  calibration (ECE 0.626 to 0.084), not the accuracy. Rating-scale questions remain the
  weakest question type: sentence similarity (STS-B, dev) reached 0.444 with the head.
- **The stance head misses its bars.** Final-tier ECE 0.052 [0.034, 0.077] against the bar of 0.05;
  dev-tier ECE 0.209 [0.189, 0.232],
  above the dev bar of 0.08, worst on scientific abstracts (dev_scientific 0.273 [0.241, 0.311]).
  On the final tier its most common error is calling no-bearing evidence "contradicts" (116 of 442
  no-bearing pairs). The general head was not scored on stance questions.
- **HealthFC (final-flagged)**: the stance head lowered accuracy from 0.718 to
  0.541 [0.506, 0.575] (ECE 0.150, n = 749); its most common error there is
  calling no-bearing pairs "contradicts" (275 of 422). The tier carries a caveat (the evidence
  often states the verdict), so it judges no bar.
- **Politeness (final-flagged)**, a three-level rating: accuracy 0.492 raw and
  0.487 [0.456, 0.519] with the head, ECE 0.046 [0.030, 0.080], n = 1,000.
- **HealthVer (final-seen)**: stance head ECE 0.100 [0.084, 0.122], accuracy 0.571 raw and
  0.548 with the head, n = 2,100.
- **Financial tweets (final-seen)** are poorly calibrated even with the head: ECE
  0.253 [0.223, 0.294], accuracy 0.481 [0.446, 0.521], n = 700.
- **Legal reasoning** (CaseHOLD, final-seen, five options, so 0.2 by chance): 0.280 raw and
  0.300 [0.267, 0.337] with the head, n = 700.
- **Word sense** (WiC, dev tier): the head lowered accuracy from 0.628 [0.568, 0.684] to
  0.564 [0.504, 0.628], n = 250, with ECE 0.259.
- **typed-decisions (bench)**: accuracy 0.591 [0.567, 0.614] and ECE 0.126 [0.111, 0.147] with the head,
  clearly less well calibrated than on the final tier.

## Determinism checks

`s1-selftest` ran before the runs; its output is
[selftest.txt](../results/qwen3-4b-q8/selftest.txt), copied in full:

```
INFO slots  " A" chosen  mean slot mass: plain 0.0000, leading space 1.0000
PASS T1  measured 0.000000  tolerance 0.000000  prompt integrity, mismatches and forged markers
PASS T2  measured 0.000010  tolerance 0.050000  rows against logits, max |d logit|
PASS T3  measured 0.008565  tolerance 0.020000  branch against recompute, max |d p| (max |d logp| 0.066569)
PASS T4  measured 0.006503  tolerance 0.020000  packed and shared against single, max |d p| (max |d logp| 0.161266)
PASS T5  measured 0.001422  tolerance 0.010000  isolation from sibling questions, max |d p|
SKIP T6  no --reference given; the acceptance check always gives one
PASS T7  measured 0.000000  tolerance 0.000100  gradient check H1 and H2, max relative error
PASS T8  measured 0.014633  tolerance 0.020000  cached features against live engine, max |d p|
PASS T9  measured 0.000000  tolerance 0.000000  rotation bookkeeping, violations
SKIP T10  the model has no sliding-window layers
PASS: 0 test(s) failed
```

The two branching checks that matter most for determinism passed with margin: T3 (branching
from the cached text against recomputing it) at 0.0086 and T4 (packed and shared batches against
single requests) at 0.0065, both against a tolerance of 0.02 in probability. T5 (isolation from
sibling questions) moved a question's probabilities by at most 0.0014. T10 does not apply (the
model has no sliding-window layers). T6, the independent CPU reference, was not run by judgly's
pipeline, so differences between this machine and other hardware are not checked here.

## Licences

- judgly's code and the template: Apache-2.0.
- General head (`heads/h2.bin`): Apache-2.0.
- Stance head (`heads/h2-stance.bin`): CC-BY-SA-4.0, because its fit data includes share-alike
  sources (MNLI, VitaminC, FEVER, SNLI, SciNLI).
- Model weights: not included; Apache-2.0 (Qwen team, Alibaba Cloud); GGUF conversion by Unsloth.

## Citation

Please cite judgly with the metadata in [CITATION.cff](../../CITATION.cff), and also cite
llama.cpp, the base model (Qwen/Qwen3-4B-Instruct-2507) and the datasets behind any number you quote
([reproduce.md](../reproduce.md#data-sources)). The card format is from:

- Mitchell, Wu, Zaldivar, Barnes, Vasserman, Hutchinson, Spitzer, Raji and Gebru 2019. Model
  Cards for Model Reporting. FAT* '19. arXiv:1810.03993.
