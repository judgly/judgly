# Reproducing the results

This page lists everything needed to regenerate the published numbers: the data, the code, the
commands, the hardware and where each output goes. It aims at the FAIR principles: the data sources are findable (pinned identifiers), accessible (public
downloads), interoperable (JSONL and documented binary formats) and reusable (licences stated
per source).

<!-- RESULTS:RUNTIME -->
> **Status.** The runs behind the published numbers are finished (one run per pack on an Apple
> M3 Max, with at most four option orders and no content-free pass;
> [methods.md](methods.md#release-settings)). Their outputs are committed in [results/](results/):
> the calibration records and tables (the fresh final-tier metrics to compare a rerun against are
> the first table of each `tables.md`), the per-item dumps and the self-test output, with SHA-256
> checksums in `results/MANIFEST`. The SHA-256 of the large outputs that are not committed
> (merged features, heads, tier files) are in `results/INPUTS.sha256`. A full run takes about
> 15 hours for Gemma 4 12B and about 5 hours for Qwen3-4B on an Apple M3 Max with 64 GB. No
> tolerance that counts as reproduced on other hardware has been set.

## Contents

- [Environment](#environment)
- [Steps](#steps)
- [Outputs](#outputs)
- [The committed snapshot](#the-committed-snapshot)
- [Checks that must pass](#checks-that-must-pass)
- [Data sources](#data-sources)

## Environment

| | used for the release runs |
|---|---|
| hardware | Apple M3 Max, 64 GB unified memory, Metal |
| OS | macOS 27.0 |
| compiler | Xcode command-line tools, linked against the MacOSX26.5 SDK (see [installation.md](installation.md#troubleshooting)) |
| llama.cpp | commit `6f41ac59e0a49a00483a316a22ada6b04edd2950` (git submodule; the build refuses another) |
| Python | 3.13, dependencies locked in `uv.lock` (`datasets` 5.0.1 built the tiers) |
| models | pinned by Hugging Face revision and SHA-256 in `src/judgly/packs/*/pack.json` |

Everything except the model runs needs only a CPU and network access.

**Disk and network.** The Hugging Face caches take about 6 GB after building the tiers (the
`datasets` cache about 5.2 GB, the files downloaded directly about 0.6 GB, most of it the first
ESCI shard); the tiers themselves 52 MB and the raw downloads (MultiVerS, Check-COVID, HealthFC,
JevBench) 6.9 MB. The model files are 4.3 GB
(Qwen3-4B) and 12.7 GB (Gemma 4 12B). A full run's features, shards and reports take about
4 GB for Qwen3-4B and 6 GB for Gemma 4 12B. Network
is needed for the downloads only.

## Steps

From a source checkout ([installation.md](installation.md#build-from-source)):

```bash
git clone --recurse-submodules https://github.com/judgly/judgly
cd judgly
uv sync --group pipeline
```

**1. Licences (network, minutes).** Re-read every dataset card at its pinned revision and
check the registry against it, including the licence policy:

```bash
make licences
```

**2. Data (network and CPU, under an hour).** Download the raw files (the MultiVerS release,
Check-COVID, HealthFC and the JevBench public items, each SHA-256 checked), build the tiers from
them and from the Hugging Face datasets at their pinned revisions, and run the contamination
checker:

```bash
make fetch data check
make verify-data        # the built tiers equal data/tiers.sha256, byte for byte
```

The tiers are not committed (52 MB, and not all sources may be redistributed); they are
rebuilt from the sources, and `make verify-data` proves the rebuild is identical (the QUICK
tiers likewise: `make data check verify-data QUICK=1` against `data/tiers-quick.sha256`).
`make pack` on the full tiers runs the same check before any GPU work and stops on a mismatch.

If a download fails its check: the MultiVerS tarball is fetched from its `latest` URL, which
its authors could replace (the other raw files are fetched at a pinned commit); a SHA-256 mismatch then means the upstream file changed, and the
stance tiers cannot be rebuilt identically without the original file (no archival copy is
recorded yet). Several Hugging Face sources (Cosmos QA, PIQA, Social IQa, TREC, CaseHOLD) are
pinned to a commit of the Hub's automatic `refs/convert/parquet` branch, which the Hub
regenerates; if such a revision stops resolving, the tiers for that source cannot be rebuilt
identically either. In both cases `make verify-data` reports which files differ.

**3. Tests (CPU, a minute; a GPU for the model tests).**

```bash
uv run pytest -q -rs
uv run --group pipeline pytest -q tests/test_pipeline.py
```

Without a model file, the tests that run the engine skip (`-rs` lists them). To run them, set
`JUDGLY_MODEL_DIR` to a directory holding the Qwen3-4B GGUF file named in
`src/judgly/packs/qwen3-4b-q8/pack.json` (or `JUDGLY_TEST_MODEL` to the file itself). They run
the built-in pack with heads off and with its shipped heads, and compare the answers with the
golden values in `tests/fixtures/golden-qwen3-4b-q8.json` within 1e-6
(`scripts/make_golden.py` writes them).

**4. Smoke run (GPU, about 20 minutes with Qwen3-4B).** The whole pipeline on tiny tiers, to
check the plumbing before committing hours of GPU time:

```bash
make pack QUICK=1 MODEL=qwen3-4b-q8 MODEL_DIR=/path/to/models
JUDGLY_MODEL_DIR=/path/to/models uv run python scripts/smoke_pack.py results-quick/qwen3-4b-q8/pack
```

A QUICK pack is marked as such and cannot be installed. The QUICK pipeline neither extracts nor
scores the fresh final and final-flagged tiers, which are read once, after both heads are
frozen; one QUICK smoke run with a throwaway head did score 173 items drawn from them (80
general, 93 stance), and the tiers were not changed because of it. A smoke run with Qwen3-4B takes about
20 minutes from start to finished pack, about 8 of them in the self-test gate
(`make run QUICK=1 ...` keeps a `run.log` with timestamps).

`MODEL_DIR` must hold the pack's GGUF file; if it does not, the run stops instead of
downloading. A resumed run checks that the model, template and engine options are the ones
the finished shards were made with, and stops if they are not.

**5. Full runs (GPU, hours; one at a time).** Each run is resumable: it works in shards, and
a stopped run keeps its finished shards.

```bash
make run MODEL=gemma4-12b-q8 MODEL_DIR=/path/to/models   # leave MODEL_DIR out to download
make status MODEL=gemma4-12b-q8
make stop MODEL=gemma4-12b-q8                            # make run resumes from here
make run MODEL=qwen3-4b-q8 MODEL_DIR=/path/to/models
```

The release settings are at most four option orders per question and no content-free pass
(the packs' `engine` block). The tiers hold 9,000 general fit, 4,500 general dev, 7,879 general
final, 1,000 general final-flagged, 10,300 general final-seen and 2,231 bench items, and 21,000
stance fit, 2,100 stance dev, 1,343 stance final, 749 stance final-flagged and 2,100 stance
final-seen items. A results directory holds the shards of
one tier build: start a full run in a fresh `results/` (move an older one aside), since shards
made from other tier files are not detected. Do not run another GPU-heavy program at the same
time.

A full run also fits the second calibration option, the per-type temperature, next to H2, and
scores both on every tier, the confirm tier included (`data/tiers/<format>/confirm.jsonl`, built
by `make data` after all the other tiers; its SHA-256 is in `data/tiers-confirm.sha256`:
`(cd data/tiers && shasum -a 256 -c ../tiers-confirm.sha256)`).

**5b. Calibration options from cached features (CPU, minutes).** With the features of a finished
run in `results/`, and the confirm tier's features as the confirmation extracted them in
`results-confirm/<pack>/<format>/confirm.feat` (set `CONFIRM_FROM` for another place):

```bash
make calibrate MODEL=gemma4-12b-q8
make calibrate MODEL=qwen3-4b-q8
uv run --no-project --with numpy python docs/tools/confirmation_check.py
```

`make calibrate` fits `temperature.bin` per format (`s1-train --head temperature`), evaluates
raw, h2 and temperature on every tier, writes the records and assembles the pack; it never runs
the model (every feature file is passed to make as old, and a missing one stops it), and it
refits H2 only if `h2.bin` is missing. `confirmation_check.py` checks the shipped temperatures
and the engine's temperature output against the frozen confirmation
([calibration-options.md](calibration-options.md#reproducing-the-numbers)).

**6. Check the snapshot and redraw the figures (CPU, seconds).** From the repository root:

```bash
(cd docs/results && shasum -a 256 -c MANIFEST)   # the committed snapshot is intact
shasum -a 256 -c docs/results/INPUTS.sha256      # your features, heads and tiers equal the release ones
make figures                                     # redraws docs/assets/results from docs/results
git status docs/assets/results                   # unchanged: the figures are byte-identical
```

The second check needs the run outputs in `results/` and the built tiers. `make figures` checks
every plotted number against `record.json` and `tables.md`; byte-identical figures were checked
with the matplotlib version in `uv.lock` on macOS. To redraw the figures without building the
native library, run the script with the figures group only:

```bash
uv run --only-group figures python scripts/make_figures.py
uv run --only-group figures python docs/tools/final_tier_stats.py   # paired differences, per-family ECE
```

**7. Install the pack** after reading its tables:

```bash
make install-pack MODEL=gemma4-12b-q8
```

This copies the heads and calibration records into `src/judgly/packs/gemma4-12b-q8/`. `NOTICE`
and each pack's `LICENSE` name the heads' fitting data (the record's `head.fitted_on`) with
their licences, the Wikipedia credit (CC BY-SA 3.0 / 4.0) for VitaminC and FEVER and the
Flickr30k credit for SNLI ([licences.md](licences.md#4-notice-licenses-and-the-packs)); a head
fitted on other sources needs them updated to match.

## Outputs

Per pack, in `results/<pack>/`:

| file | what |
|---|---|
| `run.log`, `run.pid` | the background run's log (with start and stop times) and process id (`make run`) |
| `pack-info.mk` | the model path and SHA-256, the template and its SHA-256, and the engine options the run used |
| `verify-data.txt` | the tier check against `data/tiers.sha256` (full tiers only) |
| `contamination.txt` | the contamination checker's output |
| `selftest.txt`, `selftest.log` | the self-test gate and any accepted failures |
| `<format>/{fitdev,final,final-flagged,confirm,final-seen,bench}/shards/shard-NNNN.{feat,feat.names.tsv,log}` | extracted shards and their logs (bench: general only) |
| `<format>/{fitdev,final,final-flagged,confirm,final-seen,bench}/features.feat`, `.names.tsv` | extracted features (merged from `shards/`; for confirm, a link made by `make calibrate` to the confirmation's extraction) and the task and family names |
| `<format>/h2.bin`, `h2.bin.json`, `train-h2.log` | the fitted head, the trainer's sidecar (features SHA-256, lambda per question type, validation losses) and its log |
| `<format>/temperature.bin`, `temperature.bin.json`, `train-temperature.log` | the per-type temperature, its sidecar (per type: train and validation items, the temperature and its unrounded value, losses, fallback) and its log |
| `<format>/{raw,h2,temperature}-{test,dev,final,final-flagged,confirm,final-seen,bench}.{json,txt}`, `eval.done` | the evaluation reports (bench: general only), and the marker that all finished |
| `<format>/items-{raw,h2,temperature}-{test,dev,final,final-flagged,confirm,final-seen,bench}.tsv` | per-item probabilities, for recomputing any number |
| `<format>/record.json`, `tables.md`, `tables.txt` | the calibration record (metrics with intervals, resampled by group on final, final-flagged and bench, per family, selective accuracy, stance confusion matrix, the bench scored against each benchmark's own gold, sources and the licences they are used under, model and template SHA-256, engine options, self-test output, judgly and llama.cpp commits, platform), its tables (final-flagged, with each family's caveat, and final-seen in tables of their own), and the tables as printed during the run |
| `pack/` | the finished pack: `pack.json`, template, `heads/` (with each head's sidecar), `calibration/` |

`<format>` is `general` or `stance`. Every file is written under a temporary name and renamed
when complete, so a file that exists is a file that finished.

## The committed snapshot

`docs/results/<pack>/` holds, per pack, `selftest.txt` (the self-test gate) and per format
(`general`, `stance`):

- **`record.json`**, the calibration record: `schema` (3), `pack`, `format`, `quick`, `head`
  (`kind`, `sha256`, `licence`, `fitted_on`: source, licences, citation and item count per
  fitting source), `temperature` (the second option: `sha256`, `licence`, how it is applied and
  fitted, its per-type sidecar entries and training log), `conditions` (what `raw`, `h2` and
  `temperature` mean), `data`, `model` (file, SHA-256,
  template SHA-256, `engine` settings), `software` (judgly and llama.cpp commits), `platform`,
  `selftest` (the gate's lines), `metrics_note`, and `tiers`. Each of
  `tiers.<test|dev|final|final-flagged|confirm|final-seen|bench>` (bench: general only) has a
  `note`, `items`, `groups`, `sources` (items per task; final-flagged and confirm also
  `caveats`) and `conditions.<raw|h2|temperature>` with `items`, `interval_unit` (`item` or `group`), `metrics`
  (`accuracy`, `log_loss`, `brier`, `ece`, `slot_mass`, `slot_mass_p05`, `rps`, each a point
  value with its 95% interval; on grouped tiers also `groups` and `metrics_item_resampled`),
  `reliability` (ten bins), `selective` (seven thresholds), `by_family` (accuracy, log loss,
  Brier score and ECE per family, with intervals), for stance `confusion`, and in the bench tier
  `benchmark` (each benchmark scored against its own gold).
- **`tables.md`**: the record as tables. **`train-h2.log`**, **`train-temperature.log`**: the
  trainers' logs.
- **`items-<raw|h2|temperature>-<test|dev|final|final-flagged|confirm|final-seen|bench>.tsv.gz`**, one line per item after a header, tab-separated:
  `id_hash` (16 hex digits, the 64-bit FNV-1a hash of the item id), `task_id` (the low 32 bits
  of the FNV-1a-64 hash of the task name), `family_id` (the low 16 bits of the hash of the family
  name), `type` (0 choice, 1 yes or no, 2 score), `K` (number of options), `label` (index of the
  correct option, from 0) and `p` (the probability of every option, comma-separated, in option
  order).
- **`names-<fitdev|final|final-flagged|confirm|final-seen|bench>.tsv.gz`**, tab-separated without a header: `task` or `family`, the id
  as in the item dumps, and the name.

`docs/results/calibration-study/` holds the exploratory analyses and the pre-registered
confirmation behind the temperature option, as they were run
([its README](results/calibration-study/README.md), with one README per analysis).
`uv run --no-project --with numpy --with scipy --with scikit-learn python
docs/results/calibration-study/rerun.py all` reruns the unchanged scripts in a scratch workspace
(after `make data tools` and a finished pack run in `results/`) and compares their output with the
committed files.

`docs/results/MANIFEST` has the SHA-256 of every file in the snapshot, and
`docs/results/INPUTS.sha256` those of the large inputs that are not committed. The snapshot and
the figures made from it are under CC-BY-4.0 ([licences.md](licences.md#5-results-and-figures)).

## Checks that must pass

- `make licences`: every card matches the registry and the policy holds.
- `make check`: `PASS contamination` for both formats.
- `make verify-data`: the tiers equal `data/tiers.sha256`; the confirm tier equals
  `data/tiers-confirm.sha256` (`cd data/tiers && shasum -a 256 -c ../tiers-confirm.sha256`).
- The self-test gate: all checks within tolerance ([methods.md](methods.md#engine-self-tests)),
  except T6 (the independent CPU reference), which the pipeline does not run: its output says
  `SKIP T6`, and the difference between this hardware and others is therefore not checked.
- `scripts/calibration_record.py` stops if its recomputed metrics differ from the evaluator's.
- `docs/tools/confirmation_check.py`: the shipped temperatures are the confirmed ones, and the
  engine's temperature output equals the frozen confirmation scorer's.

## Data sources

Every source judgly's heads are fitted or evaluated on, from `data/registry.yaml`. The
licence is as written on the dataset card at the pinned revision; "not used" marks sources the
licence policy excludes ([methods.md](methods.md#data-and-tiers)). Only permissive sources (and
share-alike ones for stance) are used for fitting; everything else is evaluation only. Please
cite the datasets you rely on.

<!-- SOURCES:BEGIN (generated by docs/tools/sources_table.py) -->
Registry: `data/registry.yaml`, schema 1, licences checked 2026-09-27, SHA-256 `404eaab9455ed7be8b6fd325d232ad5b0d293f44fbadef8894528f680a6adaee`.

**general**

| source | family | tier | where | revision | licence (as on the card) | citation |
|---|---|---|---|---|---|---|
| `banking77` | intent | fit | [`legacy-datasets/banking77`](https://huggingface.co/datasets/legacy-datasets/banking77) | `f54121560de4` | cc-by-4.0 | Casanueva et al. 2020, Efficient Intent Detection with Dual Sentence Encoders. NLP4ConvAI workshop. |
| `civil_comments` | moderation | fit | [`google/civil_comments`](https://huggingface.co/datasets/google/civil_comments) | `f2970eb3a557` | cc0-1.0 | Borkan et al. 2019, Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification. WWW Companion. |
| `civil_comments_score` | moderation | fit | [`google/civil_comments`](https://huggingface.co/datasets/google/civil_comments) | `f2970eb3a557` | cc0-1.0 | Borkan et al. 2019 (as civil_comments). |
| `commonsense_qa` | knowledge | fit | [`tau/commonsense_qa`](https://huggingface.co/datasets/tau/commonsense_qa) | `94630fe30dad` | mit | Talmor et al. 2019, CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge. NAACL. |
| `cosmos_qa` | reading | fit | [`allenai/cosmos_qa`](https://huggingface.co/datasets/allenai/cosmos_qa) | `ed50fe83db62` | cc-by-4.0 | Huang et al. 2019, Cosmos QA: Machine Reading Comprehension with Contextual Commonsense Reasoning. EMNLP. |
| `generated` | generated | fit | generated by `scripts/prep_tiers.py` | n/a | apache-2.0 | judgly (this repository). |
| `go_emotions` | emotion | fit | [`google-research-datasets/go_emotions`](https://huggingface.co/datasets/google-research-datasets/go_emotions) | `add492243ff9` | apache-2.0 | Demszky et al. 2020, GoEmotions: A Dataset of Fine-Grained Emotions. ACL. |
| `helpsteer2` | helpfulness | fit | [`nvidia/HelpSteer2`](https://huggingface.co/datasets/nvidia/HelpSteer2) | `990b2711a361` | cc-by-4.0 | Wang et al. 2024, HelpSteer2: Open-source dataset for training top-performing reward models. arXiv:2406.08673. |
| `hh_rlhf` | preference | fit | [`Anthropic/hh-rlhf`](https://huggingface.co/datasets/Anthropic/hh-rlhf) | `09be8c5bbc57` | mit | Bai et al. 2022, Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. arXiv:2204.05862. |
| `ledgar` | contracts | fit | [`coastalcph/lex_glue`](https://huggingface.co/datasets/coastalcph/lex_glue) | `c23fdff1a6bf` | cc-by-4.0 | Tuggener et al. 2020, LEDGAR: A Large-Scale Multi-label Corpus for Text Classification of Legal Provisions in Contracts. LREC; Chalkidis et al. 2022, LexGLUE. ACL. |
| `mmlu` | knowledge | fit | [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu) | `c30699e8356d` | mit | Hendrycks et al. 2021, Measuring Massive Multitask Language Understanding. ICLR. |
| `qasc` | knowledge | fit | [`allenai/qasc`](https://huggingface.co/datasets/allenai/qasc) | `a34ba204eb9a` | cc-by-4.0 | Khot et al. 2020, QASC: A Dataset for Question Answering via Sentence Composition. AAAI. |
| `strategyqa` | knowledge | fit | [`ChilleD/StrategyQA`](https://huggingface.co/datasets/ChilleD/StrategyQA) | `705562638fe1` | mit | Geva et al. 2021, Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies. TACL. |
| `anli_general` | inference | dev | [`facebook/anli`](https://huggingface.co/datasets/facebook/anli) | `8e4813d81f46` | cc-by-nc-4.0 | Nie et al. 2020, Adversarial NLI: A New Benchmark for Natural Language Understanding. ACL. |
| `cb` | inference | dev | [`aps/super_glue`](https://huggingface.co/datasets/aps/super_glue) | `3de24cf8022e` | other | Wang et al. 2019, SuperGLUE. NeurIPS; de Marneffe et al. 2019, The CommitmentBank. Sinn und Bedeutung. |
| `copa` | commonsense | dev | [`aps/super_glue`](https://huggingface.co/datasets/aps/super_glue) | `3de24cf8022e` | other | Roemmele et al. 2011, Choice of Plausible Alternatives. AAAI Spring Symposium. |
| `hellaswag` | commonsense | dev | [`Rowan/hellaswag`](https://huggingface.co/datasets/Rowan/hellaswag) | `218ec52e09a7` | none on card | Zellers et al. 2019, HellaSwag: Can a Machine Really Finish Your Sentence? ACL. |
| `imdb` | sentiment | dev | [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) | `e6281661ce1c` | other | Maas et al. 2011, Learning Word Vectors for Sentiment Analysis. ACL. |
| `mrpc` | paraphrase | dev | [`SetFit/mrpc`](https://huggingface.co/datasets/SetFit/mrpc) | `2e2058c90792` | none on card | Dolan and Brockett 2005, Automatically Constructing a Corpus of Sentential Paraphrases. IWP. |
| `piqa` | commonsense | dev | [`ybisk/piqa`](https://huggingface.co/datasets/ybisk/piqa) | `142c51238b3c` | unknown | Bisk et al. 2020, PIQA: Reasoning about Physical Commonsense in Natural Language. AAAI. |
| `qqp` | paraphrase | dev | [`nyu-mll/glue`](https://huggingface.co/datasets/nyu-mll/glue) | `bcdcba79d07b` | other | Quora Question Pairs (Iyer et al. 2017), via GLUE (Wang et al. 2019). |
| `rotten_tomatoes` | sentiment | dev | [`cornell-movie-review-data/rotten_tomatoes`](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes) | `aa13bc287fa6` | unknown | Pang and Lee 2005, Seeing Stars: Exploiting Class Relationships for Sentiment Categorization. ACL. |
| `rte` | inference | dev | [`nyu-mll/glue`](https://huggingface.co/datasets/nyu-mll/glue) | `bcdcba79d07b` | other | Wang et al. 2019, GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding. ICLR; RTE: Dagan et al. 2006 and successors. |
| `siqa` | commonsense | dev | [`allenai/social_i_qa`](https://huggingface.co/datasets/allenai/social_i_qa) | `537a2ec8ec56` | none on card | Sap et al. 2019, Social IQa: Commonsense Reasoning about Social Interactions. EMNLP. |
| `sst2` | sentiment | dev | [`stanfordnlp/sst2`](https://huggingface.co/datasets/stanfordnlp/sst2) | `8d51e7e4887a` | unknown | Socher et al. 2013, Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. EMNLP. |
| `stsb` | similarity | dev | [`sentence-transformers/stsb`](https://huggingface.co/datasets/sentence-transformers/stsb) | `ab7a5ac0e35a` | none on card | Cer et al. 2017, SemEval-2017 Task 1: Semantic Textual Similarity. SemEval. |
| `swag` | commonsense | dev | [`allenai/swag`](https://huggingface.co/datasets/allenai/swag) | `dc48148372b3` | unknown | Zellers et al. 2018, SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference. EMNLP. |
| `tweet_irony` | sentiment | dev | [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval) | `b3a375baf0f4` | unknown | Barbieri et al. 2020 (as tweet_sentiment); Van Hee et al. 2018, SemEval-2018 Task 3. |
| `tweet_sentiment` | sentiment | dev | [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval) | `b3a375baf0f4` | unknown | Barbieri et al. 2020, TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification. Findings of EMNLP. |
| `wic` | word_sense | dev | [`aps/super_glue`](https://huggingface.co/datasets/aps/super_glue) | `3de24cf8022e` | other | Pilehvar and Camacho-Collados 2019, WiC: the Word-in-Context Dataset. NAACL. |
| `winogrande` | commonsense | dev | [`allenai/winogrande`](https://huggingface.co/datasets/allenai/winogrande) | `01e74176c635` | none on card | Sakaguchi et al. 2020, WinoGrande: An Adversarial Winograd Schema Challenge at Scale. AAAI. |
| `bbq` | social_bias | final | [`heegyu/bbq`](https://huggingface.co/datasets/heegyu/bbq) | `5d6faae52070` | cc-by-4.0 | Parrish et al. 2022, BBQ: A Hand-Built Bias Benchmark for Question Answering. Findings of ACL. |
| `blimp` | grammar | final | [`nyu-mll/blimp`](https://huggingface.co/datasets/nyu-mll/blimp) | `877fba0801ff` | cc-by-4.0 | Warstadt et al. 2020, BLiMP: The Benchmark of Linguistic Minimal Pairs for English. TACL. |
| `cefr_sp` | difficulty | final | [`UniversalCEFR/cefr_sp_en`](https://huggingface.co/datasets/UniversalCEFR/cefr_sp_en) | `b78901348bda` | cc-by-nc-sa-4.0 | Arase, Uchida and Kajiwara 2022, CEFR-Based Sentence Difficulty Annotation and Assessment. EMNLP. |
| `circa` | pragmatics | final | [`google-research-datasets/circa`](https://huggingface.co/datasets/google-research-datasets/circa) | `faa1b5a78dd9` | cc-by-4.0 | Louis, Roth and Radlinski 2020, "I'd rather just go to bed": Understanding Indirect Answers. EMNLP. |
| `esci` | relevance | final | [`tasksource/esci`](https://huggingface.co/datasets/tasksource/esci) | `8113b17a5d40` | apache-2.0 | Reddy et al. 2022, Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search. arXiv:2206.06588. |
| `ethics_deontology` | ethics | final | [`hendrycks/ethics`](https://huggingface.co/datasets/hendrycks/ethics) | `b8b47c589f8b` | mit | Hendrycks et al. 2021, Aligning AI With Shared Human Values. ICLR. |
| `ethics_justice` | ethics | final | [`hendrycks/ethics`](https://huggingface.co/datasets/hendrycks/ethics) | `b8b47c589f8b` | mit | Hendrycks et al. 2021 (as ethics_deontology). |
| `fig_qa` | figurative | final | [`nightingal3/fig-qa`](https://huggingface.co/datasets/nightingal3/fig-qa) | `b29ec2faf6ef` | mit | Liu et al. 2022, Testing the Ability of Language Models to Interpret Figurative Language. NAACL. |
| `tabfact` | tables | final | [`wenhu/tab_fact`](https://huggingface.co/datasets/wenhu/tab_fact) | `cecfbe87432b` | cc-by-4.0 | Chen et al. 2020, TabFact: A Large-scale Dataset for Table-based Fact Verification. ICLR. |
| `politeness` | politeness | final-flagged | [`Cleanlab/stanford-politeness`](https://huggingface.co/datasets/Cleanlab/stanford-politeness) | `9fdefb9b4206` | mit | Danescu-Niculescu-Mizil et al. 2013, A computational approach to politeness with application to social factors. ACL. |
| `ag_news` | topic | final-seen | [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) | `eb185aade064` | unknown | Zhang, Zhao and LeCun 2015, Character-level Convolutional Networks for Text Classification. NeurIPS. |
| `aqua_rat` | math | final-seen | [`deepmind/aqua_rat`](https://huggingface.co/datasets/deepmind/aqua_rat) | `33301c6a050c` | apache-2.0 | Ling et al. 2017, Program Induction by Rationale Generation: Learning to Solve and Explain Algebraic Word Problems. ACL. |
| `boolq` | yesno | final-seen | [`google/boolq`](https://huggingface.co/datasets/google/boolq) | `35b264d03638` | cc-by-sa-3.0 | Clark et al. 2019, BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions. NAACL. |
| `casehold` | legal | final-seen | [`casehold/casehold`](https://huggingface.co/datasets/casehold/casehold) | `8a4dbd58704b` | none on card | Zheng et al. 2021, When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset. ICAIL. |
| `dbpedia` | topic | final-seen | [`fancyzhx/dbpedia_14`](https://huggingface.co/datasets/fancyzhx/dbpedia_14) | `9abd46cf7fc8` | cc-by-sa-3.0 | Zhang, Zhao and LeCun 2015 (as ag_news); Lehmann et al. 2015, DBpedia. Semantic Web. |
| `fin_tweets` | finance | final-seen | [`zeroshot/twitter-financial-news-sentiment`](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | `ccbe24de388e` | mit | Twitter Financial News Sentiment, zeroshot on Hugging Face (no paper). |
| `med_qa` | biomedical | final-seen | [`bigbio/med_qa`](https://huggingface.co/datasets/bigbio/med_qa) | `484a6c066fe8` | unknown | Jin et al. 2021, What Disease Does This Patient Have? A Large-Scale Open Domain Question Answering Dataset from Medical Exams. Applied Sciences. |
| `medmcqa` | biomedical | final-seen | [`openlifescienceai/medmcqa`](https://huggingface.co/datasets/openlifescienceai/medmcqa) | `91c6572c4540` | apache-2.0 | Pal et al. 2022, MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering. CHIL. |
| `mmlu_pro` | knowledge_pro | final-seen | [`TIGER-Lab/MMLU-Pro`](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | `b189ec765aa7` | mit | Wang et al. 2024, MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. NeurIPS Datasets and Benchmarks. |
| `newsgroups` | topic | final-seen | [`SetFit/20_newsgroups`](https://huggingface.co/datasets/SetFit/20_newsgroups) | `f1b91292074e` | none on card | Lang 1995, NewsWeeder: Learning to Filter Netnews. ICML. |
| `pubmedqa` | biomedical | final-seen | [`qiaojin/PubMedQA`](https://huggingface.co/datasets/qiaojin/PubMedQA) | `9001f2853fb8` | mit | Jin et al. 2019, PubMedQA: A Dataset for Biomedical Research Question Answering. EMNLP. |
| `trec` | topic | final-seen | [`CogComp/trec`](https://huggingface.co/datasets/CogComp/trec) | `65752bf53af2` | unknown | Li and Roth 2002, Learning Question Classifiers. COLING. |
| `truthful_qa` | truthfulness | final-seen | [`truthfulqa/truthful_qa`](https://huggingface.co/datasets/truthfulqa/truthful_qa) | `741b8276f2d1` | apache-2.0 | Lin, Hilton and Evans 2022, TruthfulQA: Measuring How Models Mimic Human Falsehoods. ACL. |
| `yahoo` | topic | final-seen | [`community-datasets/yahoo_answers_topics`](https://huggingface.co/datasets/community-datasets/yahoo_answers_topics) | `6652a1e7c94f` | unknown | Zhang, Zhao and LeCun 2015 (as ag_news). |
| `yelp` | rating | final-seen | [`Yelp/yelp_review_full`](https://huggingface.co/datasets/Yelp/yelp_review_full) | `c1f9ee939b7d` | other | Zhang, Zhao and LeCun 2015 (as ag_news); Yelp Dataset Challenge. |
| `arc` | knowledge | not used | [`allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc) | `210d026faf99` | cc-by-sa-4.0 | Clark et al. 2018, Think you have Solved Question Answering? Try ARC. arXiv:1803.05457. |
| `argument_quality` | argument_quality | confirm | [`ibm-research/argument_quality_ranking_30k`](https://huggingface.co/datasets/ibm-research/argument_quality_ranking_30k) | `590726b3765b` | cc-by-3.0 | Gretz et al. 2020, A Large-scale Dataset for Argument Quality Ranking: Construction and Analysis. AAAI. |
| `clutrr` | kinship | confirm | [`tasksource/clutrr`](https://huggingface.co/datasets/tasksource/clutrr) | `3f0016e8d7bb` | none on card | Sinha et al. 2019, CLUTRR: A Diagnostic Benchmark for Inductive Reasoning from Text. EMNLP. |
| `code_outcome` | code_outcome | confirm | [`Fsoft-AIC/CodeMMLU`](https://huggingface.co/datasets/Fsoft-AIC/CodeMMLU) | `f7c1221269df` | mit | Manh et al. 2025, CodeMMLU: A Multi-Task Benchmark for Assessing Code Understanding Capabilities of CodeLLMs. ICLR; Puri et al. 2021, Project CodeNet. NeurIPS Datasets and Benchmarks. |
| `dair_emotion` | emotion | not used | [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) | `cab853a1dbdf` | other | Saravia et al. 2018, CARER: Contextualized Affect Representations for Emotion Recognition. EMNLP. |
| `humicroedit` | humour | confirm | [`tasksource/humicroedit`](https://huggingface.co/datasets/tasksource/humicroedit) | `f5a16e65b085` | unknown | Hossain, Krumm and Gamon 2019, "President Vows to Cut <Taxes> Hair": Dataset and Analysis of Creative Text Editing for Humorous Headlines. NAACL; Hossain et al. 2020, SemEval-2020 Task 7. SemEval. |
| `jevbench` | bench_jevbench | bench | [github.com/fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench) (below) | `1df665e3956d` | mit | Standhartinger 2026, JevBench (github.com/fstandhartinger/jevbench), v1.4.2. |
| `openbookqa` | knowledge | not used | [`allenai/openbookqa`](https://huggingface.co/datasets/allenai/openbookqa) | `388097ea7776` | unknown | Mihaylov et al. 2018, Can a Suit of Armor Conduct Electricity? EMNLP. |
| `race` | reading | not used | [`ehovy/race`](https://huggingface.co/datasets/ehovy/race) | `2fec9fd81f1d` | other | Lai et al. 2017, RACE: Large-scale ReAding Comprehension Dataset From Examinations. EMNLP. |
| `sciq` | knowledge | not used | [`allenai/sciq`](https://huggingface.co/datasets/allenai/sciq) | `2c94ad3e1aaf` | cc-by-nc-3.0 | Welbl et al. 2017, Crowdsourcing Multiple Choice Science Questions. W-NUT. |
| `spartqa` | spatial | confirm | [`tasksource/spartqa-yn`](https://huggingface.co/datasets/tasksource/spartqa-yn) | `150c819e88bb` | apache-2.0 | Mirzaee et al. 2021, SPARTQA: A Textual Question Answering Benchmark for Spatial Reasoning. NAACL. |
| `typed_decisions` | bench_typed_decisions | bench | [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | `f7a2487edd7a` | apache-2.0 | LocalLLaMA/typed-decisions on Hugging Face (dataset card, no paper). |

**stance**

| source | family | tier | where | revision | licence (as on the card) | citation |
|---|---|---|---|---|---|---|
| `fever` | fit_claims | fit | [`copenlu/fever_gold_evidence`](https://huggingface.co/datasets/copenlu/fever_gold_evidence) | `a6b8d891d393` | cc-by-sa-3.0, gpl-3.0 | Thorne et al. 2018, FEVER: a Large-scale Dataset for Fact Extraction and VERification. NAACL; evidence release: Atanasova, Wright and Augenstein 2020, Generating Label Cohesive and Well-Formed Adversarial Claims. EMNLP. |
| `mnli` | fit_nli | fit | [`nyu-mll/multi_nli`](https://huggingface.co/datasets/nyu-mll/multi_nli) | `da70db2af9d0` | cc-by-3.0, cc-by-sa-3.0, mit, other | Williams, Nangia and Bowman 2018, A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference. NAACL. |
| `scinli` | fit_scientific | fit | [`tasksource/scinli`](https://huggingface.co/datasets/tasksource/scinli) | `61419d7f2cec` | apache-2.0 | Sadat and Caragea 2022, SciNLI: A Corpus for Natural Language Inference on Scientific Text. ACL. |
| `snli` | fit_nli | fit | [`stanfordnlp/snli`](https://huggingface.co/datasets/stanfordnlp/snli) | `cdb5c3d5eed6` | cc-by-sa-4.0 | Bowman et al. 2015, A large annotated corpus for learning natural language inference. EMNLP. |
| `vitaminc` | fit_claims | fit | [`tals/vitaminc`](https://huggingface.co/datasets/tals/vitaminc) | `be6febb761b0` | cc-by-sa-3.0 | Schuster, Fisch and Barzilay 2021, Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence. NAACL. |
| `wanli` | fit_nli | fit | [`alisawuffles/WANLI`](https://huggingface.co/datasets/alisawuffles/WANLI) | `61c95318fd71` | cc-by-4.0 | Liu et al. 2022, WANLI: Worker and AI Collaboration for Natural Language Inference Dataset Creation. Findings of EMNLP. |
| `anli` | dev_nli | dev | [`facebook/anli`](https://huggingface.co/datasets/facebook/anli) | `8e4813d81f46` | cc-by-nc-4.0 | Nie et al. 2020 (as anli_general). |
| `climate_fever` | dev_claims | dev | [`tdiggelm/climate_fever`](https://huggingface.co/datasets/tdiggelm/climate_fever) | `ae61ccb9320a` | unknown | Diggelmann et al. 2020, CLIMATE-FEVER: A Dataset for Verification of Real-World Climate Claims. arXiv:2012.00614. |
| `covidfact` | dev_scientific | dev | MultiVerS release (below) | n/a | unconfirmed | Saakyan, Chakrabarty and Muresan 2021, COVID-Fact: Fact Extraction and Verification of Real-World Claims on COVID-19 Pandemic. ACL. |
| `check_covid` | final_covid_claims | final | [github.com/posuer/Check-COVID](https://github.com/posuer/Check-COVID) (below) | `3ae70f8cac4d` | mit | Wang et al. 2023, Check-COVID: Fact-Checking COVID-19 News Claims with Scientific Evidence. Findings of ACL. |
| `healthfc` | final_health_claims | final-flagged | [github.com/jvladika/HealthFC](https://github.com/jvladika/HealthFC) (below) | `9f31d765e5d4` | cc-by-nc-nd-4.0 | Vladika, Schneider and Matthes 2024, HealthFC: Verifying Health Claims with Evidence-Based Medical Fact-Checking. LREC-COLING. |
| `healthver` | final_scientific | final-seen | MultiVerS release (below) | n/a | unconfirmed | Sarrouti et al. 2021, Evidence-based Fact-Checking of Health-related Claims. Findings of EMNLP. |
| `climatecheck` | confirm_climate | confirm | [`rabuahmad/climatecheck`](https://huggingface.co/datasets/rabuahmad/climatecheck) | `93d0dc5007e9` | mit | ClimateCheck shared task, SDP 2025 workshop (rabuahmad/climatecheck on Hugging Face). |
| `scifact` | reserved_scientific | reserved | MultiVerS release (below) | n/a | unconfirmed | Wadden et al. 2020, Fact or Fiction: Verifying Scientific Claims. EMNLP. |
| `fever_nli` | unused | none | [`pietrolesci/nli_fever`](https://huggingface.co/datasets/pietrolesci/nli_fever) | `1eddac63112e` | none on card | Thorne et al. 2018, FEVER. NAACL; Nie, Chen and Bansal 2019, Combining Fact Extraction and Verification with Neural Semantic Matching Networks. AAAI. |

MultiVerS release: https://scifact.s3.us-west-2.amazonaws.com/longchecker/latest/data.tar.gz, SHA-256 `0ce25066f18572af3307438a08d0d318a734d97d1d8c27ca43b51e81ae3d683c`, 3,610,222 bytes. Wadden et al. 2022, MultiVerS: Improving scientific claim verification with weak supervision and full-document context. Findings of NAACL.

Other raw files (`make fetch`, SHA-256 checked):

| file | URL | SHA-256 | bytes |
|---|---|---|---|
| `data/raw/check_covid/Check-COVID_all.json` | https://raw.githubusercontent.com/posuer/Check-COVID/3ae70f8cac4df13a3ce56878e20d8e76790e422d/Check-COVID/Check-COVID_all.json | `008b930570d374530528efa9edc6b2a029189a6d21b844ce1857324039d9e697` | 447,025 |
| `data/raw/check_covid/corpus.json` | https://raw.githubusercontent.com/posuer/Check-COVID/3ae70f8cac4df13a3ce56878e20d8e76790e422d/Check-COVID/corpus.json | `2a56084b933f28ec02781a9e501082683ca4b9c4c1c6c8530190d6fb0002ce42` | 635,536 |
| `data/raw/healthfc/Datensatz.csv` | https://raw.githubusercontent.com/jvladika/HealthFC/9f31d765e5d4a99bd9fff667e82de503bc5684f5/Datensatz.csv | `6b832bd2a73af47f7a8d9dcd3c307defa962cff3746c26b33866170a406721c8` | 1,463,876 |
| `data/raw/jevbench/original.jsonl` | https://raw.githubusercontent.com/fstandhartinger/jevbench/1df665e3956d7aab7fa0208ff6c4f2d8557f9f90/datasets/public/original.jsonl | `5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180` | 57,237 |
| `data/raw/jevbench/easy.jsonl` | https://raw.githubusercontent.com/fstandhartinger/jevbench/1df665e3956d7aab7fa0208ff6c4f2d8557f9f90/datasets/public/easy.jsonl | `231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b` | 37,220 |
| `data/raw/jevbench/hard.jsonl` | https://raw.githubusercontent.com/fstandhartinger/jevbench/1df665e3956d7aab7fa0208ff6c4f2d8557f9f90/datasets/public/hard.jsonl | `89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb` | 651,848 |

Built tiers (`data/tiers.sha256`, checked by `make verify-data`):

| file | SHA-256 |
|---|---|
| `data/tiers/general/bench.jsonl` | `744e0066370d3d1169f87b0a15735388faa19c5a16739ab6c1c7b8dd52cdecc3` |
| `data/tiers/general/final-flagged.jsonl` | `d14ff4b0aa081ba5a18d45c018c652b7f130647d27bd0d035837a660bdc6fc50` |
| `data/tiers/general/final-seen.jsonl` | `900116dafcaec4b67d4a735e8f064bed2250e7025468604638ef64beecc17cc2` |
| `data/tiers/general/final.jsonl` | `f66810a2b35fa29ca34d3d989427acf2c52e5279f9836b4dfd22e6ac64a30366` |
| `data/tiers/general/fitdev.jsonl` | `6723f2a5767c64b8e4695ddee91e8402c706e0f10bd07328c9c77bd6d6bef65f` |
| `data/tiers/stance/final-flagged.jsonl` | `e692a789ad61367c69a2c12a57763fb452a258d3a3c8b4e7eba7ae06f174dae2` |
| `data/tiers/stance/final-seen.jsonl` | `315d67a0796217ad9583eb879521fd3b636c57d50ff4b3a56d4d52af339f038f` |
| `data/tiers/stance/final.jsonl` | `3e12644c85469f8ab0b82a04d03d746b9b5216c59d2115a39c29d402a8c681f5` |
| `data/tiers/stance/fitdev.jsonl` | `3176629a23f3afaf48f498a79f00917bcef402a8288e8f1d22267900a132c538` |
<!-- SOURCES:END -->
