# The data and head pipeline: tiers, heads, calibration records and model packs.
# docs/reproduce.md describes how to run it.
#
#   make pack MODEL=gemma4-12b-q8     everything for one pack, in the foreground
#   make run MODEL=gemma4-12b-q8      the same in the background (resumes where it stopped)
#   make stop | make status           stop it (finished steps and shards are kept) | progress
#   make fetch                        download the raw (non Hugging Face) files, SHA-256 checked
#   make data | make check            build the tiers | check them for contamination
#   make verify-data                  the tiers equal data/tiers.sha256 (-quick with QUICK=1), byte for byte
#   make licences                     check every licence against its dataset card
#   make calibrate MODEL=...          CPU only, from features already extracted: fit the per-type
#                                     temperature, evaluate raw, h2 and temperature on every tier
#                                     (confirm included), write the records and the pack; never
#                                     extracts (see "Calibration options" below)
#   make install-pack MODEL=...       copy a finished (non-QUICK) pack into src/judgly/packs
#   make figures                      the results figures from the snapshot in docs/results
#   make compare-score                CPU only: rescore the committed answers of the external comparison
#                                     with its frozen scorer and check the results equal the committed ones
#   make compare-run                  ask the Ollama models of the external comparison again (Ollama >= 0.35.0,
#                                     models pulled), into COMPARE_OUT (default results-compare), never the record
#
#   QUICK=1          small tiers (data/tiers-quick, results-quick) for a smoke run in minutes; the
#                    fresh final and final-flagged tiers and the confirm tier are neither extracted
#                    nor scored, since their quick items are items of the real tiers
#   MODEL_DIR=DIR    take the GGUF file from DIR instead of downloading it (JUDGLY_MODEL_DIR)
#
# One pack runs: tools, tiers, contamination check, self-test gate, then per format (general,
# stance) sharded extraction of every example set (fitdev: fit and dev; final: the fresh final
# tier; final-flagged: fresh families with a recorded caveat, reported beside final; confirm: the
# untouched tier of the pre-registered confirmation of the temperature; final-seen: an earlier
# held-out tier; bench: external benchmarks, general only), the fit of both calibration options
# (H2 and the per-type temperature), eval of the raw and both fitted conditions on the test, dev,
# final, final-flagged, confirm, final-seen and bench tiers,
# the calibration record, and at the end the pack directory RESULTS/MODEL/pack. Every target is written under a
# temporary name and renamed when complete, so a file that exists is a file that finished.
#
# SHARD must be a multiple of 32, the number of examples s1-features hands the model together
# (BLOCK in tools/s1-features.c): the batches then match those of one unsharded extraction, and
# the merged features are identical to it. Keep DATA and SHARD fixed for the life of a RESULTS
# directory: shards made under other settings are not detected.

MODEL     ?= qwen3-4b-q8
QUICK     ?= 0
MODEL_DIR ?=
FORMATS   ?= general stance
Q          = $(if $(filter 1,$(QUICK)),-quick,)
DATA      ?= data/tiers$(Q)
RESULTS   ?= results$(Q)
SHARD     ?= $(if $(filter 1,$(QUICK)),64,512)
BUILD     ?= build/cli
PY         = uv run --group pipeline python
OUT        = $(RESULTS)/$(MODEL)
VARS       = MODEL=$(MODEL) QUICK=$(QUICK) MODEL_DIR=$(MODEL_DIR) DATA=$(DATA) RESULTS=$(RESULTS) SHARD=$(SHARD) BUILD=$(BUILD)
TOOLS      = $(BUILD)/s1-features $(BUILD)/s1-train $(BUILD)/s1-eval $(BUILD)/s1-selftest
# Example sets per format, in extraction order; a set whose file the tiers do not have is skipped.
# A QUICK run leaves out the fresh final tiers (see QUICK above).
FRESH      = final final-flagged confirm
SETS       = fitdev $(if $(filter 1,$(QUICK)),,$(FRESH)) final-seen bench
# Evaluation tiers: name, example set, split (a tier whose example set is not in SETS is skipped).
EVAL_TIERS = test:fitdev:test dev:fitdev:heldout final:final:heldout final-flagged:final-flagged:heldout \
             confirm:confirm:heldout final-seen:final-seen:heldout bench:bench:heldout
# The confirm tier's features as the confirmation extracted them (one unsharded s1-features run per
# pack and format, same engine settings); make calibrate links them into RESULTS.
CONFIRM_FROM ?= results-confirm

ifneq ($(shell echo $$(( $(SHARD) % 32 ))),0)
$(error SHARD=$(SHARD) must be a multiple of 32 (see the top of this file))
endif

# Self-test failures accepted per model, recorded in its selftest.txt.
ACCEPT_FAIL ?=

.PHONY: pack run stop status data check verify-data licences fetch tools install-pack features format figures \
        calibrate compare-score compare-run
.DELETE_ON_ERROR:

# ---- control -------------------------------------------------------------------------------

run:
	@mkdir -p $(OUT)
	@if [ -f $(OUT)/run.pid ] && kill -0 `cat $(OUT)/run.pid` 2>/dev/null; then \
	    echo "already running (pid `cat $(OUT)/run.pid`); make status, or make stop"; exit 1; fi
	@echo "=== run started `date '+%Y-%m-%d %H:%M'`: $(MODEL)" >> $(OUT)/run.log
	@perl -e 'setpgrp(0, 0); exec @ARGV' nohup caffeinate -i $(MAKE) pack $(VARS) \
	    >> $(OUT)/run.log 2>&1 < /dev/null & echo $$! > $(OUT)/run.pid
	@echo "running in the background (pid `cat $(OUT)/run.pid`), log in $(OUT)/run.log"

stop:
	@if [ -f $(OUT)/run.pid ] && kill -0 `cat $(OUT)/run.pid` 2>/dev/null; then \
	    kill -INT -- -`cat $(OUT)/run.pid`; \
	    echo "=== run stopped `date '+%Y-%m-%d %H:%M'`" >> $(OUT)/run.log; echo "stopped; make run resumes"; \
	else echo "not running"; fi
	@rm -f $(OUT)/run.pid

status:
	@if [ -f $(OUT)/run.pid ] && kill -0 `cat $(OUT)/run.pid` 2>/dev/null; then \
	    echo "running (pid `cat $(OUT)/run.pid`)"; else echo "not running"; fi
	@for f in $(FORMATS); do for s in $(SETS); do \
	    [ -f $(DATA)/$$f/$$s.jsonl ] || continue; \
	    n=`cat $(DATA)/$$f/$$s.jsonl 2>/dev/null | wc -l | tr -d ' '`; \
	    if [ -f $(OUT)/$$f/$$s/features.feat ]; then echo "$$f/$$s: extracted"; \
	    else echo "$$f/$$s: `ls $(OUT)/$$f/$$s/shards/*.feat 2>/dev/null | wc -l | tr -d ' '` of $$(( (n + $(SHARD) - 1) / $(SHARD) )) shards"; fi; \
	done; [ -f $(OUT)/$$f/record.json ] && echo "$$f: record written"; done; true
	@[ -f $(OUT)/pack/pack.json ] && echo "pack: $(OUT)/pack" || true
	@[ -f $(OUT)/run.log ] && { echo "--- $(OUT)/run.log"; tail -5 $(OUT)/run.log; } || true

# ---- tools, raw data, tiers --------------------------------------------------------------

tools: $(TOOLS)
$(TOOLS):
	cmake -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=Release > /dev/null
	cmake --build $(BUILD) -j --target s1-features s1-train s1-eval s1-selftest

# The raw files that do not come from the Hugging Face Hub (data/registry.yaml, `raw`); the
# Hub sources are downloaded by the tier builder at their pinned revisions.
FETCHED = data/raw/fetched.stamp
fetch:
	$(PY) scripts/fetch_raw.py
	@date > $(FETCHED)
$(FETCHED):
	$(PY) scripts/fetch_raw.py
	@date > $@

# The tiers are built once and never remade behind a run's back (order-only prerequisite):
# delete DATA to rebuild them.
data: $(DATA)/manifest.json
$(DATA)/manifest.json: | $(FETCHED)
	$(PY) scripts/prep_tiers.py $(DATA).tmp $(if $(filter 1,$(QUICK)),--quick,) > $(DATA).log 2>&1 || \
	    { tail -20 $(DATA).log; exit 1; }
	rm -rf $(DATA) && mv $(DATA).tmp $(DATA)

check: $(DATA)/manifest.json
	$(PY) scripts/check_contamination.py $(DATA)

# The tiers must reproduce byte for byte: data/tiers.sha256 (data/tiers-quick.sha256 with
# QUICK=1) holds the SHA-256 of the files built for the released heads.
verify-data: $(DATA)/manifest.json
	cd $(DATA) && shasum -a 256 -c ../$(notdir $(DATA)).sha256

licences:
	$(PY) scripts/verify_licences.py

# ---- one pack ----------------------------------------------------------------------------

# The model file (verified against the pack's SHA-256), template and engine options, as make
# variables for the sub-makes below. Rewritten at the start of every run (not a rule, so that
# other targets never resolve or download the model); pack_info.py fails if the model, template
# or engine options differ from the previous run's, since finished shards used those.
# The full-size tiers must equal data/tiers.sha256 before any GPU time is spent on them.
VERIFY = $(if $(and $(filter 0,$(QUICK)),$(filter data/tiers,$(DATA))),1,)
pack: tools $(DATA)/manifest.json
	@mkdir -p $(OUT)
	@$(PY) scripts/pack_info.py $(MODEL) "$(MODEL_DIR)" $(OUT)/pack-info.mk > $(OUT)/pack-info.tmp && \
	    mv $(OUT)/pack-info.tmp $(OUT)/pack-info.mk
	@cat $(OUT)/pack-info.mk
	@if [ -n "$(VERIFY)" ]; then (cd data/tiers && shasum -a 256 -c ../tiers.sha256) > $(OUT)/verify-data.txt || \
	    { cat $(OUT)/verify-data.txt; echo "the tiers differ from data/tiers.sha256"; exit 1; }; fi
	$(PY) scripts/check_contamination.py $(DATA) > $(OUT)/contamination.txt || { cat $(OUT)/contamination.txt; exit 1; }
	@tail -1 $(OUT)/contamination.txt
	$(MAKE) --no-print-directory $(OUT)/selftest.txt $(VARS)
	@for f in $(FORMATS); do \
	    for s in $(SETS); do [ -f $(DATA)/$$f/$$s.jsonl ] || continue; \
	        $(MAKE) --no-print-directory features $(VARS) FMT=$$f SET=$$s || exit 1; done; \
	    $(MAKE) --no-print-directory format $(VARS) FMT=$$f || exit 1; \
	done
	$(MAKE) --no-print-directory $(OUT)/pack/pack.json $(VARS)

-include $(wildcard $(OUT)/pack-info.mk)
ROT_FLAG = $(if $(filter 1,$(ENGINE_ROTATIONS)),--rotations,)
CF_FLAG  = $(if $(filter 1,$(ENGINE_CONTENT_FREE)),--content-free,)

$(OUT)/selftest.txt:
	@mkdir -p $(@D)
	@echo "=== gate: s1-selftest on $(MODEL_PATH)"
	@if $(BUILD)/s1-selftest --model $(MODEL_PATH) --template $(TEMPLATE) > $@.tmp 2> $(OUT)/selftest.log; then cat $@.tmp; \
	else cat $@.tmp; failed=`awk '$$1 == "FAIL" && $$2 ~ /^T[0-9]+$$/ { printf "%s ", $$2 }' $@.tmp`; \
	    [ -n "$$failed" ] || { echo "self-test failed without naming a check"; exit 1; }; \
	    for t in $$failed; do case " $(ACCEPT_FAIL) " in *" $$t "*) ;; *) echo "excluded: $$t failed"; exit 1;; esac; done; \
	    echo "accepted with known failures: $$failed(ACCEPT_FAIL)" | tee -a $@.tmp; fi
	@mv $@.tmp $@

# ---- extraction of one example set: make features FMT=general SET=fitdev -----------------

SET_DIR     = $(OUT)/$(FMT)/$(SET)
EXAMPLES    = $(DATA)/$(FMT)/$(SET).jsonl
N_EXAMPLES := $(shell cat $(EXAMPLES) 2>/dev/null | wc -l | tr -d ' ')
N_SHARDS   := $(shell echo $$(( ($(or $(N_EXAMPLES),0) + $(SHARD) - 1) / $(SHARD) )))
SHARD_FEATS = $(shell i=0; while [ $$i -lt $(N_SHARDS) ]; do printf '$(SET_DIR)/shards/shard-%04d.feat ' $$i; i=$$((i + 1)); done)

features: $(SET_DIR)/features.feat

# Shard N holds examples N*SHARD+1 to (N+1)*SHARD; the rotations of one example are made inside
# s1-features, so they never cross a shard.
$(SET_DIR)/shards/%.feat: $(EXAMPLES)
	@mkdir -p $(@D)
	@i=`echo $* | sed 's/^shard-0*//'`; i=$${i:-0}; \
	sed -n "$$((i * $(SHARD) + 1)),$$(((i + 1) * $(SHARD)))p" $< > $(@D)/$*.jsonl
	$(BUILD)/s1-features --model $(MODEL_PATH) --template $(TEMPLATE) --examples $(@D)/$*.jsonl \
	    --out $@.tmp $(ROT_FLAG) --max-rotations $(ENGINE_MAX_ROTATIONS) $(CF_FLAG) 2> $(@D)/$*.log
	@grep "^s1-features: .* records, " $(@D)/$*.log | tail -1
	@mv $@.tmp.names.tsv $@.names.tsv && mv $@.tmp $@ && rm $(@D)/$*.jsonl

$(SET_DIR)/features.feat: $(SHARD_FEATS)
	sort -u $(SHARD_FEATS:%=%.names.tsv) > $@.names.tsv
	$(PY) scripts/merge_features.py $@ $^

# ---- head, evaluation and record of one format: make format FMT=general ------------------

F_DIR = $(OUT)/$(FMT)

format: $(F_DIR)/record.json

# s1-train is told the engine settings, checks the features against them and records them in
# h2.bin.json; the head is then refused when served under other settings. check_heads.py fails
# the build if any fitted question type is input-independent or does not beat the raw readout
# on validation without an explicit fallback to the identity (see docs/methods.md).
$(F_DIR)/h2.bin: $(F_DIR)/fitdev/features.feat
	$(BUILD)/s1-train --features $< --head h2 --out $@.tmp $(ROT_FLAG) \
	    --max-rotations $(ENGINE_MAX_ROTATIONS) $(CF_FLAG) > $(F_DIR)/train-h2.log && \
	    $(PY) scripts/check_heads.py $@.tmp --engine $(ENGINE_ROTATIONS) $(ENGINE_MAX_ROTATIONS) \
	        $(ENGINE_CONTENT_FREE) && \
	    mv $@.tmp.json $@.json && mv $@.tmp $@
	@grep -E "^[a-z]+:|^  H[12]:|^  [a-z]+: " $(F_DIR)/train-h2.log || true

# The second calibration option: one temperature per question type, fitted on the same training
# split (log loss of the rotation-averaged raw readout; s1-train --head temperature) and applied to
# the probabilities averaged over the option orders. Checked like the head, with the same engine
# settings recorded in temperature.bin.json.
$(F_DIR)/temperature.bin: $(F_DIR)/fitdev/features.feat
	$(BUILD)/s1-train --features $< --head temperature --out $@.tmp $(ROT_FLAG) \
	    --max-rotations $(ENGINE_MAX_ROTATIONS) $(CF_FLAG) > $(F_DIR)/train-temperature.log && \
	    $(PY) scripts/check_heads.py $@.tmp --engine $(ENGINE_ROTATIONS) $(ENGINE_MAX_ROTATIONS) \
	        $(ENGINE_CONTENT_FREE) && \
	    mv $@.tmp.json $@.json && mv $@.tmp $@
	@cat $(F_DIR)/train-temperature.log

# Conditions: raw (no head), h2 and temperature, all with the pack's engine settings (rotations,
# content-free), so that the numbers describe what judgly.Engine.load(pack, calibration=...)
# answers. Tiers: test (in-distribution, fit tier), dev, final (the fresh held-out families),
# final-flagged (fresh families with a recorded caveat, reported beside final), confirm (the
# untouched tier of the pre-registered confirmation), final-seen (an earlier held-out tier, its
# families seen during development; reported separately) and bench (external benchmarks); a tier
# whose example set the format does not have is skipped.
EVAL_SETS = $(filter-out fitdev,$(foreach s,$(SETS),$(if $(wildcard $(DATA)/$(FMT)/$(s).jsonl),$(s))))
$(F_DIR)/eval.done: $(F_DIR)/h2.bin $(F_DIR)/temperature.bin $(EVAL_SETS:%=$(F_DIR)/%/features.feat)
	@set -e; for tier in $(EVAL_TIERS); do \
	    t=$${tier%%:*}; rest=$${tier#*:}; set_=$${rest%%:*}; split=$${rest#*:}; \
	    case " $(EVAL_SETS) fitdev " in *" $$set_ "*) ;; *) continue;; esac; \
	    [ -f $(F_DIR)/$$set_/features.feat ] || continue; \
	    $(BUILD)/s1-eval --features $(F_DIR)/$$set_/features.feat --split $$split $(ROT_FLAG) $(CF_FLAG) \
	        --json $(F_DIR)/raw-$$t.json --dump-items $(F_DIR)/items-raw-$$t.tsv > $(F_DIR)/raw-$$t.txt; \
	    for cond in h2 temperature; do \
	        $(BUILD)/s1-eval --features $(F_DIR)/$$set_/features.feat --split $$split --head $(F_DIR)/$$cond.bin \
	            $(ROT_FLAG) $(CF_FLAG) --json $(F_DIR)/$$cond-$$t.json --dump-items $(F_DIR)/items-$$cond-$$t.tsv \
	            > $(F_DIR)/$$cond-$$t.txt; \
	    done; \
	done
	$(PY) scripts/check_report.py $(F_DIR)/raw-*.json $(F_DIR)/h2-*.json $(F_DIR)/temperature-*.json
	date > $@

$(F_DIR)/record.json: $(F_DIR)/eval.done
	$(PY) scripts/calibration_record.py $(F_DIR) $(DATA) $(MODEL) $(FMT) $(QUICK) > $(F_DIR)/tables.txt
	@head -12 $(F_DIR)/tables.txt

$(OUT)/pack/pack.json: $(FORMATS:%=$(OUT)/%/record.json)
	$(PY) scripts/build_pack.py $(MODEL) $(OUT)/pack $(OUT) $(FORMATS)

# ---- calibration options on CPU, from features already extracted --------------------------

# make calibrate MODEL=... fits the temperature, evaluates raw, h2 and temperature on every tier and
# writes the records and the pack, from the feature files a finished run left (and the confirm
# tier's from CONFIRM_FROM, linked into RESULTS). It never runs the model: every feature file is
# passed to make as old (-o), so nothing that makes one is run, and a missing one stops it. H2 is
# refitted only if h2.bin is missing.
CAL_FEATS = $(foreach f,$(FORMATS),$(foreach s,$(SETS),$(if $(wildcard $(DATA)/$(f)/$(s).jsonl),$(OUT)/$(f)/$(s)/features.feat)))
calibrate:
	@[ -f $(OUT)/pack-info.mk ] || { echo "no $(OUT)/pack-info.mk: make pack has not run for $(MODEL)"; exit 1; }
	@for f in $(FORMATS); do c=$(CONFIRM_FROM)/$(MODEL)/$$f/confirm.feat; d=$(OUT)/$$f/confirm; \
	    if [ -f $$c ] && [ ! -e $$d/features.feat ]; then mkdir -p $$d && \
	        ln -s $(CURDIR)/$$c $$d/features.feat && ln -s $(CURDIR)/$$c.names.tsv $$d/features.feat.names.tsv && \
	        echo "linked $$d/features.feat -> $$c"; fi; done
	@missing=; for x in $(CAL_FEATS); do [ -f $$x ] || missing="$$missing $$x"; done; \
	    [ -z "$$missing" ] || { echo "features not extracted (make pack extracts them, on the GPU):$$missing"; exit 1; }
	@for f in $(FORMATS); do \
	    $(MAKE) --no-print-directory format $(VARS) FMT=$$f $(CAL_FEATS:%=-o %) || exit 1; done
	$(MAKE) --no-print-directory $(OUT)/pack/pack.json $(VARS) $(CAL_FEATS:%=-o %)

install-pack:
	@[ "$(QUICK)" != 1 ] || { echo "a QUICK pack is a smoke test and is never installed"; exit 1; }
	@[ -f $(OUT)/pack/pack.json ] || { echo "no finished pack in $(OUT)/pack; make pack MODEL=$(MODEL) first"; exit 1; }
	cp -R $(OUT)/pack/heads $(OUT)/pack/calibration $(OUT)/pack/pack.json src/judgly/packs/$(MODEL)/

# ---- figures -----------------------------------------------------------------------------

# From the committed snapshot only (docs/results, made by scripts/snapshot_results.py); every
# plotted number is checked against record.json. Reruns give byte-identical files.
figures:
	uv run --group figures python scripts/make_figures.py

# ---- external comparison ---------------------------------------------------------------------

# docs/results/external-comparison: judgly and the Ollama decision models on the same test items.
# compare-score rebuilds every result from the committed answers with the frozen scorer (a
# hash-checked copy run in a scratch workspace; needs the tiers from make data) and checks that
# each equals the committed one, byte for byte. compare-run asks the models again (Ollama >= 0.35.0
# with nimble:9b, tev1:4b and tev1:0.8b pulled; tags can move, so the model IDs are compared with
# the recorded ones) and writes answers and results to COMPARE_OUT, never into the record.
COMPARE_OUT ?= results-compare
COMPARE     = uv run --no-project --with numpy python docs/results/external-comparison/reproduce.py

compare-score:
	$(COMPARE) score

compare-run:
	$(COMPARE) run --out $(COMPARE_OUT)
