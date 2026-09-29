# Model packs

A pack is a directory that ties together everything that must match for a head to be valid:

```text
my-pack/
  pack.json          which model file, which template, which heads, which engine settings
  template.tpl       the prompt template (the model's chat markers around judgly's text)
  heads/h2.bin, heads/temperature.bin                 the general calibration options ("*")
  heads/h2-stance.bin, heads/temperature-stance.bin   the stance calibration options
  heads/*.bin.json   each head's trainer sidecar: features, engine settings, per-type fit
  calibration/       the calibration record of each format: data, metrics, intervals, both options
```

The model file itself is not in the pack. `pack.json` names its Hugging Face repository,
revision, file name, size and SHA-256, and judgly downloads or locates the file and checks it.

## Built-in packs

| pack | model | licence of the weights | status |
|---|---|---|---|
| `gemma4-12b-q8` | Google Gemma 4 12B instruction-tuned, Q8_0 GGUF (ggml-org) | Apache-2.0 (Google) | default; heads shipped (general and stance), each as H2 and as a per-type temperature; used by default: H2 for general, the temperature for stance ([calibration-options.md](calibration-options.md)); the H2 stance head meets its final-tier bar narrowly and misses its dev-tier bar, see [model card](model-cards/gemma4-12b-q8.md) |
| `qwen3-4b-q8` | Qwen3-4B-Instruct-2507, Q8_0 GGUF (Unsloth) | Apache-2.0 (Qwen team) | smaller and faster; heads shipped (general and stance), each as H2 and as a per-type temperature; used by default: the temperature for both ([calibration-options.md](calibration-options.md)); the H2 stance head misses its final-tier bar narrowly and its dev-tier bar, see [model card](model-cards/qwen3-4b-q8.md) |

<!-- RESULTS:PACK-TABLE -->

## pack.json

```json
{
 "schema": 1,
 "name": "gemma4-12b-q8",
 "description": "Google Gemma 4 12B instruction-tuned, 8-bit GGUF. The default model.",
 "model": {
  "repo_id": "ggml-org/gemma-4-12B-it-GGUF",
  "revision": "e3e681731089efaa3f0917336944ac64752db8ba",
  "filename": "gemma-4-12B-it-Q8_0.gguf",
  "sha256": "abfc3044b93795f6a446fc1a04468ac2d5fc5e9c92e237f09a5b3384c5a433e4",
  "size": 12669646976
 },
 "template": "template.tpl",
 "engine": {"rotations": true, "content_free": false, "max_rotations": 4},
 "heads": {
  "*":      {"default": "h2",
             "options": {"h2": {"file": "heads/h2.bin"},
                         "temperature": {"file": "heads/temperature.bin"}}},
  "stance": {"default": "temperature",
             "options": {"h2": {"file": "heads/h2-stance.bin"},
                         "temperature": {"file": "heads/temperature-stance.bin"}}}
 }
}
```

This is the shape of the built-in `pack.json` (schema 2), shortened; its `engine` block is
copied from it. A finished pack also records, per option, its SHA-256, licence, calibration
record and final-tier and confirm-tier metrics, and per format why its default is what it is
(`default_basis`; see [calibration-options.md](calibration-options.md)). `engine` holds the
settings the heads were fitted under; `Engine.load` applies them. A pack of schema 1 (one head
per format, `"*": {"file": "heads/h2.bin"}`) still loads: its head is that format's only option
and its default.

## Adding a model

A new model needs a template, a pack entry and a pipeline run. You need a source checkout
([installation.md](installation.md#build-from-source)) and a few hours of GPU time.

1. **Choose a GGUF file.** Single-file GGUF, 8-bit or 16-bit if you can. judgly's self-tests
   compare the engine against llama.cpp's own logits and check that branching and batching do
   not change the answers (T3 and T4, tolerance 0.02 in probability). A file whose probabilities
   move more than that with batch shape fails the gate and cannot become a pack; check
   low-bit mixture-of-experts files with particular care.
2. **Write the template.** A template is a small JSON file with two strings: `open` (the
   system turn and the start of the user turn, ending where the state begins) and `close` (the
   end of the user turn and the start of the model's turn, ending in `Answer:`). The letter is
   read right after `close`. Copy `src/judgly/packs/qwen3-4b-q8/template.tpl` and replace the
   chat markers with your model's own. Turn reasoning off: the Gemma template does this with
   an empty thought channel.
3. **Add a pack entry.** Create `src/judgly/packs/<name>/pack.json` with the model's
   repository, pinned revision, file name, size and SHA-256, the template, the engine settings,
   and `heads` entries with `"status": "not yet available"` (schema 1 is enough). Until the
   head files exist, load the pack with `heads=False`.
4. **Run the self-tests.** They check prompt integrity, the letter readout against llama.cpp,
   branching against recomputation, isolation and more (see [methods.md](methods.md#engine-self-tests)):

   ```bash
   make tools
   build/cli/s1-selftest --model /models/<file>.gguf --template src/judgly/packs/<name>/template.tpl
   ```

   Do not continue past a failure unless you understand it.
5. **Smoke run, then the full run.**

   ```bash
   make pack QUICK=1 MODEL=<name> MODEL_DIR=/models      # minutes: tiny tiers, checks the plumbing
   make run MODEL=<name> MODEL_DIR=/models               # hours, in the background, resumable
   make status MODEL=<name>
   ```

6. **Read the results** in `results/<name>/{general,stance}/tables.md`. Look at held-out ECE
   and accuracy with their intervals, not only the point estimates.
7. **Choose the defaults.** A new pack uses H2 by default for every format
   (`DEFAULTS` in `scripts/build_pack.py`); make the temperature a format's default only on
   evidence from data that played no part in any choice, as for the built-in packs.
8. **Install the pack**: `make install-pack MODEL=<name>` copies the heads and calibration
   records into `src/judgly/packs/<name>/`.

[reproduce.md](reproduce.md) has the details of each step and where every file goes.

## Using your own heads with a built-in pack

A head fitted on your own data with the same model, template and engine settings can be
passed directly:

```python
Engine.load("gemma4-12b-q8", heads={"*": "heads/general.bin", "tickets": "heads/tickets-h1.bin"})
```

Questions with `format="tickets"` then use your head. [calibration.md](calibration.md#fitting-a-head-on-your-own-data)
shows how to fit one.
