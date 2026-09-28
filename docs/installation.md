# Installation

judgly runs on macOS with Apple Silicon (M1 or later). It uses llama.cpp's Metal backend on the
GPU. Linux, Windows and CUDA are not set up: the C code is portable, but nothing is built or
tested there yet.

## Requirements

- macOS 14 or later on Apple Silicon (M1 or later)
- Python 3.10 or later
- disk space and memory for the model you use:

| pack | model file | download | memory |
|---|---|---|---|
| `gemma4-12b-q8` (default) | `gemma-4-12B-it-Q8_0.gguf` from `ggml-org/gemma-4-12B-it-GGUF` | 12.7 GB | peak about 25 GB (22.9 GiB); a Mac with 32 GB or more |
| `qwen3-4b-q8` | `Qwen3-4B-Instruct-2507-Q8_0.gguf` from `unsloth/Qwen3-4B-Instruct-2507-GGUF` | 4.3 GB | peak about 9 to 10 GB; a Mac with 16 GB or more |

Peak memory was measured during batch runs with the default engine settings (the model plus a
context of 32,768 tokens shared by the text and the question branches). A single request on a
short text needs less.

## Install from PyPI

With [uv](https://docs.astral.sh/uv/):

```bash
uv add judgly            # in a uv project
uv pip install judgly    # into the active environment
```

With pip:

```bash
python -m pip install judgly
```

Check that the native library loads and sees the GPU:

```bash
python -c "import judgly, json; print(json.dumps(json.loads(judgly.native_version()), indent=1))"
```

The output lists the judgly version, the llama.cpp commit compiled in, and the backends. On a
Mac it should include `"MTL"` (Metal).

## The model file

The first `Engine.load(pack)` downloads the pack's model file from Hugging Face at a pinned
revision into the Hugging Face cache (`~/.cache/huggingface/hub`). judgly then checks the
file's size and SHA-256 against the pack. The SHA-256 is computed once and remembered in
`~/.cache/judgly/sha256.json`, keyed by path, size and modification time, so later loads are
fast. Set `JUDGLY_CACHE` to move that file.

To use a copy you already have, either pass its path:

```python
Engine.load("gemma4-12b-q8", model_path="/models/gemma-4-12B-it-Q8_0.gguf")
```

or point `JUDGLY_MODEL_DIR` at the directory that holds it:

```bash
export JUDGLY_MODEL_DIR=/models
```

The file must be byte-identical to the pinned one. A different quantisation or conversion
raises `ModelMismatch`, because the heads only fit the exact file they were fitted on.

The usual Hugging Face settings apply to the download, for example `HF_HOME` for the cache
location and `HF_TOKEN` if you need to authenticate.

## Build from source

You need the Xcode command-line tools, CMake 3.20 or later, and uv.

```bash
git clone --recurse-submodules https://github.com/judgly/judgly
cd judgly
uv sync                    # builds libjudgly (llama.cpp included) and installs judgly
uv run pytest -q           # tests that need no model
```

`uv sync` compiles llama.cpp, which takes a few minutes the first time. llama.cpp is a git
submodule pinned to one commit, and the build refuses a different checkout.

Tests that need a model are marked `model`. They find the Qwen3-4B file through
`JUDGLY_TEST_MODEL` (the file) or `JUDGLY_MODEL_DIR` (its folder) and are skipped without
either; the other variables they read are listed in `tests/conftest.py`:

```bash
JUDGLY_TEST_MODEL=/models/Qwen3-4B-Instruct-2507-Q8_0.gguf uv run pytest -q -m model
```

To build the command-line tools used by the pipeline (`s1-selftest`, `s1-features`,
`s1-train`, `s1-eval` and others):

```bash
cmake -S . -B build/cli -DCMAKE_BUILD_TYPE=Release
cmake --build build/cli -j
```

## Troubleshooting

**The build fails to link on a macOS beta.** With a macOS 27 beta and Xcode 26.1, the linker
can fail against the macOS 27 SDK. The build picks the MacOSX26.5 SDK from the
command-line tools when that SDK exists and neither `CMAKE_OSX_SYSROOT` nor `SDKROOT` is set.
To choose an SDK yourself:

```bash
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk uv sync
```

**`HeadsNotAvailable` on load.** The pack names heads whose files are missing (the built-in
packs ship theirs; a pack you built may not). Load with
`heads=False` for raw probabilities, or pass your own heads (see [calibration.md](calibration.md)).

**`ModelMismatch` on load.** The model file is not the pinned one. Delete it and let judgly
download it again, or check that `JUDGLY_MODEL_DIR` points at the right file.

**Load messages on stderr.** The engine prints a load line and error details to stderr even
with `verbose=False`. This is known and harmless. The first call into the native library in a
process (for example `judgly.native_version()` above) also compiles the Metal kernels, which took
about 15 seconds on an M3 Max and prints a few dozen `ggml_metal` initialisation lines.

**A request fails with a message about the context.** The text and the questions together
do not fit in the engine's context (`n_ctx` tokens). judgly checks this before running and refuses the
request, and the engine stays usable. Shorten the text, ask fewer questions per request, or set
`max_state_tokens` (see [usage.md](usage.md#engine-options)).

**Another program is using the GPU heavily.** On Apple Silicon, two programs running large
models on Metal at once can slow each other down badly. Run one at a time.
