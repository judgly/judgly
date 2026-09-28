# Contributing

judgly is a small hobby project, and issues and pull requests are welcome.

## Building from source

```
git clone --recurse-submodules https://github.com/judgly/judgly
cd judgly
uv sync                       # builds libjudgly (llama.cpp and ggml included) and installs judgly
uv run python -c "import judgly; print(judgly.native_version())"
```

The command-line tools (`s1-run`, `s1-features`, `s1-train`, `s1-eval`, `s1-compare`,
`s1-selftest`, `s1-probe`) are built with plain CMake:

```
cmake -S . -B build/cli && cmake --build build/cli -j
```

## Rules for changes

- C code in `csrc/` and `tools/` compiles without warnings under
  `-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror`.
- Nothing under `third_party/` is edited. llama.cpp is a git submodule pinned to one commit;
  moving the pin is its own change and needs the licence check below to pass.
- Every vendored component needs its licence text in `LICENSES/` and an entry in `NOTICE`:
  `uv run python scripts/check_licenses.py` must exit 0.

## macOS toolchain note

On some machines (macOS 27 with Xcode 26.1) the linker cannot link against the
macOS 27 SDK. When `/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk` exists and no
SDK is set (`CMAKE_OSX_SYSROOT` or `SDKROOT`), CMakeLists.txt uses it. Set `SDKROOT` to
override.

## Tests

```
uv run pytest -q                                   # no model needed
JUDGLY_MODEL_DIR=/path/to/models uv run pytest -q -m model   # needs Qwen3-4B locally
uv run --group pipeline pytest -q tests/test_pipeline.py     # the data pipeline and checker
```

The model tests compare the engine's output with golden values in `tests/fixtures/` (tolerance
1e-6) and are skipped when the Qwen3-4B model file is not found (`JUDGLY_MODEL_DIR` or
`JUDGLY_TEST_MODEL`).

## Documentation and examples

- Keep the tone plain and modest: judgly is a hobby project built on other people's work.
  Credit sources, and never present a number without its source and interval.
- Do not type numbers into the docs by hand where a generated source exists. The data-source
  tables in `docs/reproduce.md` are written by `docs/tools/sources_table.py` from
  `data/registry.yaml`.
- Results go into the marked slots (`<!-- RESULTS:... -->`) from a finished pipeline run.
- Every example in `examples/` must run against a pack. An example that only makes sense with
  shipped heads carries the line `# judgly-example: needs-full-pack`.
- Before sending a change to the docs or examples, run:

  ```
  JUDGLY_MODEL_DIR=/path/to/models uv run --group pipeline python docs/tools/check_docs.py
  ```

  It checks every relative link, runs every example and the README quickstart against the QUICK
  smoke pack (`JUDGLY_PACK` to use another), and checks the citation file, the assets and the
  source tables.

## Data and heads

New data sources go into `data/registry.yaml` with a pinned revision, the licence exactly as
the dataset card states it, and a citation. `make licences` must pass. A source that is not
permissive is only used for evaluation. The rules are at the top of the registry and in
[docs/methods.md](docs/methods.md#data-and-tiers).
