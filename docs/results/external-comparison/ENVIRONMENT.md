# Environment of the external comparison

Times are local (UTC+8).

## Hardware and system

| | |
|---|---|
| machine | Apple M3 Max, 64 GB unified memory |
| OS | macOS 27.0 (`macOS-27.0-arm64-arm-64bit-Mach-O`, as recorded in `final/timing.json`) |
| Ollama | 0.35.0 (`ollama --version`: `ollama version is 0.35.0`), serving `http://localhost:11434/v1/systemone` |
| judgly | branch `calibration-options`, commit `3e76204` (packs `gemma4-12b-q8` and `qwen3-4b-q8`) |
| Python | 3.13; the scorer ran with `uv run --no-project --with numpy` (numpy unpinned; the version used at 21:58 on 2026-09-29 was not recorded. The rescoring on 2026-09-30 used numpy 2.5.3 and found every file identical; `make compare-score` pins numpy 2.5.3 and Python 3.13, since another numpy may draw other bootstrap resamples) |

## External models

Pulled from the Ollama library by their default tags, run with their default settings. The ID is
the one `ollama list` prints: the first 12 hex digits of the SHA-256 of the tag's manifest.

| tag | ID | manifest SHA-256 | model layer (GGUF) | model layer SHA-256 | bytes |
|---|---|---|---|---|---|
| `nimble:9b` | `aa4a79f08ae0` | `aa4a79f08ae089919b82cab5612f073868307306c4213f724588e38a4a764aca` | `Bespoke-Nimble-9B-merged-current-Q8_0.gguf` | `bbf1d6fc03bb0ed24d88f4c214ed7b5d1768aeb43d5cf433fb69eff0c8578013` | 9,527,501,312 |
| `tev1:4b` | `d18e9174f4db` | `d18e9174f4dbc07f534f75962786fb71d9235e8afe627dfd4625861698d76648` | `Tev1-4B-Q8_0.gguf` | `35f9281a3df58b566b24091572467001906a5a6aac879fe8005c4db19c8d4a2e` | 4,482,403,072 |
| `tev1:0.8b` | `c0099a86fcbd` | `c0099a86fcbd81bc5876a0d1f94998d2b038f7f1b5f7329a3dba43a36903c652` | `Tev1-0.8B-Q8_0.gguf` | `fa9732e3924db99f614181a7a28384a0891ae17f478db228a47c989462b2405a` | 811,843,424 |

Context lengths as Ollama enforces them: Tev1 (both sizes) accepts prompts of 1 to 2,050
tokens and refuses longer ones with HTTP 400 ("input is never truncated"); Nimble's context is
8,194 tokens.

## Dates

| what | when |
|---|---|
| protocol, runner and scorer frozen (`PROTOCOL.sha256`) | 2026-09-29 16:47 |
| decision recorded in `NOTES.md` | 2026-09-29 16:54 |
| `tev1:4b` answered every item | 2026-09-29 16:47 to 18:29 |
| `tev1:0.8b` answered every item | 2026-09-29 18:29 to 18:56 |
| `nimble:9b` answered every item | 2026-09-29 18:56 to 21:57 |
| scored (`final/result-*.json`, `final/score-*.txt`) | 2026-09-29 21:58 |
| single-request timings (`final/timing.json`) | 2026-09-30, written 06:18 |

## Commands

The run (`chain.sh`, one model after the other, each through the frozen runner; its output is
`run.log`):

```bash
uv run --no-project python run_external.py /Users/timo/code/judgly tev1:4b
uv run --no-project python run_external.py /Users/timo/code/judgly tev1:0.8b
uv run --no-project python run_external.py /Users/timo/code/judgly nimble:9b
```

The scoring, once per model and once with all three (in this order); the printed output is
`final/score-<name>.txt` and the scorer's `result-external.json` is `final/result-<name>.json`,
`<name>` being `nimble_9b`, `tev1_4b`, `tev1_0.8b` or `all`:

```bash
uv run --no-project --with numpy python score_external.py /Users/timo/code/judgly nimble:9b
uv run --no-project --with numpy python score_external.py /Users/timo/code/judgly tev1:4b
uv run --no-project --with numpy python score_external.py /Users/timo/code/judgly tev1:0.8b
uv run --no-project --with numpy python score_external.py /Users/timo/code/judgly nimble:9b tev1:4b tev1:0.8b
```

These are the commands as run, with numpy unpinned. To rescore, use `make compare-score`, which
pins `--python 3.13 --with numpy==2.5.3`.

The timings:

```bash
JUDGLY_MODEL_DIR=/Users/timo/judgly-models uv run python time_single.py /Users/timo/code/judgly timing.json
```

The answers were compressed for the record with `gzip -n -9` (no name or time stamp in the
header), one file per `answers/<model>/<format>-<tier>.jsonl`.
