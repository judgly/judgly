# Environment of the external comparison

Times are local (UTC+8).

## Hardware and system

| | |
|---|---|
| machine | Apple M3 Max, 64 GB unified memory |
| OS | macOS 27.0 (`macOS-27.0-arm64-arm-64bit-Mach-O`, as recorded in `final/timing.json`); macOS 27.0.1 for `clef:27b`'s single-request timing (2026-10-04) |
| Ollama | 0.35.0 (`ollama --version`: `ollama version is 0.35.0`), serving `http://localhost:11434/v1/systemone`, for `nimble:9b`, `tev1:4b` and `tev1:0.8b`; 0.35.1 for `clef:27b`, which requires it (below) |
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
| `clef:27b` | `2bb11a61d1fb` | `2bb11a61d1fb5d7a51f136ad9f569f970727377cf7b96037a06285d81fa25b73` | `clef-Q4_K_M.gguf` | `6c02216a0055c1e1e92994d240e37c440f1c09b30281f538f4a38ebab0743da9` | 17,059,634,656 |

Context lengths as Ollama enforces them: Tev1 (both sizes) accepts prompts of 1 to 2,050
tokens and refuses longer ones with HTTP 400 ("input is never truncated"); Nimble's context is
8,194 tokens; Clef's is 16,384 (`num_ctx` in its manifest's parameters).

`clef:27b` was added on 2026-10-03 ([NOTES.md](NOTES.md)). It is Cloudflare's Clef
([model card](https://huggingface.co/Cloudflare/clef),
[announcement](https://blog.cloudflare.com/clef-decision-models/), 2026-10-01), post-trained from
Qwen/Qwen3.8-27B, as packaged by Ollama. Ollama reports it as architecture `qwen35`, 27.0B
parameters, Q4_K_M, base model "Qwen3.8 27B" (GGUF metadata), capabilities `decision` and
`vision`, licence Apache 2.0, and Ollama 0.35.1 as its minimum version. Cloudflare's release is
safetensors with a separate joint schema head and a 64K context; whether Ollama's package scores
exactly as Cloudflare's release does was not checked. Its
manifest also holds a vision projector (`sha256:f0e2930e301586f0932262c7d232c74e385d7b4281170111c36a686e16e59ba8`,
927,607,360 bytes), which text-only requests do not use. Its tag's ID on the registry on
2026-10-03 was the same, `2bb11a61d1fb`.

Ollama 0.35.1 serves `/v1/systemone` only to models that declare the `decision` capability.
`nimble:9b`, `tev1:4b` and `tev1:0.8b`, as pulled under 0.35.0 (the IDs above), do not
(`ollama show`: tools, thinking, completion), and 0.35.1 refuses every request to them with
HTTP 400 ("does not support decision"; 200 of 200 `tev1:0.8b` requests in a spot check on
2026-10-03). Their tags have since moved on the registry (on 2026-10-03: `tev1:0.8b`
`8d11b3146b7f`, `tev1:4b` `9b5bb969e46c`, `nimble:9b` `9b953de7a533`), so they were not pulled
again. Their answers are those of 0.35.0; Clef's are those of 0.35.1. Whether the two versions
serve the same weights identically was not checked. The tags' newer versions (the IDs above) were
pulled under 0.35.1 on 2026-10-04: they declare `decision` and are served, but are other weights
than the record's and were not used for any result ([NOTES.md](NOTES.md)).

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
| `clef:27b` added; decision recorded in `NOTES.md` | 2026-10-03 16:02 |
| first `clef:27b` run, lost (below) | 2026-10-03 16:02 to about 20:00 |
| `clef:27b` answered every test item (`run-clef.log`) | 2026-10-03 20:29 to 2026-10-04 05:03 |
| `clef:27b` answered the control's training sample (`run-clef.log`) | 2026-10-04 05:03 to 05:52 |
| `clef:27b` single-request timing, added to `final/timing.json` | 2026-10-04 |
| rescored with four models (`final/`, `calibrated/result.*`) | 2026-10-04 |

The first `clef:27b` run was started at 16:02 on 2026-10-03 in a temporary directory, which was
deleted when the machine restarted for a macOS update; its answers were lost. Of them, only the
first 43 general confirm answers were looked at (for errors and speed, and one response for its
format). The run was started again from the first item at 20:29, in a directory that survives a
restart; every recorded answer comes from that second run.

## Commands

The run (`chain.sh`, one model after the other, each through the frozen runner; its output is
`run.log`):

```bash
uv run --no-project python run_external.py /Users/timo/code/judgly tev1:4b
uv run --no-project python run_external.py /Users/timo/code/judgly tev1:0.8b
uv run --no-project python run_external.py /Users/timo/code/judgly nimble:9b
```

Clef (`run-clef.sh`, run from a copy of the frozen runners, checked against their SHA-256; its
output is `run-clef.log`): the test items, then the control's training sample:

```bash
uv run --no-project python run_external.py /Users/timo/code/judgly clef:27b
uv run --no-project python calibrated/run_calibration.py /Users/timo/code/judgly clef:27b
```

The scoring, once per model and once with all models (in this order); the printed output is
`final/score-<name>.txt` and the scorer's `result-external.json` is `final/result-<name>.json`,
`<name>` being `nimble_9b`, `tev1_4b`, `tev1_0.8b`, `clef_27b` or `all`. The commands as first
run, with three models:

```bash
uv run --no-project --with numpy python score_external.py /Users/timo/code/judgly nimble:9b
uv run --no-project --with numpy python score_external.py /Users/timo/code/judgly tev1:4b
uv run --no-project --with numpy python score_external.py /Users/timo/code/judgly tev1:0.8b
uv run --no-project --with numpy python score_external.py /Users/timo/code/judgly nimble:9b tev1:4b tev1:0.8b
```

These are the commands as run, with numpy unpinned. With Clef (2026-10-04), `make compare-score`
(which pins `--python 3.13 --with numpy==2.5.3`) ran the scorer once per model and once with all
four (`nimble:9b tev1:4b tev1:0.8b clef:27b`); the three earlier per-model files were rebuilt byte
for byte, `result-clef_27b.json` and `score-clef_27b.txt` are new, and `result-all.json` and
`score-all.txt` were replaced by the four-model run. Every point value, interval and difference of
the earlier systems in `result-all.json` is unchanged (Clef answered every item, so the items
every model answered are the same).

The timings:

```bash
JUDGLY_MODEL_DIR=/Users/timo/judgly-models uv run python time_single.py /Users/timo/code/judgly timing.json
```

Clef's, on 2026-10-04 (Ollama 0.35.1, which refuses the other Ollama models; merged into
`final/timing.json` with its Ollama version and platform):

```bash
uv run python time_single.py /Users/timo/code/judgly timing-clef.json --only clef:27b
```

The answers were compressed for the record with `gzip -n -9` (no name or time stamp in the
header), one file per `answers/<model>/<format>-<tier>.jsonl`.
