# Usage

This page is the guide to the Python API. The examples in [../examples/](../examples/) are
complete scripts.

## Load an engine

```python
from judgly import Engine

engine = Engine.load("gemma4-12b-q8")          # built-in pack, with its default calibration
engine = Engine.load("gemma4-12b-q8", calibration="temperature")  # the per-type temperature
engine = Engine.load("gemma4-12b-q8", calibration="h2")           # the H2 heads
engine = Engine.load("gemma4-12b-q8", heads=False)            # raw probabilities (H0)
engine = Engine.load("path/to/pack")           # a pack directory
engine = Engine.load("gemma4-12b-q8", heads={"*": "my-h1.bin"})  # your own heads
```

`Engine.load` finds the pack, fetches or locates the model file, checks its SHA-256 and loads
the model, template and heads. The first load downloads the model file (12.7 GB for Gemma 4 12B)
and hashes it once; after that, loading Gemma 4 12B takes a few seconds. The first native call
in a process also compiles the Metal kernels (about 15 seconds).

**Calibration.** Each built-in pack ships two calibration options per format: the H2 head and
the per-type temperature (one temperature per question type, applied after the answers are
averaged over the option orders). `calibration="default"` (the default) uses the option the
pack names for each format: the temperature for Qwen3-4B (general and stance) and for Gemma 4
12B stance, H2 for Gemma 4 12B general, as a pre-registered comparison on untouched data
decided ([calibration-options.md](calibration-options.md)). `"h2"` or `"temperature"` uses that
option for every format, `"raw"` none (the same as `heads=False`). Any other value raises
`ValueError`. `engine.calibration` tells which option each format uses, for example
`{"*": "h2", "stance": "temperature"}`.

A `heads` mapping replaces the pack's heads entirely (and cannot be combined with a
`calibration` other than `"default"`). A question uses the entry named by its
`format`, or else the entry `"*"`; if the mapping has no `"*"` entry, questions without a
matching format get raw probabilities (`head` is `None` in the answer).

An `Engine` holds the model in memory until you close it. Use it as a context manager, or call
`engine.close()`:

```python
with Engine.load("gemma4-12b-q8") as engine:
    ...
```

`judgly.list_packs()` names the built-in packs.

## Ask questions

```python
from judgly import Binary, Choice, Score

d = engine.decide(state, {
    "topic":  Choice(instructions="What is the email mainly about?",
                     options={"delivery": "A late or missing delivery",
                              "defect": "A faulty product",
                              "other": "Something else"}),
    "refund": Binary(instructions="Does the customer ask for a refund?"),
    "anger":  Score(instructions="How upset is the customer, from 1 (calm) to 5 (furious)?",
                    levels=5),
})
d["topic"].probs      # {"delivery": 0.91, "defect": 0.03, "other": 0.06}
d["topic"].top        # "delivery"
d["refund"].p_true    # 0.97
d["anger"].probs      # [p(1), p(2), p(3), p(4), p(5)]
d["anger"].mean       # expected level
```

(The numbers above only show the shapes.)

- `state` is the text the questions are about. It is read once per request.
- Question ids (the dictionary keys) are yours; the answers come back under the same ids.
- For a `Choice`, option keys are what the answer reports and option texts are what the model
  reads. A list of strings uses each string as both.
- Questions can also be plain dictionaries of the same shape, for example
  `{"type": "bool", "instructions": "..."}`.

[question-formats.md](question-formats.md) covers writing good questions and the `format`
field.

## What comes back

`decide` returns a `Decision`. `d[qid]` (or `d.answers[qid]`) is the answer to one question.
Every answer also carries:

| field | meaning |
|---|---|
| `slot_mass` | how much probability the model put on the answer letters at all, averaged over option orders. Near 1 is healthy. A low value means the model wanted to say something else, and the answer is less reliable. |
| `rotation_spread` | the range of the top option's probability across the option orders asked. A large spread means the answer depended on the order of the options. With H2 it is measured on each order's calibrated probabilities; with the temperature (and with raw) on the raw per-order probabilities, before the temperature, so it is not on the same scale as the temperature's `probs`. |
| `n_rotations` | how many option orders were asked |
| `head`, `head_format` | the SHA-256 of the head file used and which heads entry it was (`"stance"`, `"*"`, or `None` for raw probabilities) |

The `Decision` itself records `model_sha256`, `template_sha256`, `truncated` (the state was cut
to fit), `tokens` (state and question tokens) and `timing_ms`. Together with the judgly
version, these identify exactly what produced an answer. Store them if you need an audit trail.

## Many questions at once

Put all questions about one text into one request. The text is processed once, and each
question adds only its own tokens. The questions are isolated: adding or removing a question
does not change the others' answers beyond floating-point noise (the self-test T5 checks this
to 0.01; [many_questions.py](../examples/many_questions.py) shows it).

A request can hold many questions; large lists are processed in groups inside the engine. If the
text and the questions together do not fit in the engine's context (`n_ctx` tokens), the request
is refused with a message and the engine stays usable.

## Threads and asyncio

A native handle answers one request at a time. `Engine` serialises calls with a lock, so an
engine can be shared between threads, but requests wait for each other. `adecide` runs
`decide` in a worker thread for asyncio code:

```python
d = await engine.adecide(state, questions)
```

Load one engine at start-up and share it. Every `Engine` holds its own copy of the model, so
two engines on the same model use twice the memory. See
[async_usage.py](../examples/async_usage.py).

## The JSON interface

`engine.decide_json(request_json)` takes and returns the JSON that the C library speaks. It is
useful for logging requests or calling judgly from other languages through `libjudgly`. The
shapes are documented at the top of [`csrc/judgly.h`](../csrc/judgly.h):

```python
import json
request = {"schema": 1, "state": "The parcel arrived today.",
           "questions": {"late": {"type": "bool", "instructions": "Was the parcel late?"}}}
response = json.loads(engine.decide_json(json.dumps(request)))
```

`decide_json` returns `{"error": "..."}` instead of raising; `decide` raises `JudglyError`.

## Engine options

`Engine.load(pack, **options)` overrides the pack's engine settings. The built-in packs set
`rotations=True`, `max_rotations=4` and `content_free=False` in their `engine` block, and those
are the settings their heads were fitted with. Each head's sidecar (`heads/*.bin.json`) records
them, and the engine refuses to load a head under other settings (`JudglyError`), because a head
fitted with one setting is not calibrated under another; pass `heads=False` to change them and
read raw probabilities. The default column
below is the library default (`judgly.EngineConfig`), used when a pack sets nothing.

| option | default | meaning |
|---|---|---|
| `rotations` | `True` | ask each question in several cyclic orders of its options (up to `max_rotations`) and average (always on for yes/no questions; never for scores, whose levels keep their natural order) |
| `max_rotations` | `4` | at most this many orders per question, evenly spaced; 0 means all of them |
| `content_free` | `False` | also ask each question against the empty state "N/A" and give the head those scores, to discount the model's standing preferences among options |
| `n_ctx` | `32768` | context size in tokens, shared by the text and the question branches (at least 4096) |
| `n_seq` | `65` | sequences per GPU call, the state included |
| `max_state_tokens` | `n_ctx / 2` | longer states are cut to their first this many tokens; the response says `truncated` |
| `max_request_bytes` | `4194304` | requests larger than this (4 MiB) are refused before parsing |
| `max_questions` | `256` | requests with more questions are refused; each question (its instructions and options) must also fit in 4,096 tokens |
| `n_gpu_layers` | `999` | layers on the GPU; 0 runs on the CPU (slow) |
| `verbose` | `False` | let llama.cpp log to stderr |
| `plain_slots` | `False` | read the answer letters without a leading space (" A" is read by default); only for a template that ends the prompt with a space |

The C library has two different defaults: `max_rotations` 0 (all orders) and `content_free`
true, as documented in [`csrc/judgly.h`](../csrc/judgly.h). The Python `EngineConfig` defaults
above are the ones the built-in packs use, and `Engine.load` sends every setting explicitly, so
the C defaults only matter when calling `libjudgly` directly.

## Errors

- `JudglyError` (a `RuntimeError`): the engine refused or could not answer a request, or a head
  was fitted under other engine settings. The message says why. If the engine itself failed
  during a request, it marks itself broken and asks to be reopened: close it and load a new one.
- `RuntimeError`: the native library could not open the model, template or heads (for example
  a corrupt file or too little memory); the message comes from the library.
- `OSError`: the native library is missing from the installed package.
- `FileNotFoundError`: `Engine.load` or `Pack.find` got a name that is neither a built-in pack
  nor a directory holding `pack.json`.
- `HeadsNotAvailable` (a `FileNotFoundError`): the pack names heads whose files are missing.
- `ModelMismatch` (a `ValueError`): the model file's size or SHA-256 is not the one the pack
  pins, or a head file is not the one the pack names.
- `ValueError`: `pack.json` has an unsupported schema, or `calibration` is not one of
  `"default"`, `"h2"`, `"temperature"` and `"raw"`, names an option the pack does not offer, or
  contradicts `heads`.
- `TypeError`: `Engine.load` was given `model`, `template` or `model_sha256`, which the pack
  fixes (use `model_path=` or `Engine(config)`).
- `pydantic.ValidationError`: a question or an engine option is malformed (for example a single
  option, 10 levels, or an unknown option name). An option value outside the range the library
  accepts (for example `n_ctx` below 4096) is refused by the library when the engine loads,
  with `JudglyError`.
- `MemoryError`: the library ran out of memory while answering.

## API reference

Everything below is importable from `judgly`.

- `Engine.load(pack, *, calibration="default", heads=True, model_path=None, **options) -> Engine`:
  `pack` is a built-in pack name or a pack directory. `calibration` is `"default"` (the pack's
  default option per format), `"h2"`, `"temperature"` or `"raw"` (see
  [Load an engine](#load-an-engine)). `heads=True` uses the pack's heads for that choice,
  `heads=False` none (raw probabilities), and a mapping `{format: path}` those files. `model_path` uses a
  local copy of the pack's model file (checked like a download); otherwise
  `JUDGLY_MODEL_DIR/<file name>` is used if it exists, and the file is downloaded if not.
  `options` override `EngineConfig` fields (see [Engine options](#engine-options)).
- `Engine(config, *, pack=None)`: an engine from an `EngineConfig` or a dict of its fields. No
  pack is involved: whatever `model`, `template` and `heads` the configuration names are
  loaded. If it gives `model_sha256`, the file is trusted to have that hash and is not hashed.
- `Engine.decide(state, questions) -> Decision`, `await Engine.adecide(state, questions)`,
  `Engine.decide_json(request_json) -> str`, `Engine.close()`; `Engine.config`, `Engine.pack`
  and `Engine.calibration` (the option used per format; `{}` for raw or your own head files)
  hold what was loaded.
- `EngineConfig`: `model`, `template`, `heads` (`{format: path}`), `model_sha256`, and the
  options in the table above. Unknown fields are refused.
- `Pack.find(name_or_path) -> Pack`; `Pack.heads(calibration="default") -> dict[str, Path]`
  (the head file per format for that choice, checked against the SHA-256 in `pack.json`);
  `Pack.defaults() -> dict[str, str]` (the default option per format);
  `Pack.calibration_options()` (per format, the default and every option's `pack.json` entry);
  `Pack.name`, `Pack.directory`, `Pack.model`, `Pack.template`, `Pack.engine_options`.
- `list_packs() -> list[str]`: the built-in pack names.
- `native_version() -> str`: JSON with the judgly version, the llama.cpp commit and the
  backends compiled into the native library.
- `__version__`: the installed package version.
- Questions (keyword arguments only): `Choice(*, instructions, options, format=None)`,
  `Binary(*, instructions, format=None)`, `Score(*, instructions, levels, format=None)`.
- Answers (`ChoiceAnswer`, `BinaryAnswer`, `ScoreAnswer`) all carry `type` (`"choice"`,
  `"bool"` or `"score"`), `format` (as asked), `head` (the head file's SHA-256, or `None` for
  raw probabilities), `head_format` (the heads entry used: the format, `"*"` or `None`),
  `slot_mass`, `rotation_spread` and `n_rotations`, plus `probs` and `top` (choice and score),
  `p_true` and `top` (yes or no) and `mean` (score).
- `Decision`: `answers`, `model_sha256`, `template_sha256`, `truncated`, `tokens.state`,
  `tokens.questions`, `timing_ms.state`, `timing_ms.questions`; `d[qid]` is `d.answers[qid]`.
  `tokens.state` counts the state together with the template text around it, so it can be a
  little above `max_state_tokens` after truncation.

## Speed

A request costs one pass over the text plus a short pass per question and option order (at
most four orders in the built-in packs). As a rough idea: on an M3 Max with Qwen3-4B,
[many_questions.py](../examples/many_questions.py) (a 109-token review, 11 questions) took
about 0.6 s. Gemma 4 12B is slower.
