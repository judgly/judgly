"""The Engine: one loaded model answering typed questions about a state.

    from judgly import Engine, Choice, Binary

    with Engine.load("qwen3-4b-q8", heads=False) as engine:
        d = engine.decide("The kettle is boiling.", {
            "hot": Binary(instructions="Is the water hot?"),
            "room": Choice(instructions="Which room is this most likely?",
                           options=["kitchen", "garage", "bedroom"]),
        })
        print(d["hot"].p_true, d["room"].probs)

One Engine holds one handle of the native library. Calls on an Engine are serialised with a
lock, so it may be shared between threads, but only one decision runs at a time.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from judgly import _native
from judgly._native import JudglyError
from judgly.models import Decision, Question, Request
from judgly.packs import Pack, resolve_model

__all__ = ["Engine", "EngineConfig", "JudglyError"]


class EngineConfig(BaseModel):
    """What the native library loads: see csrc/judgly.h for each field."""

    model_config = ConfigDict(extra="forbid")

    model: Path
    template: Path
    heads: dict[str, Path] = Field(default_factory=dict)
    model_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    n_ctx: int = 32768
    n_seq: int = 65
    n_gpu_layers: int = 999
    rotations: bool = True
    max_rotations: int = 4
    content_free: bool = False
    max_state_tokens: int | None = None
    max_request_bytes: int = Field(default=4 << 20, ge=64, le=1 << 30)
    max_questions: int = Field(default=256, ge=1, le=1 << 16)
    plain_slots: bool = False
    verbose: bool = False


_QUESTIONS = TypeAdapter(dict[str, Question])

ENGINE_SETTINGS = ("rotations", "max_rotations", "content_free")


def check_head_settings(config: EngineConfig) -> None:
    """Refuses a head fitted under other engine settings than the configuration's.

    s1-train records the settings its features were read with in the head's sidecar
    (HEAD.bin.json, "engine"); a head only describes readouts taken the same way. A head file
    without a sidecar is taken as it is."""
    for fmt, head in config.heads.items():
        sidecar = Path(f"{head}.json")
        if not sidecar.is_file():
            continue
        fitted = json.loads(sidecar.read_text(encoding="utf-8")).get("engine")
        if fitted is None:
            raise JudglyError(f"judgly: head {fmt!r} ({head}): its sidecar {sidecar.name} does not "
                              "record the engine settings it was fitted under")
        wanted = {k: getattr(config, k) for k in ENGINE_SETTINGS}
        if any(fitted.get(k) != v for k, v in wanted.items()):
            raise JudglyError(f"judgly: head {fmt!r} ({head}) was fitted with engine settings "
                              f"{ {k: fitted.get(k) for k in ENGINE_SETTINGS} }, not {wanted}; "
                              "use the pack's settings, or heads=False")


class Engine:
    """A loaded model, template and heads. Close it (or use ``with``) to free the model."""

    def __init__(self, config: EngineConfig | Mapping[str, Any], *, pack: Pack | None = None):
        self.config = config if isinstance(config, EngineConfig) else EngineConfig(**config)
        check_head_settings(self.config)
        self.pack = pack
        self._lock = threading.Lock()
        self._handle: int | None = _native.open_handle(
            self.config.model_dump_json(exclude_none=True))

    @classmethod
    def load(cls, pack: str | os.PathLike, *, heads: bool | Mapping[str, str | os.PathLike] = True,
             model_path: str | os.PathLike | None = None, allow_unverified_heads: bool = False,
             **options: Any) -> Engine:
        """An Engine from a model pack: a built-in pack name or a pack directory.

        The model file is downloaded into the Hugging Face cache on first use (or taken from
        ``model_path``) and its SHA-256 is checked against the pack. ``heads=True`` uses the
        pack's heads, ``heads=False`` none (raw letter probabilities, H0), and a mapping of
        format to head file uses those. Other keyword arguments override EngineConfig fields,
        except the ones the pack fixes and verifies (model, model_sha256, template, heads).
        A pack head without a SHA-256 is refused unless ``allow_unverified_heads=True``.
        """
        fixed = sorted(set(options) & {"model", "model_sha256", "template"})
        if fixed:
            raise TypeError(f"judgly: Engine.load does not take {', '.join(fixed)}: the pack fixes "
                            "them (use model_path= for a local copy of the pack's model file, or "
                            "Engine(config) for a free configuration)")
        p = Pack.find(pack)
        path, sha = resolve_model(p, model_path)
        if heads is True:
            head_files = p.heads(allow_unverified=allow_unverified_heads)
        elif heads is False:
            head_files = {}
        else:
            head_files = {k: Path(v) for k, v in heads.items()}
        config = {**p.engine_options, **options, "model": path, "template": p.template,
                  "heads": head_files, "model_sha256": sha}
        return cls(EngineConfig(**config), pack=p)

    # ---- deciding -----------------------------------------------------------------------

    def decide_json(self, request_json: str) -> str:
        """The raw interface: a request as JSON in, the response as JSON out (which may be
        ``{"error": ...}``). See csrc/judgly.h for both shapes."""
        with self._lock:
            if self._handle is None:
                raise JudglyError("judgly: the engine is closed")
            return _native.decide(self._handle, request_json)

    def decide(self, state: str,
               questions: Mapping[str, Question | Mapping[str, Any]]) -> Decision:
        """Answers every question about ``state``. Questions are models (Choice, Binary,
        Score) or plain dicts of the same shape; the state is read once and each question
        branches from it on its own."""
        request = Request(state=state, questions=_QUESTIONS.validate_python(dict(questions)))
        response = json.loads(self.decide_json(request.to_json()))
        if "error" in response:
            raise JudglyError(f"judgly: {response['error']}")
        return Decision.model_validate(response)

    async def adecide(self, state: str,
                      questions: Mapping[str, Question | Mapping[str, Any]]) -> Decision:
        """``decide`` in a worker thread. Concurrent calls queue on the engine's lock."""
        return await asyncio.to_thread(self.decide, state, questions)

    # ---- lifetime ---------------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                _native.close_handle(self._handle)
                self._handle = None

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
