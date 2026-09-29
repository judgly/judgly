"""Model packs: which GGUF file, which prompt template and which heads belong together.

A pack is a directory with a ``pack.json``. The GGUF file itself is not in the pack; the pack
names the Hugging Face repository, revision and file it comes from and the file's SHA-256.
``resolve_model`` downloads it into the Hugging Face cache on first use (or takes a local copy)
and checks the SHA-256 before the file is used. Heads are only valid for the exact model file
and template they were fitted with.

Calibration options. A pack of schema 2 offers, per format, several calibration options, each a
head file: ``"h2"`` (the fitted head H2, applied to every option order's letter logits) and
``"temperature"`` (one temperature per question type, applied to the probabilities averaged over
the option orders), and names the one it uses by ``"default"``. ``"raw"`` means no head. A pack
of schema 1 has one head per format, which is its only option and its default.

    "heads": {"*":      {"default": "h2", "default_basis": "...",
                         "options": {"h2": {"file": ..., "sha256": ...},
                                     "temperature": {"file": ..., "sha256": ...}}},
              "stance": {...}}
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

__all__ = ["Pack", "HeadsNotAvailable", "ModelMismatch", "list_packs", "resolve_model", "CALIBRATIONS"]

CALIBRATIONS = ("default", "h2", "temperature", "raw")
"""The values of ``calibration`` in ``Pack.heads`` and ``Engine.load``: the pack's default per
format, the H2 head, the per-type temperature, or no calibration (raw letter probabilities)."""

_BUILTIN = files("judgly") / "packs"


class HeadsNotAvailable(FileNotFoundError):
    """The pack names heads whose files are not present (yet)."""


class ModelMismatch(ValueError):
    """A model file's SHA-256 is not the one the pack names."""


@dataclass(frozen=True)
class Pack:
    name: str
    directory: Path
    spec: dict

    @classmethod
    def find(cls, name_or_path: str | os.PathLike) -> Pack:
        """A built-in pack by name, or a pack directory (or its pack.json) by path.

        A built-in name wins: a string is taken as a path only when it is not a built-in name
        and contains a path separator ("./my-pack", not "my-pack"). A PathLike is always a
        path."""
        text = os.fspath(name_or_path)
        is_path = isinstance(name_or_path, os.PathLike) or _has_separator(text)
        if not is_path:
            if text in list_packs():
                return cls._read(Path(str(_BUILTIN / text)))
            known = ", ".join(list_packs()) or "none"
            raise FileNotFoundError(f"judgly: no built-in pack {text!r} (built-in packs: {known}; "
                                    "for a pack directory give a path such as ./my-pack)")
        path = Path(text)
        if path.name == "pack.json":
            path = path.parent
        if not (path / "pack.json").is_file():
            raise FileNotFoundError(f"judgly: {path} has no pack.json")
        return cls._read(path)

    @classmethod
    def _read(cls, path: Path) -> Pack:
        spec = json.loads((path / "pack.json").read_text(encoding="utf-8"))
        if spec.get("schema") not in (1, 2):
            raise ValueError(f"judgly: {path / 'pack.json'}: unsupported pack schema")
        return cls(spec["name"], path, spec)

    def _inside(self, relative: str, what: str) -> Path:
        """A file the pack names, which must lie inside the pack directory."""
        root = self.directory.resolve()
        path = (self.directory / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"judgly: pack {self.name!r}: {what} {relative!r} lies outside the "
                             f"pack directory {self.directory}")
        return self.directory / relative

    @property
    def model(self) -> dict:
        return self.spec["model"]

    @property
    def template(self) -> Path:
        return self._inside(self.spec["template"], "template")

    @property
    def engine_options(self) -> dict:
        return dict(self.spec.get("engine", {}))

    def calibration_options(self) -> dict[str, dict]:
        """Per format: {"default": option, "options": {option: head entry}}, the form of schema 2.
        A schema-1 entry (one head per format) is its format's only option, "head"."""
        out = {}
        for fmt, entry in self.spec.get("heads", {}).items():
            if self.spec.get("schema") == 1:
                out[fmt] = {"default": "head", "options": {"head": entry}}
            else:
                if entry.get("default") not in entry.get("options", {}):
                    raise ValueError(f"judgly: pack {self.name!r}: format {fmt!r} names default "
                                     f"{entry.get('default')!r}, which is not one of its options "
                                     f"{sorted(entry.get('options', {}))}")
                out[fmt] = entry
        return out

    def defaults(self) -> dict[str, str]:
        """The calibration option each format uses by default."""
        return {fmt: e["default"] for fmt, e in self.calibration_options().items()}

    def heads(self, *, allow_unverified: bool = False,
              calibration: str = "default") -> dict[str, Path]:
        """Head file per format for one calibration choice (see CALIBRATIONS): the pack's default
        option of each format, "h2" or "temperature" for every format, or "raw" (no heads, {}).

        Raises ValueError for an unknown choice or one a format does not offer, HeadsNotAvailable
        if any named file is missing, and ModelMismatch if a file differs from the SHA-256 the
        pack names for it. An entry that names no SHA-256 is refused (ValueError) unless
        ``allow_unverified`` is true."""
        if calibration not in CALIBRATIONS:
            raise ValueError(f"judgly: calibration must be one of {', '.join(map(repr, CALIBRATIONS))}, "
                             f"not {calibration!r}")
        if calibration == "raw":
            return {}
        heads: dict[str, Path] = {}
        missing = []
        for fmt, choice in self.calibration_options().items():
            option = choice["default"] if calibration == "default" else calibration
            if option not in choice["options"]:
                raise ValueError(f"judgly: pack {self.name!r} offers no calibration {option!r} for "
                                 f"format {fmt!r} (it offers {', '.join(sorted(choice['options']))})")
            entry = choice["options"][option]
            path = self._inside(entry["file"], f"head {fmt!r}")
            if not entry.get("sha256") and not allow_unverified:
                raise ValueError(f"judgly: pack {self.name!r}: head {fmt!r} names no SHA-256; add "
                                 "one to pack.json, or pass allow_unverified_heads=True to use it "
                                 "unchecked")
            if path.is_file():
                if entry.get("sha256") and sha256_file(path) != entry["sha256"]:
                    raise ModelMismatch(f"judgly: head {fmt!r} ({path}) does not have the SHA-256 "
                                        f"that pack {self.name!r} names ({entry['sha256']}); its "
                                        "calibration record would describe another head")
                heads[fmt] = path
            else:
                missing.append(f"{fmt!r} ({entry.get('status', 'missing')})")
        if missing:
            raise HeadsNotAvailable(
                f"judgly: pack {self.name!r} has no head file yet for {', '.join(missing)}. "
                "Heads are fitted separately, per model and template. Pass heads=False for raw, "
                "uncalibrated letter probabilities (H0), or give heads={format: path} yourself.")
        return heads


def _has_separator(text: str) -> bool:
    return "/" in text or os.sep in text or bool(os.altsep and os.altsep in text)


def list_packs() -> list[str]:
    """Names of the built-in packs."""
    return sorted(p.name for p in _BUILTIN.iterdir() if p.joinpath("pack.json").is_file())


def _cache_dir() -> Path:
    root = os.environ.get("JUDGLY_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "judgly")
    return Path(root)


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verified_sha256(path: Path) -> str:
    """SHA-256 of a file, remembered by path, size and modification time so that a large model
    file is hashed once, not on every load."""
    path = path.resolve()
    st = path.stat()
    key = f"{path}|{st.st_size}|{st.st_mtime_ns}"
    record = _cache_dir() / "sha256.json"
    try:
        known = json.loads(record.read_text())
    except (OSError, ValueError):
        known = {}
    if key not in known:
        known[key] = sha256_file(path)
        try:
            record.parent.mkdir(parents=True, exist_ok=True)
            tmp = record.with_suffix(".tmp")
            tmp.write_text(json.dumps(known, indent=1))
            tmp.replace(record)
        except OSError:
            pass  # the cache is a convenience; the hash is still checked
    return known[key]


def resolve_model(pack: Pack, model_path: str | os.PathLike | None = None) -> tuple[Path, str]:
    """The pack's model file and its SHA-256, checked against the pack.

    With ``model_path``, that local file is used; otherwise the file is downloaded from the
    pack's Hugging Face repository at the pinned revision (or found in the Hugging Face cache).
    Set ``JUDGLY_MODEL_DIR`` to a directory holding the file to skip the download.
    """
    spec = pack.model
    if model_path is None and os.environ.get("JUDGLY_MODEL_DIR"):
        candidate = Path(os.environ["JUDGLY_MODEL_DIR"]) / spec["filename"]
        model_path = candidate if candidate.is_file() else None
    if model_path is None:
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(repo_id=spec["repo_id"], filename=spec["filename"],
                                     revision=spec.get("revision"))
    path = Path(model_path)
    if path.stat().st_size != spec["size"]:
        raise ModelMismatch(f"judgly: {path} has {path.stat().st_size} bytes; pack "
                            f"{pack.name!r} expects {spec['size']}")
    sha = verified_sha256(path)
    if sha != spec["sha256"]:
        raise ModelMismatch(f"judgly: {path} has SHA-256 {sha}; pack {pack.name!r} expects "
                            f"{spec['sha256']}")
    return path, sha
