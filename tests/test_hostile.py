"""Hostile and malformed input: pack resolution, request limits, duplicate keys, NUL characters,
head files and native load failures. Each rule is refused with a clear message."""

import hashlib
import json
import struct
from pathlib import Path

import pytest

from conftest import QWEN, REPO, need
from judgly import Binary, Engine, EngineConfig, JudglyError, Pack, _native
from judgly.packs import list_packs

SHIPPED_HEAD = REPO / "src" / "judgly" / "packs" / "qwen3-4b-q8" / "heads" / "h2.bin"


# ---- packs ---------------------------------------------------------------------------------


def write_pack(directory: Path, *, template="template.tpl", head_file="heads/h.bin",
               head_sha: str | None = "set") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "template.tpl").write_text('{"open": "", "close": ""}')
    (directory / "heads").mkdir(exist_ok=True)
    (directory / "heads" / "h.bin").write_bytes(b"a head")
    entry = {"file": head_file}
    if head_sha == "set":
        entry["sha256"] = hashlib.sha256(b"a head").hexdigest()
    spec = {"schema": 1, "name": "custom", "template": template, "heads": {"*": entry},
            "model": {"repo_id": "none/none", "revision": "0" * 40, "filename": "m.gguf",
                      "sha256": "0" * 64, "size": 1}}
    (directory / "pack.json").write_text(json.dumps(spec))
    return directory


def test_builtin_name_wins_over_local_directory(tmp_path, monkeypatch):
    name = list_packs()[0]
    write_pack(tmp_path / name)
    monkeypatch.chdir(tmp_path)
    assert Pack.find(name).name == name
    assert Pack.find(f"./{name}").name == "custom"
    assert Pack.find(Path(name)).name == "custom"


def test_bare_name_is_not_a_path(tmp_path, monkeypatch):
    write_pack(tmp_path / "mine")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="no built-in pack 'mine'"):
        Pack.find("mine")
    assert Pack.find("./mine").name == "custom"


def test_template_outside_pack_refused(tmp_path):
    (tmp_path / "outside.tpl").write_text("{}")
    pack = Pack.find(write_pack(tmp_path / "p", template="../outside.tpl"))
    with pytest.raises(ValueError, match="outside the pack directory"):
        _ = pack.template


def test_head_outside_pack_refused(tmp_path):
    pack = Pack.find(write_pack(tmp_path / "p", head_file="../../etc/passwd"))
    with pytest.raises(ValueError, match="outside the pack directory"):
        pack.heads()


def test_head_symlink_outside_pack_refused(tmp_path):
    (tmp_path / "elsewhere.bin").write_bytes(b"a head")
    directory = write_pack(tmp_path / "p", head_file="heads/link.bin")
    (directory / "heads" / "link.bin").symlink_to(tmp_path / "elsewhere.bin")
    with pytest.raises(ValueError, match="outside the pack directory"):
        Pack.find(directory).heads()


def test_head_without_sha256_needs_opt_out(tmp_path):
    pack = Pack.find(write_pack(tmp_path / "p", head_sha=None))
    with pytest.raises(ValueError, match="names no SHA-256"):
        pack.heads()
    assert pack.heads(allow_unverified=True) == {"*": tmp_path / "p" / "heads" / "h.bin"}


# ---- native load failures --------------------------------------------------------------------


def test_open_failure_is_judgly_error():
    with pytest.raises(JudglyError, match="cannot read the model file"):
        _native.open_handle('{"model": "/nonexistent.gguf", "template": "t"}')
    assert issubclass(JudglyError, RuntimeError)


def test_missing_library_is_judgly_error(monkeypatch):
    _native._lib.cache_clear()
    monkeypatch.setattr(_native, "_LIBNAME", "libjudgly-missing.dylib")
    try:
        with pytest.raises(JudglyError, match="not found"):
            _native._lib()
    finally:
        monkeypatch.undo()
        _native._lib.cache_clear()


@pytest.mark.parametrize("config, message", [
    ('{"model": "m", "template": "t", "max_request_bytes": 0}', '"max_request_bytes" must be'),
    ('{"model": "m", "template": "t", "max_questions": 0}', '"max_questions" must be'),
])
def test_limit_config_bounds(config, message):
    with pytest.raises(JudglyError, match=message):
        _native.open_handle(config)


def test_limit_defaults():
    config = EngineConfig(model="m", template="t")
    assert config.max_request_bytes == 4 << 20 and config.max_questions == 256


# ---- head files ------------------------------------------------------------------------------


def head_load(path: Path) -> int:
    return _native._lib().judgly_test_head_load(str(path).encode())


def test_head_file_parser(tmp_path):
    good = SHIPPED_HEAD.read_bytes()
    assert head_load(SHIPPED_HEAD) == 0

    trailing = tmp_path / "trailing.bin"
    trailing.write_bytes(good + b"\0")
    assert head_load(trailing) == -1

    truncated = tmp_path / "truncated.bin"
    truncated.write_bytes(good[:-1])
    assert head_load(truncated) == -1

    for n_embd in (0, 65537, 0xFFFFFFFF):
        bad = tmp_path / f"n_embd_{n_embd}.bin"
        bad.write_bytes(good[:16] + struct.pack("<I", n_embd) + good[20:])
        assert head_load(bad) == -1


# ---- build -----------------------------------------------------------------------------------


def test_library_has_no_build_paths():
    lib = Path(_native._lib()._name)
    assert str(REPO).encode() not in lib.read_bytes()


# ---- requests (need the model) ---------------------------------------------------------------

BOOL = '{"type": "bool", "instructions": "Is it?"}'


def error_of(engine, request_json: str) -> str:
    return json.loads(engine.decide_json(request_json))["error"]


def test_request_too_large(qwen_h0):
    big = json.dumps({"state": "x" * (4 << 20), "questions": {"q": json.loads(BOOL)}})
    assert "more than max_request_bytes 4194304" in error_of(qwen_h0, big)


def test_too_many_questions(qwen_h0):
    qs = ", ".join(f'"q{i}": {BOOL}' for i in range(257))
    assert "257 questions, more than max_questions 256" in error_of(
        qwen_h0, f'{{"state": "s", "questions": {{{qs}}}}}')
    with pytest.raises(JudglyError, match="max_questions"):
        qwen_h0.decide("s", {f"q{i}": Binary(instructions="x") for i in range(257)})


@pytest.mark.parametrize("request_json, message", [
    (f'{{"state": "a", "state": "b", "questions": {{"q": {BOOL}}}}}',
     'request: duplicate key "state"'),
    (f'{{"state": "s", "questions": {{"q": {BOOL}, "q": {BOOL}}}}}',
     'request: "questions": duplicate key "q"'),
    ('{"state": "s", "questions": {"q": {"type": "bool", "instructions": "a", '
     '"instructions": "b"}}}', 'question "q": duplicate key "instructions"'),
    ('{"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", '
     '"options": {"a": "A", "b": "B", "a": "C"}}}}', 'question "q": "options": duplicate key "a"'),
])
def test_duplicate_keys(qwen_h0, request_json, message):
    assert message in error_of(qwen_h0, request_json)


@pytest.mark.parametrize("request_json", [
    f'{{"state": "a\\u0000b", "questions": {{"q": {BOOL}}}}}',
    f'{{"state": "s", "questions": {{"q\\u0000x": {BOOL}}}}}',
    f'{{"state\\u0000x": "s", "state": "s", "questions": {{"q": {BOOL}}}}}',
    '{"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", '
    '"options": {"a\\u0000": "A", "b": "B"}}}}',
])
def test_embedded_nul(qwen_h0, request_json):
    assert "contains a NUL character" in error_of(qwen_h0, request_json)


def test_embedded_nul_through_decide(qwen_h0):
    with pytest.raises(JudglyError, match="NUL"):
        qwen_h0.decide("a\0b", {"q": Binary(instructions="Is it?")})


def test_state_cut_by_bytes_first():
    """A state far longer than max_state_tokens is cut by bytes, at a UTF-8 boundary, before
    tokenising; the answer still comes back and says it was truncated."""
    need(QWEN)
    with Engine.load("qwen3-4b-q8", heads=False, model_path=QWEN, max_state_tokens=16,
                     n_ctx=4096) as engine:
        for state in ("é" * 200_000, "a" * 200_000):
            d = engine.decide(state, {"q": Binary(instructions="Is this repetitive?")})
            assert d.truncated and d.tokens.state < 60
