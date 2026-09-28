"""Model packs: lookup, head availability, and the SHA-256 check of the model file."""

import hashlib
import json

import pytest

from judgly import ModelMismatch, Pack, list_packs
from judgly.packs import resolve_model


def test_builtin_packs():
    assert {"qwen3-4b-q8", "gemma4-12b-q8"} <= set(list_packs())
    for name in list_packs():
        p = Pack.find(name)
        assert p.template.is_file()
        m = p.model
        assert len(m["sha256"]) == 64 and m["size"] > 0 and len(m["revision"]) == 40
        assert m["repo_id"] and m["filename"].endswith(".gguf")


@pytest.mark.parametrize("name", ["gemma4-12b-q8", "qwen3-4b-q8"])
def test_builtin_packs_ship_general_and_stance_heads(name):
    heads = Pack.find(name).heads()
    assert set(heads) == {"*", "stance"}
    assert all(p.is_file() for p in heads.values())


def test_unknown_pack():
    with pytest.raises(FileNotFoundError, match="qwen3-4b-q8"):
        Pack.find("no-such-pack")


def fake_pack(tmp_path, content: bytes, sha: str, size: int):
    (tmp_path / "template.tpl").write_text('{"open": "", "close": ""}')
    spec = {"schema": 1, "name": "fake", "template": "template.tpl", "heads": {},
            "model": {"repo_id": "none/none", "revision": "0" * 40, "filename": "m.gguf",
                      "sha256": sha, "size": size}}
    (tmp_path / "pack.json").write_text(json.dumps(spec))
    model = tmp_path / "m.gguf"
    model.write_bytes(content)
    return Pack.find(tmp_path), model


def test_model_hash_checked(tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGLY_CACHE", str(tmp_path / "cache"))
    content = b"not really a model"
    good = hashlib.sha256(content).hexdigest()
    pack, model = fake_pack(tmp_path, content, good, len(content))
    assert resolve_model(pack, model) == (model, good)
    assert pack.heads() == {}

    pack, model = fake_pack(tmp_path, content, "f" * 64, len(content))
    with pytest.raises(ModelMismatch, match="SHA-256"):
        resolve_model(pack, model)
    pack, model = fake_pack(tmp_path, content, good, len(content) + 1)
    with pytest.raises(ModelMismatch, match="bytes"):
        resolve_model(pack, model)


def test_local_model_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGLY_CACHE", str(tmp_path / "cache"))
    content = b"x" * 10
    pack, model = fake_pack(tmp_path, content, hashlib.sha256(content).hexdigest(), 10)
    monkeypatch.setenv("JUDGLY_MODEL_DIR", str(tmp_path))
    assert resolve_model(pack)[0] == model


def test_head_hash_checked(tmp_path):
    (tmp_path / "template.tpl").write_text('{"open": "", "close": ""}')
    (tmp_path / "heads").mkdir()
    (tmp_path / "heads" / "h2.bin").write_bytes(b"a head")
    good = hashlib.sha256(b"a head").hexdigest()
    spec = {"schema": 1, "name": "fake", "template": "template.tpl",
            "heads": {"*": {"file": "heads/h2.bin", "sha256": good}},
            "model": {"repo_id": "none/none", "revision": "0" * 40, "filename": "m.gguf",
                      "sha256": "0" * 64, "size": 1}}
    (tmp_path / "pack.json").write_text(json.dumps(spec))
    assert Pack.find(tmp_path).heads() == {"*": tmp_path / "heads" / "h2.bin"}
    (tmp_path / "heads" / "h2.bin").write_bytes(b"a refitted head")
    with pytest.raises(ModelMismatch, match="SHA-256"):
        Pack.find(tmp_path).heads()


@pytest.mark.parametrize("option", ["model", "model_sha256", "template"])
def test_load_refuses_pack_fixed_options(option):
    from judgly import Engine
    with pytest.raises(TypeError, match=option):
        Engine.load("qwen3-4b-q8", heads=False, **{option: "/tmp/other"})
