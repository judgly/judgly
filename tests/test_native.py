"""The C API without a model: configuration errors come back as messages."""

import json

import pytest

import judgly
from judgly import _native


def test_version():
    v = json.loads(judgly.native_version())
    assert v["judgly"] == judgly.__version__ and len(v["llama_cpp_commit"]) == 40


@pytest.mark.parametrize("config, message", [
    ("not json", "not a JSON object"),
    ('{"model": 1}', "must be file paths"),
    ('{"model": "m", "template": "t", "colour": 1}', 'unknown field "colour"'),
    ('{"model": "m", "template": "t", "n_seq": 1}', '"n_seq" must be an integer'),
    ('{"model": "m", "template": "t", "model_sha256": "abc"}', "64 lower-case hex"),
    ('{"model": "/nonexistent.gguf", "template": "t"}', "cannot read the model file"),
])
def test_open_errors(config, message):
    with pytest.raises(RuntimeError, match=message):
        _native.open_handle(config)


def test_small_context_refused():
    """The engine decodes up to 4096 tokens per call; llama.cpp aborts if n_ctx is smaller."""
    with pytest.raises(RuntimeError, match='"n_ctx" must be an integer from 4096'):
        _native.open_handle('{"model": "m", "template": "t", "n_ctx": 512}')
