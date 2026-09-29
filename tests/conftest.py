"""Fixtures. Model-dependent tests are marked ``model`` and skip when the model file is absent.

The model tests run the built-in ``qwen3-4b-q8`` pack with its shipped settings and heads. They
need the Qwen3-4B-Instruct-2507 Q8_0 GGUF file named in
``src/judgly/packs/qwen3-4b-q8/pack.json``, found through either of:

- ``JUDGLY_TEST_MODEL``: the path of the file itself;
- ``JUDGLY_MODEL_DIR``: a directory holding it under its pack file name.

Without it, every test that runs the engine skips.
"""

import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures"
PACK_DIR = REPO / "src" / "judgly" / "packs" / "qwen3-4b-q8"
QWEN_FILE = json.loads((PACK_DIR / "pack.json").read_text())["model"]["filename"]


def _model() -> Path:
    if os.environ.get("JUDGLY_TEST_MODEL"):
        return Path(os.environ["JUDGLY_TEST_MODEL"])
    if os.environ.get("JUDGLY_MODEL_DIR"):
        return Path(os.environ["JUDGLY_MODEL_DIR"]) / QWEN_FILE
    return Path("JUDGLY_TEST_MODEL or JUDGLY_MODEL_DIR") / QWEN_FILE


QWEN = _model()


def need(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip("missing: " + ", ".join(missing))


@pytest.fixture(scope="session")
def qwen_h0():
    """The built-in pack with heads off: raw letter probabilities."""
    need(QWEN)
    from judgly import Engine

    with Engine.load("qwen3-4b-q8", heads=False, model_path=QWEN) as engine:
        yield engine


@pytest.fixture(scope="session")
def qwen_h2():
    """The built-in pack with its shipped H2 heads (general and stance). The pack's default for
    Qwen3-4B is the temperature (tests/test_temperature.py), so H2 is chosen explicitly."""
    need(QWEN)
    from judgly import Engine

    with Engine.load("qwen3-4b-q8", calibration="h2", model_path=QWEN) as engine:
        yield engine
