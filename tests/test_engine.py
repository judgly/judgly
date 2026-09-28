"""judgly on a real model (Qwen3-4B, the built-in pack): request checks in C, answers, and
regression against golden values in tests/fixtures/golden-qwen3-4b-q8.json (heads off and the
shipped heads on), written by scripts/make_golden.py."""

import asyncio
import json

import pytest
from pydantic import ValidationError

from conftest import FIXTURES, QWEN, need
from judgly import Binary, Engine, JudglyError, Score

pytestmark = pytest.mark.model

GOLDEN = json.loads((FIXTURES / "golden-qwen3-4b-q8.json").read_text())
REQUEST = GOLDEN["request"]
TOL = 1e-6


def assert_golden(ours, gold):
    assert ours["model_sha256"] == gold["model_sha256"]
    for qid, spec in REQUEST["questions"].items():
        a, g = ours["answers"][qid], gold["answers"][qid]
        if spec["type"] == "bool":
            assert abs(a["p_true"] - g["p_true"]) <= TOL
        elif spec["type"] == "choice":
            assert a["probs"].keys() == g["probs"].keys()
            for k in a["probs"]:
                assert abs(a["probs"][k] - g["probs"][k]) <= TOL
        else:
            assert len(a["probs"]) == len(g["probs"])
            for p, q in zip(a["probs"], g["probs"]):
                assert abs(p - q) <= TOL
            assert abs(a["mean"] - g["mean"]) <= TOL
        assert abs(a["slot_mass"] - g["slot_mass"]) <= TOL
        assert a["n_rotations"] == g["n_rotations"]
        assert a["head"] == g["head"] and a.get("head_format") == g.get("head_format")


def test_golden_h0(qwen_h0):
    ours = json.loads(qwen_h0.decide_json(json.dumps(REQUEST)))
    assert_golden(ours, GOLDEN["h0"])
    assert all(a["head"] is None for a in ours["answers"].values())


def test_golden_h2(qwen_h2):
    ours = json.loads(qwen_h2.decide_json(json.dumps(REQUEST)))
    assert_golden(ours, GOLDEN["h2"])
    for a in ours["answers"].values():
        assert a["head"] is not None and a["head_format"] == "*"


def test_decide_models_and_dicts(qwen_h0):
    d = qwen_h0.decide("The kettle is boiling and steam fills the room.", {
        "hot": Binary(instructions="Is the water hot?"),
        "room": {"type": "choice", "instructions": "Which room is this most likely?",
                 "options": ["kitchen", "garage", "bedroom"]},
        "loud": Score(instructions="How loud is it, from 1 (silent) to 5 (very loud)?", levels=5),
    })
    assert d["hot"].top is True and d["hot"].p_true > 0.5
    assert d["room"].top == "kitchen" and abs(sum(d["room"].probs.values()) - 1) < 1e-9
    assert len(d["loud"].probs) == 5 and 1 <= d["loud"].mean <= 5
    assert d["hot"].n_rotations == 2 and d["room"].n_rotations == 3
    assert not d.truncated


def test_deterministic(qwen_h0):
    a = qwen_h0.decide_json(json.dumps(REQUEST))
    b = qwen_h0.decide_json(json.dumps(REQUEST))
    strip = lambda s: {k: v for k, v in json.loads(s).items() if k != "timing_ms"}  # noqa: E731
    assert strip(a) == strip(b)


def test_head_by_format(qwen_h2):
    """A format with its own head uses it; any other format falls back to "*"."""
    d = qwen_h2.decide("Water boils at 100 C at sea level.", {
        "s": Binary(instructions="Is this statement true?", format="stance"),
        "q": Binary(instructions="Is this statement true?", format="trivia")})
    assert d["s"].format == "stance" and d["s"].head_format == "stance"
    assert d["q"].format == "trivia" and d["q"].head_format == "*"


@pytest.mark.parametrize("request_json, message", [
    ("{", "not valid JSON"),
    ('{"state": "s", "questions": {}}', "non-empty object"),
    ('{"schema": 2, "state": "s", "questions": {"q": {"type": "bool", "instructions": "x"}}}',
     '"schema" must be 1'),
    ('{"state": "s", "questions": {"q": {"type": "bool", "instructions": "x", "levels": 3}}}',
     "no \\\"options\\\" or \\\"levels\\\""),
    ('{"state": "s", "questions": {"q": {"type": "score", "instructions": "x", "levels": 12}}}',
     "from 2 to 9"),
    ('{"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", '
     '"options": {"a": "A"}}}}', "2 to 26 entries"),
    ('{"state": "s", "questions": {"q": {"type": "bool", "instructions": "x", "fmt": "y"}}}',
     'unknown field \\"fmt\\"'),
])
def test_request_errors(qwen_h0, request_json, message):
    response = json.loads(qwen_h0.decide_json(request_json))
    assert message.replace("\\", "") in response["error"]


def test_invalid_question_rejected_before_c(qwen_h0):
    with pytest.raises(ValidationError):
        qwen_h0.decide("s", {"bad": {"type": "bool", "instructions": "z", "options": {"a": "A"}}})


def test_over_budget_refused():
    """A state that fits max_state_tokens but leaves no room in n_ctx for the questions is
    refused before the engine runs, and the handle stays usable."""
    need(QWEN)
    with Engine.load("qwen3-4b-q8", heads=False, model_path=QWEN, n_ctx=4096,
                     max_state_tokens=4090) as engine:
        with pytest.raises(JudglyError, match="more than n_ctx 4096"):
            engine.decide("word " * 5000, {"q": Binary(instructions="Is this repetitive?")})
        d = engine.decide("A short state.", {"q": Binary(instructions="Is this short?")})
        assert 0.0 <= d["q"].p_true <= 1.0


def test_truncation():
    need(QWEN)
    with Engine.load("qwen3-4b-q8", heads=False, model_path=QWEN, max_state_tokens=16,
                     n_ctx=4096) as engine:
        d = engine.decide("word " * 200, {"q": Binary(instructions="Is this repetitive?")})
        assert d.truncated and d.tokens.state < 60


async def test_adecide(qwen_h0):
    qs = {"q": Binary(instructions="Is the sky blue on a clear day?")}
    results = await asyncio.gather(*(qwen_h0.adecide("A clear summer day.", qs) for _ in range(3)))
    assert len({r["q"].p_true for r in results}) == 1


def test_closed_engine():
    need(QWEN)
    engine = Engine.load("qwen3-4b-q8", heads=False, model_path=QWEN, n_ctx=4096)
    engine.close()
    with pytest.raises(JudglyError, match="closed"):
        engine.decide("s", {"q": Binary(instructions="x")})
