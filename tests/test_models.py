"""The pydantic request and answer models: validation and JSON round trips."""

import json

import pytest
from pydantic import TypeAdapter, ValidationError

from judgly import Binary, Choice, Decision, Question, Request, Score

QUESTIONS = TypeAdapter(dict[str, Question])


def test_discriminated_from_dicts():
    qs = QUESTIONS.validate_python({
        "a": {"type": "choice", "instructions": "Pick", "options": {"x": "X", "y": "Y"}},
        "b": {"type": "bool", "instructions": "True?"},
        "c": {"type": "score", "instructions": "How much?", "levels": 5, "format": "stance"},
    })
    assert isinstance(qs["a"], Choice) and isinstance(qs["b"], Binary)
    assert isinstance(qs["c"], Score) and qs["c"].format == "stance"


def test_request_round_trip():
    req = Request(state="s", questions={
        "a": Choice(instructions="Pick", options=["red", "green"]),
        "b": Binary(instructions="True?", format="stance"),
        "c": Score(instructions="How much?", levels=3)})
    text = req.to_json()
    raw = json.loads(text)
    assert raw["schema"] == 1
    assert raw["questions"]["a"] == {"instructions": "Pick", "type": "choice",
                                     "options": {"red": "red", "green": "green"}}
    assert "format" not in raw["questions"]["a"]
    assert Request.model_validate_json(text) == req


@pytest.mark.parametrize("bad", [
    {"type": "choice", "instructions": "x", "options": {"a": "A"}},             # one option
    {"type": "choice", "instructions": "x", "options": {f"o{i}": "t" for i in range(27)}},
    {"type": "choice", "instructions": "x", "options": ["a", "a"]},             # duplicates
    {"type": "score", "instructions": "x", "levels": 1},
    {"type": "score", "instructions": "x", "levels": 10},
    {"type": "bool", "instructions": "x", "options": {"a": "A", "b": "B"}},     # extra field
    {"type": "maybe", "instructions": "x"},
    {"type": "bool"},                                                            # no instructions
])
def test_invalid_questions(bad):
    with pytest.raises(ValidationError):
        QUESTIONS.validate_python({"q": bad})


def test_request_needs_questions():
    with pytest.raises(ValidationError):
        Request(state="s", questions={})
    with pytest.raises(ValidationError):
        Request.model_validate({"schema": 2, "state": "s",
                                "questions": {"b": {"type": "bool", "instructions": "x"}}})


def test_decision_round_trip():
    common = {"slot_mass": 0.99, "rotation_spread": 0.01, "n_rotations": 2, "format": None,
              "head": None, "head_format": None}
    raw = {"schema": 1,
           "answers": {"a": {"type": "choice", "probs": {"x": 0.7, "y": 0.3}, "top": "x", **common},
                       "b": {"type": "bool", "p_true": 0.2, "top": False, **common},
                       "c": {"type": "score", "probs": [0.5, 0.5], "mean": 1.5, "top": 1, **common}},
           "model_sha256": "0" * 64, "template_sha256": "1" * 64, "truncated": False,
           "tokens": {"state": 3, "questions": 9}, "timing_ms": {"state": 1.0, "questions": 2.0}}
    d = Decision.model_validate(raw)
    assert d["b"].p_true == 0.2 and d["c"].top == 1 and d["a"].probs["x"] == 0.7
    assert json.loads(d.model_dump_json(by_alias=True)) == raw
