"""Write tests/fixtures/golden-qwen3-4b-q8.json: the responses of the built-in qwen3-4b-q8 pack
to one fixed request, with no calibration (h0), with the shipped H2 heads (h2) and with the
shipped per-type temperatures (temperature). tests/test_engine.py and tests/test_temperature.py
compare the live engine with these values within 1e-6.

    JUDGLY_MODEL_DIR=/path/to/models uv run python scripts/make_golden.py

Regenerate only when a change to the engine, template, settings or heads is meant to change the
numbers.
"""

import json
import os
import sys
from pathlib import Path

from judgly import Engine

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "golden-qwen3-4b-q8.json"

STATE = ("A 58-year-old man reports crushing chest pain radiating to the left arm for the past "
         "40 minutes, with sweating and nausea. He has a history of hypertension and smoking.")
REQUEST = {
    "schema": 1,
    "state": STATE,
    "questions": {
        "urgency": {"type": "choice", "instructions": "How urgently does he need care?",
                    "options": {"now": "Emergency care immediately",
                                "today": "See a doctor today",
                                "week": "Book an appointment this week",
                                "none": "No care needed"}},
        "cardiac": {"type": "bool", "instructions": "Is a heart attack a plausible cause?"},
        "severity": {"type": "score", "instructions": "How severe is the situation, from 1 "
                     "(trivial) to 5 (life-threatening)?", "levels": 5},
    },
}


def respond(calibration: str) -> dict:
    with Engine.load("qwen3-4b-q8", calibration=calibration) as engine:
        out = json.loads(engine.decide_json(json.dumps(REQUEST)))
    out.pop("timing_ms", None)
    return out


def main() -> int:
    if not os.environ.get("JUDGLY_MODEL_DIR"):
        print("set JUDGLY_MODEL_DIR to the directory holding the Qwen3-4B GGUF file",
              file=sys.stderr)
        return 2
    golden = {"pack": "qwen3-4b-q8", "request": REQUEST, "h0": respond("raw"),
              "h2": respond("h2"), "temperature": respond("temperature")}
    OUT.write_text(json.dumps(golden, indent=1) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
