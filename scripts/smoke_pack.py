"""Load a built pack with its heads in judgly.Engine and answer a stance question and a
general question, checking that each went through the intended head.

    JUDGLY_MODEL_DIR=... uv run python scripts/smoke_pack.py PACK_DIR
"""

import sys

from judgly import Engine
from judgly.models import Binary, Choice

STATE = ("Claim: Regular handwashing reduces the spread of respiratory infections.\n\n"
         "Evidence: In a cluster-randomised trial in 40 schools, pupils taught to wash their hands "
         "five times a day had 31% fewer absences from respiratory illness than pupils in control schools.")


def main(pack_dir: str) -> int:
    with Engine.load(pack_dir, heads=True) as engine:
        d = engine.decide(STATE, {
            "stance": Choice(format="stance", instructions="How does the evidence bear on the claim?",
                             options={"supports": "The evidence supports the claim",
                                      "contradicts": "The evidence contradicts the claim",
                                      "no_bearing": "The evidence has no bearing on the claim"}),
            "general": Binary(instructions="The evidence comes from a randomised trial."),
        })
    stance, general = d.answers["stance"], d.answers["general"]
    print(f"stance: head {stance.head_format}, top {stance.top}, probs "
          + ", ".join(f"{k} {v:.3f}" for k, v in stance.probs.items()))
    print(f"general: head {general.head_format}, p_true {general.p_true:.3f}")
    ok = (stance.head_format == "stance" and general.head_format == "*"
          and abs(sum(stance.probs.values()) - 1) < 1e-6 and 0 <= general.p_true <= 1)
    print("PASS smoke pack" if ok else "FAIL smoke pack")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
