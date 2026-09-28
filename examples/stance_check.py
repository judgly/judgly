"""Stance: does a piece of evidence support a claim, contradict it, or have no bearing on it?

    uv run python examples/stance_check.py

The "stance" format has its own calibration head, fitted on MNLI, SNLI, WANLI, VitaminC, FEVER
and SciNLI. On the fresh final tier (Check-COVID, n = 1,343) its ECE was 0.048 for Gemma 4 12B
and 0.052 for Qwen3-4B, around the bar of 0.05, but it missed the dev-tier bar in both packs and
was worse on other health claims (see the README results), so read its probabilities with
caution. Keep the state in the shape the head was fitted on:
"Claim: ...\\n\\nEvidence: ...". The option keys below are the ones the head knows; the option
texts may be worded differently.

Environment:
    JUDGLY_PACK       pack name or directory (default gemma4-12b-q8)
    JUDGLY_MODEL_DIR  directory that already holds the pack's GGUF file (skips the download)

For a pack without head files, this falls back to raw letter probabilities (H0) and says so;
those rank the options but are overconfident, so read them as a ranking, not as probabilities.
"""

import os

from judgly import Choice, Engine, HeadsNotAvailable

PACK = os.environ.get("JUDGLY_PACK", "gemma4-12b-q8")

STANCE = Choice(
    format="stance",
    instructions="How does the evidence bear on the claim?",
    options={"supports": "The evidence supports the claim",
             "contradicts": "The evidence contradicts the claim",
             "no_bearing": "The evidence has no bearing on the claim"},
)

# Made-up pairs for illustration; they are not from any dataset.
PAIRS = [
    ("Drinking coffee raises blood pressure for a short time.",
     "In a crossover study of 40 adults, systolic pressure rose by a mean of 8 mmHg in the hour "
     "after 200 mg of caffeine and returned to baseline within three hours."),
    ("Vitamin C prevents the common cold.",
     "A pooled analysis of 29 trials found that regular vitamin C did not reduce how often "
     "people in the general population caught colds."),
    ("Walking after meals lowers blood sugar.",
     "The survey recorded the number of hours participants spent watching television each "
     "week, broken down by age group."),
]


def load() -> Engine:
    try:
        return Engine.load(PACK)
    except HeadsNotAvailable:
        print(f"note: pack {PACK!r} has no head files; showing raw probabilities\n")
        return Engine.load(PACK, heads=False)


def main() -> None:
    with load() as engine:
        for claim, evidence in PAIRS:
            state = f"Claim: {claim}\n\nEvidence: {evidence}"
            answer = engine.decide(state, {"stance": STANCE})["stance"]
            probs = "  ".join(f"{k} {p:.2f}" for k, p in answer.probs.items())
            print(f"{claim}\n  -> {answer.top:<12} {probs}  (head: {answer.head_format})\n")


if __name__ == "__main__":
    main()
