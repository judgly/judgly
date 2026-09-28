"""Many questions about one text: the text is read once, every question branches from it.

    uv run python examples/many_questions.py

Each question sees the state and itself, never its siblings, so adding a question does not
change the answers to the others (the self-test T5 checks this). The cost of a request is one
pass over the state plus a short pass per question and rotation.

Environment: JUDGLY_PACK (default gemma4-12b-q8), JUDGLY_MODEL_DIR (see stance_check.py).

With JUDGLY_PACK=qwen3-4b-q8, read the 'stars' rating with care: rating scales are that pack's
weakest question type (for example sentence difficulty on the fresh final tier: accuracy 0.287
raw and 0.253 with the head, n = 874; see its model card).
"""

import os

from judgly import Binary, Choice, Engine, HeadsNotAvailable, Score

PACK = os.environ.get("JUDGLY_PACK", "gemma4-12b-q8")

REVIEW = ("We stayed four nights in March. The room was spotless and the bed comfortable, but "
          "the street noise kept us awake until 2 am on the weekend. Breakfast was generous. "
          "Staff at the desk were friendly, although check-in took forty minutes because the "
          "card machine was broken. The pool was closed for repairs, which nobody told us "
          "when we booked. Great location for the old town.")

# A checklist: a few yes/no questions, a rating and two choices.
ASPECTS = ["cleanliness", "noise", "breakfast", "staff", "check-in", "pool", "location"]
QUESTIONS = {f"mentions_{a}": Binary(instructions=f"Does the review mention the {a}?")
             for a in ASPECTS}
QUESTIONS |= {
    "negative_noise": Binary(instructions="Does the reviewer complain about noise?"),
    "stars": Score(instructions="How many stars would this reviewer give, from 1 to 5?",
                   levels=5),
    "recommend": Choice(instructions="Would the reviewer recommend the hotel?",
                        options={"yes": "Yes", "mixed": "With reservations", "no": "No"}),
    "main_problem": Choice(instructions="What was the biggest problem during the stay?",
                           options=["noise", "slow check-in", "closed pool", "none"]),
}


def load() -> Engine:
    try:
        return Engine.load(PACK)
    except HeadsNotAvailable:
        print(f"note: pack {PACK!r} has no head files; showing raw probabilities\n")
        return Engine.load(PACK, heads=False)


def main() -> None:
    if PACK == "qwen3-4b-q8":
        print("note: rating scales are the weakest question type of the qwen3-4b-q8 pack "
              "(see its model card)\n")
    with load() as engine:
        d = engine.decide(REVIEW, QUESTIONS)
        for qid, a in d.answers.items():
            if a.type == "bool":
                print(f"{qid:<22} p(true) {a.p_true:.2f}")
            elif a.type == "score":
                print(f"{qid:<22} mean {a.mean:.2f}  most likely {a.top}")
            else:
                print(f"{qid:<22} {a.top}  ({a.probs[a.top]:.2f})")
        print(f"\n{len(QUESTIONS)} questions; state {d.tokens.state} tokens read once in "
              f"{d.timing_ms.state:.0f} ms, questions {d.tokens.questions} tokens in "
              f"{d.timing_ms.questions:.0f} ms")

        # Isolation: ask one question alone and compare with its answer in the full request.
        alone = engine.decide(REVIEW, {"stars": QUESTIONS["stars"]})["stars"]
        diff = max(abs(x - y) for x, y in zip(alone.probs, d["stars"].probs))
        print(f"'stars' alone vs with {len(QUESTIONS) - 1} siblings: max difference {diff:.4f}")


if __name__ == "__main__":
    main()
