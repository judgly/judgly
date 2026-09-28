"""judgly quickstart: typed questions about a piece of text, answered with probabilities.

    uv run python examples/quickstart.py

The first run downloads the Qwen3-4B model (4.3 GB) into the Hugging Face cache and checks its
SHA-256. To use a copy you already have, point JUDGLY_MODEL_DIR at the directory holding
Qwen3-4B-Instruct-2507-Q8_0.gguf.

This example uses heads=False on purpose, to show what the engine reads before calibration:
the raw probabilities the model puts on the answer letters (H0). They rank options sensibly but
tend to be overconfident. Drop heads=False to use the pack's shipped calibration heads, as the
README quickstart does.
"""

import asyncio
import json

from judgly import Binary, Choice, Engine, Score

STATE = ("Customer email: I ordered the blue kettle two weeks ago and it still has not "
         "arrived. The tracking page has said 'in transit' for nine days. I would like a "
         "refund if it cannot be delivered this week.")

QUESTIONS = {
    "topic": Choice(instructions="What is the email mainly about?",
                    options={"delivery": "A late or missing delivery",
                             "defect": "A faulty product",
                             "billing": "A billing or payment problem",
                             "other": "Something else"}),
    "refund": Binary(instructions="Does the customer ask for a refund?"),
    "anger": Score(instructions="How upset is the customer, from 1 (calm) to 5 (furious)?",
                   levels=5),
    # Plain dicts work too; a list of options uses each string as its own key.
    "reply": {"type": "choice", "instructions": "Which team should reply?",
              "options": ["shipping", "returns", "sales"]},
}


def main() -> None:
    with Engine.load("qwen3-4b-q8", heads=False) as engine:
        d = engine.decide(STATE, QUESTIONS)
        print(f"topic:  {d['topic'].top}  {d['topic'].probs}")
        print(f"refund: p_true = {d['refund'].p_true:.3f}")
        print(f"anger:  mean level {d['anger'].mean:.2f}, probs "
              f"{[round(p, 3) for p in d['anger'].probs]}")
        print(f"reply:  {d['reply'].top}  (rotation spread {d['reply'].rotation_spread:.3f}, "
              f"slot mass {d['reply'].slot_mass:.3f})")
        print(f"state tokens {d.tokens.state}, {d.timing_ms.state + d.timing_ms.questions:.0f} ms")

        # The same decision through the JSON interface the C library speaks.
        request = {"schema": 1, "state": STATE,
                   "questions": {"refund": {"type": "bool",
                                            "instructions": "Does the customer ask for a refund?"}}}
        response = json.loads(engine.decide_json(json.dumps(request)))
        print("decide_json:", response["answers"]["refund"]["p_true"])

        # From async code: calls run in a worker thread, one at a time per engine.
        async def several():
            states = ["The parcel arrived today, thanks!", STATE]
            return await asyncio.gather(*(engine.adecide(s, {"refund": QUESTIONS["refund"]})
                                          for s in states))
        for result in asyncio.run(several()):
            print("adecide:", round(result["refund"].p_true, 3))


if __name__ == "__main__":
    main()
