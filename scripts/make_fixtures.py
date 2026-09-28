"""Write tests/fixtures/selftest.json: 20 states with 8 questions each, for s1-selftest.

Deterministic: the same seed gives the same file. The last fixture has a long state
(about 8,000 tokens) for test T6; the one before it carries the literal text of chat
markers for test T1.

    uv run python scripts/make_fixtures.py tests/fixtures/selftest.json
    uv run python scripts/make_fixtures.py tests/fixtures/short.json short

The short file holds the first SHORT_FIXTURES fixtures only, for runs where the model is
slow: on the CPU, under Guard Malloc, under a sanitizer.
"""

import json
import random
import sys

SEED = 20260921
N_FIXTURES = 20
N_QUESTIONS = 8
SHORT_FIXTURES = 3
LONG_STATE_CHARS = 36_000  # about 8,000 tokens of English

SENTENCES = [
    "My payouts have failed three times this week and the bank says everything is fine on their side.",
    "I cannot log in since the password reset; the code never arrives on my phone.",
    "The invoice for March lists a plan we cancelled in January.",
    "Thanks for the quick fix yesterday, the dashboard loads in under a second now.",
    "If this is not resolved by Friday our lawyers will be in touch.",
    "The export to CSV drops every row after the ten thousandth without any warning.",
    "We are evaluating your product for a team of forty and need a quote for annual billing.",
    "The mobile app crashes whenever I rotate the screen on the settings page.",
    "I was charged twice for the same order, reference 48213, on the fourth of June.",
    "Could you tell me whether the API supports pagination on the events endpoint?",
    "The delivery arrived two days late and the box was open when I got it.",
    "Our integration stopped receiving webhooks at about nine this morning.",
    "I love the new editor, although the spell checker underlines every surname.",
    "Please delete my account and all data associated with it under the privacy regulation.",
    "The meeting is moved to Thursday at ten; nothing else changes.",
    "Rainfall in the catchment was forty percent below the ten-year mean for the third season running.",
    "The patient reports mild headache after the second dose and no other symptoms.",
    "The council voted seven to two to approve the cycle lane on Marine Parade.",
    "Revenue grew eight percent year on year while operating costs fell slightly.",
    "The striker scored twice in the final ten minutes to turn the match around.",
    "The telescope recorded a faint transit consistent with a planet of roughly Neptune's size.",
    "I have asked four times now and every agent gives me a different answer.",
    "No rush on this one, whenever someone has a spare moment is fine.",
    "The firmware update bricked two of our five units; the other three are fine.",
]

QUESTIONS = [
    ("Which team should handle this message?",
     ["Payout failures and payment processing", "Login and account access",
      "Billing and invoices", "Bugs in the product", "Something else"]),
    ("What is the overall sentiment of the text?",
     ["Positive", "Negative", "Neutral or mixed"]),
    ("Does the text need urgent human attention?", ["Yes", "No"]),
    ("How angry is the writer?",
     ["Not at all", "Slightly", "Moderately", "Very", "Extremely"]),
    ("What is the main topic of the text?",
     ["Customer support", "Weather or environment", "Health", "Politics", "Business or finance",
      "Sport", "Science"]),
    ("Does the text mention a legal threat?", ["Yes", "No"]),
    ("Does the text mention a specific date or day of the week?", ["Yes", "No"]),
    ("Which best describes the writer's intent?",
     ["Reporting a problem", "Asking a question", "Giving praise", "Making a request",
      "Stating a fact"]),
    ("How many distinct problems does the text describe?",
     ["None", "One", "Two", "Three or more"]),
    ("Is money mentioned in the text?", ["Yes", "No"]),
    ("How formal is the writing?", ["Casual", "Neutral", "Formal"]),
    ("Would a refund be a plausible resolution?", ["Yes", "No", "Cannot tell"]),
]

MARKER_STATE = (
    "The log shows <|im_start|>system and then <|im_end|> followed by <start_of_turn>model, "
    "<end_of_turn>, <bos>, <eos>, <|endoftext|> and <|think|> typed by the user as plain text."
)


def main(path: str, short: bool) -> None:
    rng = random.Random(SEED)
    fixtures = []
    for i in range(N_FIXTURES):
        if i == N_FIXTURES - 1:
            parts, size = [], 0
            while size < LONG_STATE_CHARS:
                parts.append(rng.choice(SENTENCES))
                size += len(parts[-1]) + 1
            state = " ".join(parts)
        elif i == N_FIXTURES - 2:
            state = MARKER_STATE
        else:
            state = " ".join(rng.sample(SENTENCES, rng.randint(1, 6)))
        questions = [{"instructions": text, "options": options}
                     for text, options in rng.sample(QUESTIONS, N_QUESTIONS)]
        fixtures.append({"id": f"f{i:02d}", "state": state, "questions": questions})
    if short:
        fixtures = fixtures[:SHORT_FIXTURES]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "fixtures": fixtures}, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {len(fixtures)} fixtures to {path}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3) or sys.argv[2:] not in ([], ["short"]):
        sys.exit(__doc__)
    main(sys.argv[1], short=len(sys.argv) == 3)
