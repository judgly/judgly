"""Using judgly from asyncio code, for example inside a web handler.

    uv run python examples/async_usage.py

``Engine.adecide`` runs ``decide`` in a worker thread so the event loop stays free. One engine
answers one request at a time (the native handle is not re-entrant); concurrent calls wait on
its lock. Load the engine once at start-up and share it: every Engine holds its own copy of the
model in memory.

Environment: JUDGLY_PACK (default gemma4-12b-q8), JUDGLY_MODEL_DIR (see stance_check.py).
"""

import asyncio
import os
import time

from judgly import Binary, Choice, Engine, HeadsNotAvailable

PACK = os.environ.get("JUDGLY_PACK", "gemma4-12b-q8")

QUESTIONS = {
    "urgent": Binary(instructions="Does the sender need an answer today?"),
    "kind": Choice(instructions="What kind of message is this?",
                   options=["question", "complaint", "thanks", "spam"]),
}

INBOX = [
    "Hi, is the shop open on Sunday? We would like to come by this afternoon.",
    "Thank you so much for the quick repair, the bike rides like new.",
    "CONGRATULATIONS! You have been selected for a free cruise. Click here to claim.",
    "The replacement part you sent is the wrong size again. This is the third time.",
]


def load() -> Engine:
    try:
        return Engine.load(PACK)
    except HeadsNotAvailable:
        print(f"note: pack {PACK!r} has no head files; showing raw probabilities\n")
        return Engine.load(PACK, heads=False)


async def triage(engine: Engine, message: str) -> str:
    d = await engine.adecide(message, QUESTIONS)
    return f"{d['kind'].top:<10} urgent {d['urgent'].p_true:.2f}  {message[:50]}"


async def main() -> None:
    engine = await asyncio.to_thread(load)   # loading takes seconds; keep the loop free
    try:
        start = time.perf_counter()
        for line in await asyncio.gather(*(triage(engine, m) for m in INBOX)):
            print(line)
        print(f"{len(INBOX)} messages in {time.perf_counter() - start:.1f} s "
              "(answered one after another by the one engine)")
    finally:
        engine.close()


if __name__ == "__main__":
    asyncio.run(main())
