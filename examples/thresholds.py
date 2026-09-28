"""Choosing a confidence threshold: answer automatically when judgly is confident, send the
rest to a person.

    uv run python examples/thresholds.py [examples.jsonl] [--target 0.9]

For each threshold t, "answered" is the share of items whose top probability is above t and
"accuracy" is how often those answers are right (selective accuracy). A higher threshold
answers fewer items more accurately. The script picks the lowest threshold whose accuracy is
at least --target with the lower end of a 95% Wilson interval, so that a lucky small sample
does not set it.

Use labelled examples from your own workflow (same schema as own_calibration.py), and choose
the threshold on one set and check it on another. The default file holds 40 made-up tickets:
enough to show the output, far too few to choose a real threshold.

Environment: JUDGLY_PACK (default gemma4-12b-q8), JUDGLY_MODEL_DIR (see stance_check.py).
"""

import argparse
import json
import math
import os

from judgly import Choice, Engine, HeadsNotAvailable

PACK = os.environ.get("JUDGLY_PACK", "gemma4-12b-q8")
HERE = os.path.dirname(os.path.abspath(__file__))
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def load() -> Engine:
    try:
        return Engine.load(PACK)
    except HeadsNotAvailable:
        print(f"note: pack {PACK!r} has no head files; using raw probabilities\n")
        return Engine.load(PACK, heads=False)


def wilson_low(hits: int, n: int, z: float = 1.959964) -> float:
    if n == 0:
        return 0.0
    p = hits / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / (1 + z * z / n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("examples", nargs="?", default=os.path.join(HERE, "data", "support_tickets.jsonl"))
    ap.add_argument("--target", type=float, default=0.9, help="accuracy wanted on answered items")
    args = ap.parse_args()
    items = [json.loads(line) for line in open(args.examples, encoding="utf-8") if line.strip()]

    scored = []   # (top probability, correct)
    with load() as engine:
        for x in items:
            q = Choice(instructions=x["instructions"], options=x["options"], format=x.get("format"))
            a = engine.decide(x["state"], {"q": q})["q"]
            scored.append((a.probs[a.top], a.top == x["label"]))

    print(f"{len(scored)} items; target accuracy {args.target}")
    print("threshold  answered  accuracy  95% lower bound")
    chosen = None
    for t in THRESHOLDS:
        kept = [c for p, c in scored if p > t]   # strictly above, as in the published tables
        n, hits = len(kept), sum(kept)
        acc = hits / n if n else float("nan")
        low = wilson_low(hits, n)
        print(f"  {t:.2f}     {n / len(scored):6.0%}    {acc:6.3f}    {low:.3f}")
        if chosen is None and n and low >= args.target:
            chosen = t
    if chosen is None:
        print("\nNo threshold reaches the target with confidence on this data: label more "
              "examples, lower the target, or send everything to review.")
    else:
        print(f"\nLowest threshold meeting the target: {chosen}. Check it on fresh examples.")


if __name__ == "__main__":
    main()
