"""How well calibrated is judgly on YOUR task? Evaluate it on your own labelled examples.

    uv run python examples/own_calibration.py [examples.jsonl] [--out predictions.tsv]

Published calibration numbers describe public benchmarks; the ones that matter are on your own
data. Label a few hundred real cases, run this, and read the reliability table: when the model
says 0.8, is it right about 80% of the time on your cases?

Input: one JSON object per line with "id", "state", "instructions", "options" (key to text)
and "label" (the correct option key), plus optional "format". This is the example schema of
the C tools, so the same file can later fit a head of your own (docs/calibration.md). The
default file, examples/data/support_tickets.jsonl, holds 40 made-up tickets: far too few for
tight intervals, but enough to show the output.

Metrics are defined as in the pipeline's s1-eval: accuracy of the top option, log loss of the
correct option, ECE over 10 equal-width bins of the top probability, each with a 95% bootstrap
interval from 1,000 resamples of the items.

Environment: JUDGLY_PACK (default gemma4-12b-q8), JUDGLY_MODEL_DIR (see stance_check.py).
"""

import argparse
import json
import math
import os
import random

from judgly import Choice, Engine, HeadsNotAvailable

PACK = os.environ.get("JUDGLY_PACK", "gemma4-12b-q8")
HERE = os.path.dirname(os.path.abspath(__file__))
BINS = 10


def load() -> Engine:
    try:
        return Engine.load(PACK)
    except HeadsNotAvailable:
        print(f"note: pack {PACK!r} has no head files; evaluating raw probabilities\n")
        return Engine.load(PACK, heads=False)


def metrics(rows: list[tuple[float, bool, float]]) -> dict[str, float]:
    """rows: (top probability, top is correct, probability of the correct option)."""
    n = len(rows)
    conf, hit, count = [0.0] * BINS, [0.0] * BINS, [0] * BINS
    for p_top, correct, _ in rows:
        b = min(int(p_top * BINS), BINS - 1)
        conf[b] += p_top
        hit[b] += correct
        count[b] += 1
    ece = sum(abs(hit[b] - conf[b]) for b in range(BINS)) / n
    return {"accuracy": sum(r[1] for r in rows) / n,
            "log_loss": -sum(math.log(max(r[2], 1e-300)) for r in rows) / n,
            "ece": ece}


def bootstrap(rows, resamples=1000, seed=20260926):
    rng = random.Random(seed)
    draws = [metrics([rows[rng.randrange(len(rows))] for _ in rows]) for _ in range(resamples)]
    out = {}
    for k in draws[0]:
        v = sorted(d[k] for d in draws)
        out[k] = (v[int(0.025 * (resamples - 1))], v[int(0.975 * (resamples - 1))])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("examples", nargs="?", default=os.path.join(HERE, "data", "support_tickets.jsonl"))
    ap.add_argument("--out", help="write one line per item: id, label, top, p_top, p_label")
    args = ap.parse_args()
    items = [json.loads(line) for line in open(args.examples, encoding="utf-8") if line.strip()]

    rows, lines = [], []
    with load() as engine:
        for x in items:
            q = Choice(instructions=x["instructions"], options=x["options"], format=x.get("format"))
            a = engine.decide(x["state"], {"q": q})["q"]
            rows.append((a.probs[a.top], a.top == x["label"], a.probs[x["label"]]))
            lines.append(f"{x['id']}\t{x['label']}\t{a.top}\t{a.probs[a.top]:.4f}\t"
                         f"{a.probs[x['label']]:.4f}")

    m, ci = metrics(rows), bootstrap(rows)
    print(f"{len(rows)} items from {args.examples}")
    for k in ("accuracy", "log_loss", "ece"):
        print(f"  {k:<9} {m[k]:.3f}  [{ci[k][0]:.3f}, {ci[k][1]:.3f}]")

    print("\nreliability: top probability bin, items, mean confidence, observed accuracy")
    for b in range(BINS):
        in_bin = [r for r in rows if min(int(r[0] * BINS), BINS - 1) == b]
        if in_bin:
            print(f"  {b / BINS:.1f}-{(b + 1) / BINS:.1f}  {len(in_bin):4d}  "
                  f"{sum(r[0] for r in in_bin) / len(in_bin):.2f}  "
                  f"{sum(r[1] for r in in_bin) / len(in_bin):.2f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("id\tlabel\ttop\tp_top\tp_label\n" + "\n".join(lines) + "\n")
        print(f"\nper-item predictions in {args.out}")


if __name__ == "__main__":
    main()
