"""Write the fixed training sample (PROTOCOL.md): general 500 per question type and stance 500,
from judgly's fit-tier train split, the first by the SHA-256 of the item id.

    uv run python sample.py JUDGLY_ROOT > sample.json
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])


def first(rows, n):
    return sorted(rows, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())[:n]


def main():
    out = {}
    for fmt in ("general", "stance"):
        train = [r for r in map(json.loads, open(ROOT / "data" / "tiers" / fmt / "fitdev.jsonl")) if r["split"] == "train"]
        types = ("choice", "bool", "score") if fmt == "general" else ("choice",)
        out[fmt] = [r["id"] for t in types for r in first([x for x in train if x["type"] == t], 500)]
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
