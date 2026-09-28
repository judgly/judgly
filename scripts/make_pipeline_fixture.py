"""Write tests/fixtures/pipeline.jsonl, the examples s1-selftest's T8 reads (cached features
against the live engine; it uses the first 24).

    uv run --group pipeline python scripts/make_pipeline_fixture.py tests/fixtures/pipeline.jsonl

The fixture is drawn from judgly's own generated
tasks (scripts/prep_tiers.py), so the committed file holds no third-party text. Every
question type is present. Deterministic.
"""

import json
import sys
from pathlib import Path

import prep_tiers

N = 48


def main(out: Path) -> None:
    src = prep_tiers.Source("generated", "general", "fit", "generated", "mixed", prep_tiers.Spec(None))
    src.items = list(prep_tiers.generated(N))
    splits = ["train", "validation", "test", "heldout"]
    with open(out, "w", encoding="utf-8") as f:
        for i, it in enumerate(src.items):
            ex = prep_tiers.render_general(src, it, splits[i % 4], src.items)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"wrote {N} examples to {out}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
