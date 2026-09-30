"""Ask the external models the fixed training sample (PROTOCOL.md), with the comparison's frozen
request mapping (run_external.ask). Resumable; answers go to answers/<model>/<format>-train.jsonl.

    uv run --no-project python run_calibration.py JUDGLY_ROOT MODEL [MODEL ...]
"""

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = Path(sys.argv[1])
MODELS = sys.argv[2:]


def runner():
    argv = sys.argv
    sys.argv = [argv[0], str(ROOT)]  # the frozen runner reads its arguments at import
    spec = importlib.util.spec_from_file_location("run_external", HERE.parent / "run_external.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


def main():
    ask = runner().ask
    sample = json.load(open(HERE / "sample.json"))
    for model in MODELS:
        for fmt, ids in sample.items():
            rows = {r["id"]: r for r in map(json.loads, open(ROOT / "data" / "tiers" / fmt / "fitdev.jsonl"))}
            dst = HERE / "answers" / model.replace(":", "_") / f"{fmt}-train.jsonl"
            dst.parent.mkdir(parents=True, exist_ok=True)
            done = {json.loads(line)["id"] for line in open(dst)} if dst.exists() else set()
            todo = [i for i in ids if i not in done]
            print(f"{model} {fmt}/train: {len(ids)} items, {len(todo)} to ask", flush=True)
            with open(dst, "a") as f:
                for n, i in enumerate(todo, 1):
                    resp, err, sec = ask(model, rows[i])
                    f.write(json.dumps({"id": i, "response": resp, "error": err, "seconds": round(sec, 4)}) + "\n")
                    f.flush()
                    if n % 250 == 0:
                        print(f"  {n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
