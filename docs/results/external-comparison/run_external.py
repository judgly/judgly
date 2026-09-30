"""Ask an Ollama decision model every item of judgly's test tiers (PROTOCOL.md). Resumable:
answers are appended to answers/<model>/<format>-<tier>.jsonl and items already answered are
skipped.

    uv run --no-project python run_external.py JUDGLY_ROOT MODEL [MODEL ...]
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(sys.argv[1])
MODELS = sys.argv[2:]
OUT = Path(__file__).parent / "answers"
ORDER = [("general", "confirm"), ("stance", "confirm"), ("general", "final"), ("stance", "final"),
         ("general", "bench"), ("general", "final-flagged"), ("stance", "final-flagged")]
URL = "http://localhost:11434/v1/systemone"


def question(item):
    t = item["type"]
    if t == "choice":
        return {"type": "choice", "instructions": item["instructions"], "criteria": item["options"]}
    if t == "bool":
        return {"type": "noul", "instructions": item["instructions"]}
    return {"type": "score", "instructions": item["instructions"],
            "criteria": [str(i) for i in range(1, item["levels"] + 1)]}


def ask(model, item):
    body = json.dumps({"model": model, "state": item["state"], "questions": {"q": question(item)}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.loads(r.read())
        err = None
    except urllib.error.HTTPError as e:
        resp, err = None, f"HTTP {e.code}: {e.read()[:500].decode(errors='replace')}"
    except Exception as e:  # noqa: BLE001
        resp, err = None, f"{type(e).__name__}: {e}"
    return resp, err, time.perf_counter() - t0


def main():
    for model in MODELS:
        for fmt, tier in ORDER:
            src = ROOT / "data" / "tiers" / fmt / f"{tier}.jsonl"
            dst = OUT / model.replace(":", "_") / f"{fmt}-{tier}.jsonl"
            dst.parent.mkdir(parents=True, exist_ok=True)
            done = set()
            if dst.exists():
                done = {json.loads(l)["id"] for l in open(dst) if l.strip()}
            items = [json.loads(l) for l in open(src)]
            todo = [x for x in items if x["id"] not in done]
            print(f"{model} {fmt}/{tier}: {len(items)} items, {len(todo)} to ask", flush=True)
            with open(dst, "a") as f:
                for n, item in enumerate(todo, 1):
                    resp, err, sec = ask(model, item)
                    f.write(json.dumps({"id": item["id"], "response": resp, "error": err,
                                        "seconds": round(sec, 4)}) + "\n")
                    f.flush()
                    if n % 250 == 0:
                        print(f"  {n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
