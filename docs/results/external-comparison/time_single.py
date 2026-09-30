"""Time one-question requests, one at a time, for judgly and the Ollama decision models on the
same 100 items (50 from each untouched confirm tier, the first by a hash of their id).
One warm-up request per system is sent first and not counted. Hardware and versions are
recorded in the output.

    JUDGLY_MODEL_DIR=... uv run python time_single.py JUDGLY_ROOT OUT.json
"""

import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])


def pick(fmt, n=50):
    rows = [json.loads(l) for l in open(ROOT / "data" / "tiers" / fmt / "confirm.jsonl")]
    rows.sort(key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
    return rows[:n]


ITEMS = pick("general") + pick("stance")


def judgly_question(item):
    from judgly import Binary, Choice, Score
    fmt = "stance" if item["format"] == "stance" else None
    if item["type"] == "choice":
        return Choice(format=fmt, instructions=item["instructions"], options=item["options"])
    if item["type"] == "bool":
        return Binary(format=fmt, instructions=item["instructions"])
    return Score(format=fmt, instructions=item["instructions"], levels=item["levels"])


def time_judgly(pack):
    from judgly import Engine
    with Engine.load(pack) as engine:
        engine.decide(ITEMS[0]["state"], {"q": judgly_question(ITEMS[0])})  # warm-up
        secs = []
        for item in ITEMS:
            q = {"q": judgly_question(item)}
            t0 = time.perf_counter()
            engine.decide(item["state"], q)
            secs.append(time.perf_counter() - t0)
    return secs


def time_ollama(model):
    import run_external  # the frozen runner's request mapping
    run_external.ask(model, ITEMS[0])  # warm-up (loads the model)
    secs, errors = [], 0
    for item in ITEMS:
        resp, err, sec = run_external.ask(model, item)
        if err:
            errors += 1
            continue
        secs.append(sec)
    return secs, errors


def summary(secs):
    s = sorted(secs)
    return {"n": len(s), "median_s": round(statistics.median(s), 4),
            "p95_s": round(s[min(len(s) - 1, int(0.95 * len(s)))], 4), "mean_s": round(statistics.mean(s), 4)}


def main():
    sys.argv = [sys.argv[0], str(ROOT)]  # run_external reads argv at import
    sys.path.insert(0, str(Path(__file__).parent))
    out = {"items": [i["id"] for i in ITEMS],
           "machine": {"platform": platform.platform(), "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
                       "memory_gb": int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) // 2**30},
           "ollama": subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip(),
           "method": "one question per request, requests sent one at a time, one warm-up request per system not counted; judgly in-process through its Python API (pack default calibration), Ollama models through http://localhost:11434/v1/systemone",
           "systems": {}}
    for pack in ("gemma4-12b-q8", "qwen3-4b-q8"):
        out["systems"][f"judgly {pack}"] = summary(time_judgly(pack))
        print(pack, out["systems"][f"judgly {pack}"], flush=True)
    for model in ("nimble:9b", "tev1:4b", "tev1:0.8b"):
        secs, errors = time_ollama(model)
        out["systems"][model] = {**summary(secs), "refused": errors}
        print(model, out["systems"][model], flush=True)
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
