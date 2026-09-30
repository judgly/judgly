"""Score the external models and judgly on the same items (PROTOCOL.md).

    uv run --no-project --with numpy python score_external.py JUDGLY_ROOT MODEL [MODEL ...]
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
MODELS = sys.argv[2:]
HERE = Path(__file__).parent
EPS = 1e-12
CTX = {"tev1:4b": 2050, "tev1:0.8b": 2050, "nimble:9b": 8194}
TIERS = [("general", "confirm"), ("stance", "confirm"), ("general", "final"), ("stance", "final"),
         ("general", "bench"), ("general", "final-flagged"), ("stance", "final-flagged")]
JUDGLY = {"gemma": "gemma4-12b-q8", "qwen": "qwen3-4b-q8"}
DEFAULT = {("gemma", "general"): "h2", ("gemma", "stance"): "temperature",
           ("qwen", "general"): "temperature", ("qwen", "stance"): "temperature"}
NAMES = ("accuracy", "ece", "brier", "log_loss")
rng = np.random.default_rng(20260929)


def fnv(text):
    h = 0xcbf29ce484222325
    for c in text.encode():
        h = ((h ^ c) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def keys_of(item):
    if item["type"] == "choice":
        return list(item["options"])
    if item["type"] == "bool":
        return ["true", "false"]
    return [str(i) for i in range(1, item["levels"] + 1)]


def gold_index(item):
    k = keys_of(item)
    return k.index(str(item["label"]).lower() if item["type"] == "bool" else str(item["label"]))


def external_probs(item, resp):
    a = resp["answers"]["q"]
    if item["type"] == "choice":
        return np.array([a["probabilities"][k] for k in item["options"]], float)
    if item["type"] == "bool":
        return np.array([a["noul"], 1 - a["noul"]], float)
    pr = a["probabilities"]
    return np.array([pr[str(i)] for i in range(item["levels"])], float)


def judgly_probs(pack, fmt, cond, tier):
    """{id_hash: probability vector in judgly's option order}, plus the gold index."""
    out = {}
    with open(ROOT / "results" / pack / fmt / f"items-{cond}-{tier}.tsv") as f:
        next(f)
        for line in f:
            h, _, _, _, k, y, p = line.rstrip("\n").split("\t")
            out[h] = (np.array([float(v) for v in p.split(",")]), int(y))
    return out


def per_item(P, y):
    top = np.array([int(np.argmax(p)) for p in P])
    conf = np.array([p.max() for p in P])
    brier = np.array([np.sum((p - np.eye(len(p))[g]) ** 2) for p, g in zip(P, y)])
    ll = np.array([-np.log(max(p[g], EPS)) for p, g in zip(P, y)])
    return np.stack([(top == np.array(y)).astype(float), conf, brier, ll])


def metrics(x, idx):
    right, conf, brier, ll = x[:, idx]
    b = np.minimum((conf * 10).astype(int), 9)
    ece = sum(abs(right[b == k].mean() - conf[b == k].mean()) * (b == k).sum() for k in range(10) if (b == k).any()) / len(b)
    return np.array([right.mean(), ece, brier.mean(), ll.mean()])


def boot(xs, grp, n=1000):
    units, inv = np.unique(grp, return_inverse=True)
    members = [np.where(inv == u)[0] for u in range(len(units))]
    draws = {k: [] for k in xs}
    for _ in range(n):
        idx = np.concatenate([members[u] for u in rng.integers(0, len(units), len(units))])
        for k, x in xs.items():
            draws[k].append(metrics(x, idx))
    return draws, len(units)


def main():
    report = {}
    for fmt, tier in TIERS:
        items = [json.loads(l) for l in open(ROOT / "data" / "tiers" / fmt / f"{tier}.jsonl")]
        byid = {x["id"]: x for x in items}
        systems, fits = {}, {}
        for model in MODELS:
            path = HERE / "answers" / model.replace(":", "_") / f"{fmt}-{tier}.jsonl"
            if not path.exists():
                continue
            ans = {}
            for l in open(path):
                r = json.loads(l)
                ans[r["id"]] = r
            systems[model] = ans
        if not systems:
            continue
        # common item set: answered (without error) by every external model
        common = [x["id"] for x in items if all(x["id"] in a and a[x["id"]]["error"] is None for a in systems.values())]
        errors = {m: sum(1 for x in items if x["id"] in a and a[x["id"]]["error"]) for m, a in systems.items()}
        missing = {m: sum(1 for x in items if x["id"] not in a) for m, a in systems.items()}
        grp = np.array([str(byid[i].get("group") or i) for i in common])
        y = [gold_index(byid[i]) for i in common]
        xs = {}
        for m, a in systems.items():
            xs[m] = per_item([external_probs(byid[i], a[i]["response"]) for i in common], y)
            toks = np.array([a[i]["response"]["usage"]["input_tokens"] for i in common])
            fits[m] = toks < CTX.get(m, 10 ** 9)
        hs = [fnv(i) for i in common]
        for short, pack in JUDGLY.items():
            if not (ROOT / "results" / pack / fmt / f"items-h2-{tier}.tsv").exists():
                continue
            for cond in ("raw", "h2", "temperature"):
                d = judgly_probs(pack, fmt, cond, tier)
                # judgly's option order may differ; metrics are order-invariant given its own gold index
                xs[f"judgly-{short}-{cond}"] = per_item([d[h][0] for h in hs], [d[h][1] for h in hs])
            xs[f"judgly-{short}-default"] = xs[f"judgly-{short}-{DEFAULT[(short, fmt)]}"]
        draws, units = boot(xs, grp)
        everything = np.arange(len(common))
        res = {"n_items": len(items), "n_common": len(common), "units": units, "errors": errors, "missing": missing,
               "systems": {}}
        for k, x in xs.items():
            p = metrics(x, everything)
            lo, hi = np.percentile(draws[k], [2.5, 97.5], axis=0)
            row = {m: [round(p[j], 4), round(lo[j], 4), round(hi[j], 4)] for j, m in enumerate(NAMES)}
            for ref in ("judgly-gemma-default", "judgly-qwen-default"):
                if ref in xs and k != ref:
                    d = np.array(draws[k]) - np.array(draws[ref])
                    pd = p - metrics(xs[ref], everything)
                    dlo, dhi = np.percentile(d, [2.5, 97.5], axis=0)
                    row[f"minus_{ref}"] = {m: [round(pd[j], 4), round(dlo[j], 4), round(dhi[j], 4)] for j, m in enumerate(NAMES)}
            if k in fits:
                f = fits[k]
                row["items_over_context"] = int((~f).sum())
                if (~f).any() and f.sum() > 0:
                    pf = metrics(x, np.where(f)[0])
                    row["items_that_fit"] = dict(zip(NAMES, np.round(pf, 4).tolist()))
                secs = np.array([systems[k][i]["seconds"] for i in common])
                row["latency_s"] = {"median": round(float(np.median(secs)), 3), "p95": round(float(np.percentile(secs, 95)), 3)}
            res["systems"][k] = row
        if tier == "bench":
            src = np.array([byid[i]["source"] for i in common])
            res["by_source"] = {s: {k: dict(zip(NAMES, np.round(metrics(x, np.where(src == s)[0]), 4).tolist()))
                                    for k, x in xs.items()} for s in np.unique(src)}
        fam = np.array([byid[i]["family"] for i in common])
        res["by_family"] = {f: {k: dict(zip(NAMES, np.round(metrics(x, np.where(fam == f)[0]), 4).tolist()))
                                for k, x in xs.items()} for f in np.unique(fam)}
        report[f"{fmt}/{tier}"] = res
        print(f"\n== {fmt}/{tier}: {len(common)} of {len(items)} items answered by every external model ({units} units); errors {errors}; missing {missing}")
        for k, row in res["systems"].items():
            c = lambda m: f"{row[m][0]:.3f} [{row[m][1]:.3f},{row[m][2]:.3f}]"
            extra = ""
            if "items_over_context" in row:
                extra = f"  over-context {row['items_over_context']}  latency median {row['latency_s']['median']}s"
            print(f"  {k:<26} acc {c('accuracy')}  ECE {c('ece')}  Brier {c('brier')}{extra}")
    json.dump(report, open(HERE / "result-external.json", "w"), indent=1)


if __name__ == "__main__":
    main()
