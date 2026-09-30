"""Fit the per-type temperature for every system on the fixed training sample and score all systems
on the comparison's test items (PROTOCOL.md). CPU only: external training answers from
answers/, their test answers from the comparison record, judgly's readouts from its cached
feature files (train split, via s1-eval) and committed per-item dumps (test tiers).

    uv run --no-project --with numpy --with scipy python score_calibrated.py JUDGLY_ROOT
"""

import gzip
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

HERE = Path(__file__).parent
REC = HERE.parent
ROOT = Path(sys.argv[1])
MODELS = ["nimble:9b", "tev1:4b", "tev1:0.8b"]
PACKS = {"gemma": "gemma4-12b-q8", "qwen": "qwen3-4b-q8"}
DEFAULT = {("gemma", "general"): "h2", ("gemma", "stance"): "temperature",
           ("qwen", "general"): "temperature", ("qwen", "stance"): "temperature"}
TIERS = [("general", "confirm"), ("stance", "confirm"), ("general", "final"), ("stance", "final"),
         ("general", "bench"), ("general", "final-flagged"), ("stance", "final-flagged")]
TYPES = {"choice": 0, "bool": 1, "score": 2}
NAMES = ("accuracy", "ece", "brier", "log_loss")
EPS = 1e-12


def scorer():
    argv = sys.argv
    sys.argv = [argv[0], str(ROOT)]  # the comparison scorer reads its arguments at import
    spec = importlib.util.spec_from_file_location("score_external", REC / "score_external.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


S = scorer()


def external_vector(item, resp):
    """Probabilities in the order of keys_of(item), and the gold index (as the comparison scorer)."""
    return np.clip(S.external_probs(item, resp), EPS, 1), S.gold_index(item)


def temper(p, T):
    z = np.log(np.clip(p, EPS, 1)) / T
    e = np.exp(z - z.max())
    return e / e.sum()


def fit(P, y, types):
    """One temperature per question type, minimising log loss."""
    out = {}
    for t in sorted(set(types)):
        idx = [i for i, x in enumerate(types) if x == t]
        f = lambda lt: -np.mean([np.log(max(temper(P[i], np.exp(lt))[y[i]], EPS)) for i in idx])
        out[t] = float(np.exp(minimize_scalar(f, bounds=(-3, 4), method="bounded").x))
    return out


def judgly_train(pack, fmt, work):
    """judgly's raw readout on the train split, from its cached features (s1-eval, as the pipeline)."""
    dump = work / f"{pack}-{fmt}-train.tsv"
    subprocess.run([str(ROOT / "build" / "cli" / "s1-eval"), "--features",
                    str(ROOT / "results" / pack / fmt / "fitdev" / "features.feat"), "--split", "train",
                    "--rotations", "--dump-items", str(dump)], check=True, stdout=subprocess.DEVNULL)
    out = {}
    for line in list(open(dump))[1:]:
        h, _, _, t, k, y, p = line.rstrip("\n").split("\t")
        out[h] = (np.array([float(v) for v in p.split(",")]), int(y), int(t))
    return out


def judgly_test(pack, fmt, cond, tier):
    out = {}
    for line in list(open(ROOT / "results" / pack / fmt / f"items-{cond}-{tier}.tsv"))[1:]:
        h, _, _, t, k, y, p = line.rstrip("\n").split("\t")
        out[h] = (np.array([float(v) for v in p.split(",")]), int(y), int(t))
    return out


def main():
    sample = json.load(open(HERE / "sample.json"))
    temps, report = {}, {"temperatures": {}, "tiers": {}}
    # --- fit external models on the training sample
    for m in MODELS:
        for fmt, ids in sample.items():
            rows = {r["id"]: r for r in map(json.loads, open(ROOT / "data" / "tiers" / fmt / "fitdev.jsonl"))}
            ans = {r["id"]: r for r in map(json.loads, open(HERE / "answers" / m.replace(":", "_") / f"{fmt}-train.jsonl"))}
            ok = [i for i in ids if ans[i]["error"] is None]
            P, y, types = [], [], []
            for i in ok:
                p, g = external_vector(rows[i], ans[i]["response"])
                P.append(p); y.append(g); types.append(rows[i]["type"])
            temps[(m, fmt)] = fit(P, y, types)
            report["temperatures"][f"{m}/{fmt}"] = {"n_fit": len(ok), "refused": len(ids) - len(ok),
                                                     "T": {t: round(v, 4) for t, v in temps[(m, fmt)].items()}}
    # --- judgly refitted on the same sample (control)
    with tempfile.TemporaryDirectory() as tmp:
        for short, pack in PACKS.items():
            for fmt, ids in sample.items():
                tr = judgly_train(pack, fmt, Path(tmp))
                hs = [S.fnv(i) for i in ids]
                inv = {v: k for k, v in TYPES.items()}
                P = [tr[h][0] for h in hs]; y = [tr[h][1] for h in hs]; types = [inv[tr[h][2]] for h in hs]
                temps[(f"judgly-{short}", fmt)] = fit(P, y, types)
                report["temperatures"][f"judgly-{short}/{fmt} (refit on the sample)"] = {
                    "n_fit": len(ids), "T": {t: round(v, 4) for t, v in temps[(f"judgly-{short}", fmt)].items()}}
    # --- score every system on the test tiers
    for fmt, tier in TIERS:
        items = [json.loads(line) for line in open(ROOT / "data" / "tiers" / fmt / f"{tier}.jsonl")]
        ext = {m: {r["id"]: r for r in map(json.loads, gzip.open(REC / "answers" / m.replace(":", "_") / f"{fmt}-{tier}.jsonl.gz", "rt"))}
               for m in MODELS}
        res = {}
        for m in MODELS:
            ids = [x["id"] for x in items if ext[m][x["id"]]["error"] is None]
            byid = {x["id"]: x for x in items}
            grp = np.array([str(byid[i].get("group") or i) for i in ids])
            raw, cal, y = [], [], []
            for i in ids:
                p, g = external_vector(byid[i], ext[m][i]["response"])
                raw.append(p); cal.append(temper(p, temps[(m, fmt)].get(byid[i]["type"], 1.0))); y.append(g)
            xs = {f"{m} as served": S.per_item(raw, y), f"{m} calibrated": S.per_item(cal, y)}
            hs = [S.fnv(i) for i in ids]
            inv = {v: k for k, v in TYPES.items()}
            for short, pack in PACKS.items():
                d = judgly_test(pack, fmt, "raw", tier)
                P = [d[h][0] for h in hs]; yy = [d[h][1] for h in hs]
                xs[f"judgly-{short} raw"] = S.per_item(P, yy)
                xs[f"judgly-{short} refit on the sample"] = S.per_item(
                    [temper(d[h][0], temps[(f"judgly-{short}", fmt)].get(inv[d[h][2]], 1.0)) for h in hs], yy)
                for cond in ("temperature", "h2"):
                    dc = judgly_test(pack, fmt, cond, tier)
                    xs[f"judgly-{short} {cond}"] = S.per_item([dc[h][0] for h in hs], [dc[h][1] for h in hs])
                xs[f"judgly-{short} default"] = xs[f"judgly-{short} {DEFAULT[(short, fmt)]}"]
            draws, units = S.boot(xs, grp)
            everything = np.arange(len(ids))
            block = {"n": len(ids), "units": units, "systems": {}}
            for k, x in xs.items():
                p = S.metrics(x, everything)
                lo, hi = np.percentile(draws[k], [2.5, 97.5], axis=0)
                row = {n: [round(p[j], 4), round(lo[j], 4), round(hi[j], 4)] for j, n in enumerate(NAMES)}
                for ref in ("judgly-gemma default", "judgly-qwen default"):
                    if k != ref:
                        dd = np.array(draws[k]) - np.array(draws[ref])
                        dp = p - S.metrics(xs[ref], everything)
                        dlo, dhi = np.percentile(dd, [2.5, 97.5], axis=0)
                        row[f"minus {ref}"] = {n: [round(dp[j], 4), round(dlo[j], 4), round(dhi[j], 4)] for j, n in enumerate(NAMES)}
                block["systems"][k] = row
            res[m] = block
        report["tiers"][f"{fmt}/{tier}"] = res
    json.dump(report, open(HERE / "result.json", "w"), indent=1)
    print(json.dumps(report["temperatures"], indent=1))
    for key, res in report["tiers"].items():
        print(f"\n== {key}")
        for m, block in res.items():
            s = block["systems"]
            show = [f"{m} as served", f"{m} calibrated", "judgly-gemma default", "judgly-qwen default",
                    "judgly-gemma refit on the sample", "judgly-qwen refit on the sample"]
            print(f"  on {m}'s {block['n']} items: " + " | ".join(
                f"{k.replace(m + ' ', '')}: ECE {s[k]['ece'][0]:.3f} Brier {s[k]['brier'][0]:.3f}" for k in show))


if __name__ == "__main__":
    main()
