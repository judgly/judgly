"""Exploratory: temperature-only calibration and Gemma+Qwen combination, from the cached per-item
results of the judgly 0.1.0 runs. Parameters (a few scalars per variant) are fitted on the fit
tier's test split; every variant is then reported on dev, final and final-seen. No variant is
selected by its final score.

    uv run --with numpy --with scipy python combine.py /Users/timo/code/judgly
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(sys.argv[1])
PACKS = {"G": "gemma4-12b-q8", "Q": "qwen3-4b-q8"}
TIERS = ["dev", "final", "final-seen"]
EPS = 1e-12
rng = np.random.default_rng(20260929)


def fnv(text):
    h = 0xcbf29ce484222325
    for c in text.encode():
        h = ((h ^ c) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def read(pack, fmt, cond, tier):
    rows = {}
    with open(ROOT / "results" / pack / fmt / f"items-{cond}-{tier}.tsv") as f:
        next(f)
        for line in f:
            h, _, _, typ, k, label, p = line.rstrip("\n").split("\t")
            rows[h] = (int(typ), int(label), np.array([float(v) for v in p.split(",")]))
    return rows


def load(fmt, tier):
    """Aligned items: ids, types, labels, and per source a list of probability vectors."""
    src = {f"{m}_{c}": read(PACKS[m], fmt, c, tier) for m in PACKS for c in ("raw", "h2")}
    ids = sorted(set.intersection(*(set(v) for v in src.values())))
    assert all(len(v) == len(ids) for v in src.values()), "item sets differ"
    typ = np.array([src["G_raw"][i][0] for i in ids])
    lab = np.array([src["G_raw"][i][1] for i in ids])
    for v in src.values():
        assert all(v[i][1] == src["G_raw"][i][1] for i in ids)
    probs = {k: [v[i][2] for i in ids] for k, v in src.items()}
    return ids, typ, lab, probs


def groups(fmt, tier, ids):
    """Resampling units: the tier file's group where present, else the item."""
    path = ROOT / "data" / "tiers" / fmt / f"{tier}.jsonl"
    g = {}
    for line in open(path):
        ex = json.loads(line)
        g[fnv(ex["id"])] = f'{ex["source"]}:{ex["group"]}' if ex.get("group") else ex["id"]
    return np.array([g.get(i, i) for i in ids])


def pool(parts, weights):
    """Log-linear pool: p proportional to prod p_j ** w_j (one weight = a temperature)."""
    out = []
    for vecs in zip(*parts):
        z = sum(w * np.log(np.clip(v, EPS, 1)) for w, v in zip(weights, vecs))
        z = z - z.max()
        e = np.exp(z)
        out.append(e / e.sum())
    return out


def nll(ps, lab):
    return -np.mean([np.log(max(p[y], EPS)) for p, y in zip(ps, lab)])


def fit(parts, lab, typ, per_type):
    """Weights >= 0 minimising log loss on the fitting split; per question type if asked."""
    types = sorted(set(typ)) if per_type else [None]
    w = {}
    for t in types:
        idx = [i for i in range(len(lab)) if t is None or typ[i] == t]
        sub = [[p[i] for i in idx] for p in parts]
        f = lambda x: nll(pool(sub, np.exp(x)), lab[idx])
        res = minimize(f, np.zeros(len(parts)), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6})
        w[t] = np.exp(res.x)
    return w


def apply(parts, typ, w):
    n = len(typ)
    return [pool([[p[i]] for p in parts], w[typ[i] if typ[i] in w else None])[0] for i in range(n)]


def per_item(ps, lab):
    top = np.array([int(np.argmax(p)) for p in ps])
    conf = np.array([p.max() for p in ps])
    right = (top == lab).astype(float)
    brier = np.array([np.sum((p - np.eye(len(p))[y]) ** 2) for p, y in zip(ps, lab)])
    ll = np.array([-np.log(max(p[y], EPS)) for p, y in zip(ps, lab)])
    return right, conf, brier, ll


def ece(right, conf):
    b = np.minimum((conf * 10).astype(int), 9)
    return sum(abs(right[b == k].mean() - conf[b == k].mean()) * (b == k).sum() for k in range(10) if (b == k).any()) / len(b)


def metrics(x, idx):
    right, conf, brier, ll = (v[idx] for v in x)
    return np.array([right.mean(), ece(right, conf), brier.mean(), ll.mean()])


def boot(x, grp, base=None, n=1000):
    units = np.unique(grp)
    members = {u: np.where(grp == u)[0] for u in units}
    point = metrics(x, np.arange(len(grp)))
    bpoint = metrics(base, np.arange(len(grp))) if base else None
    draws, diffs = [], []
    for _ in range(n):
        idx = np.concatenate([members[u] for u in rng.choice(units, len(units))])
        m = metrics(x, idx)
        draws.append(m)
        if base:
            diffs.append(m - metrics(base, idx))
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    d = None
    if base:
        dlo, dhi = np.percentile(diffs, [2.5, 97.5], axis=0)
        d = (point - bpoint, dlo, dhi)
    return point, lo, hi, d


VARIANTS = [
    # name, sources, per-type weights?, fitted?
    ("Gemma raw", ["G_raw"], False, False),
    ("Gemma one temperature", ["G_raw"], False, True),
    ("Gemma temperature per question type", ["G_raw"], True, True),
    ("Gemma shipped head (H2)", ["G_h2"], False, False),
    ("Qwen raw", ["Q_raw"], False, False),
    ("Qwen one temperature", ["Q_raw"], False, True),
    ("Qwen temperature per question type", ["Q_raw"], True, True),
    ("Qwen shipped head (H2)", ["Q_h2"], False, False),
    ("Gemma+Qwen raw, pooled", ["G_raw", "Q_raw"], False, True),
    ("Gemma+Qwen raw, pooled per question type", ["G_raw", "Q_raw"], True, True),
    ("Gemma+Qwen heads, pooled", ["G_h2", "Q_h2"], False, True),
    ("Gemma+Qwen heads, plain average", None, False, False),
]


def main():
    report = {}
    for fmt in ("general", "stance"):
        _, ftyp, flab, fprobs = load(fmt, "test")
        weights = {}
        for name, srcs, per_type, fitted in VARIANTS:
            if fitted:
                weights[name] = fit([fprobs[s] for s in srcs], flab, ftyp, per_type)
        report[fmt] = {"weights": {k: {str(t): v.round(3).tolist() for t, v in w.items()} for k, w in weights.items()}}
        for tier in TIERS:
            ids, typ, lab, probs = load(fmt, tier)
            grp = groups(fmt, tier, ids) if tier == "final" else np.array(ids)
            out = {}
            base = per_item(probs["G_h2"], lab)
            for name, srcs, per_type, fitted in VARIANTS:
                if srcs is None:
                    ps = [(a + b) / 2 for a, b in zip(probs["G_h2"], probs["Q_h2"])]
                elif fitted:
                    ps = apply([probs[s] for s in srcs], typ, weights[name])
                else:
                    ps = probs[srcs[0]]
                point, lo, hi, d = boot(per_item(ps, lab), grp, base)
                out[name] = {"n": len(ids), "groups": len(np.unique(grp)),
                             **{m: [round(point[k], 4), round(lo[k], 4), round(hi[k], 4)]
                                for k, m in enumerate(("accuracy", "ece", "brier", "log_loss"))},
                             "vs_gemma_h2": {m: [round(d[0][k], 4), round(d[1][k], 4), round(d[2][k], 4)]
                                             for k, m in enumerate(("accuracy", "ece", "brier", "log_loss"))}}
            report[fmt][tier] = out
    json.dump(report, open(Path(__file__).with_name("combine.json"), "w"), indent=1)
    for fmt in report:
        print(f"\n######## {fmt}   fitted weights: {report[fmt]['weights']}")
        for tier in TIERS:
            r = report[fmt][tier]
            first = next(iter(r.values()))
            print(f"\n== {fmt} / {tier} (n {first['n']}, {first['groups']} resampling units)")
            print(f"{'variant':<44}{'accuracy':>24}{'ECE':>24}{'Brier':>24}   acc diff vs Gemma H2")
            for name, m in r.items():
                c = lambda k: f"{m[k][0]:.3f} [{m[k][1]:.3f},{m[k][2]:.3f}]"
                d = m["vs_gemma_h2"]["accuracy"]
                print(f"{name:<44}{c('accuracy'):>24}{c('ece'):>24}{c('brier'):>24}   {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]")


if __name__ == "__main__":
    main()
