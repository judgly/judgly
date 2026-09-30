"""Temperature variants against the shipped head, fitted and judged properly.

Protocol (fixed before any result was seen):
  fit      every variant's parameters on the fit tier's TRAIN split (the data the shipped H2 head
           was fitted on), by minimising log loss;
  choose   per pack and format, the variant with the lowest log loss on the VALIDATION split
           (never used for fitting here) is the pre-registered choice;
  evaluate every variant, once, on the held-out tiers: dev, final (fresh), final-flagged,
           final-seen and bench. No variant is chosen by a held-out result; all are reported.

Variants
  raw   rotation-averaged letter probabilities, no calibration
  T1    one temperature for all questions
  Ttype one temperature per question type (choice, yes/no, score)
  TBtype per question type: a temperature and a bias per option position (27 numbers per type,
         with a small L2 penalty on the biases)
  H2    the head shipped in the pack (fitted by s1-train on the same train split)

    uv run --no-project --with numpy --with scipy python temperature.py /Users/timo/code/judgly FITDIR
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT, FIT = Path(sys.argv[1]), Path(sys.argv[2])
PACKS = ["gemma4-12b-q8", "qwen3-4b-q8"]
HELDOUT = {"general": ["dev", "final", "final-flagged", "final-seen", "bench"],
           "stance": ["dev", "final", "final-flagged", "final-seen"]}
GROUPED = {"final", "final-flagged", "bench"}
KMAX, EPS, L2 = 26, 1e-12, 1e-2
rng = np.random.default_rng(20260929)


def fnv(text):
    h = 0xcbf29ce484222325
    for c in text.encode():
        h = ((h ^ c) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def read(path):
    ids, typ, lab, P = [], [], [], []
    with open(path) as f:
        next(f)
        for line in f:
            h, _, _, t, k, y, p = line.rstrip("\n").split("\t")
            v = np.full(KMAX, np.nan)
            v[:int(k)] = [float(x) for x in p.split(",")]
            ids.append(h); typ.append(int(t)); lab.append(int(y)); P.append(v)
    return ids, np.array(typ), np.array(lab), np.array(P)


def logits(P):
    L = np.log(np.clip(P, EPS, 1))
    L[np.isnan(P)] = -np.inf
    return L


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    E[~np.isfinite(Z)] = 0
    return E / E.sum(axis=1, keepdims=True)


# ---- variants: parameters per question type; theta = [log a, b_0 .. b_25] ------------------------

def calibrated(L, typ, params):
    Z = np.full_like(L, -np.inf)
    for t, (a, b) in params.items():
        m = typ == t if t is not None else np.ones(len(typ), bool)
        Z[m] = a * L[m] + b
    return softmax(Z)


def fit_type(L, y, with_bias):
    mask = np.isfinite(L)
    Lf = np.where(mask, L, 0.0)
    n = len(y)

    def f(theta):
        a = np.exp(theta[0])
        b = theta[1:] if with_bias else np.zeros(KMAX)
        Z = np.where(mask, a * Lf + b, -np.inf)
        P = softmax(Z)
        nll = -np.mean(np.log(np.clip(P[np.arange(n), y], EPS, 1)))
        G = P.copy(); G[np.arange(n), y] -= 1; G /= n            # d nll / d Z
        G[~mask] = 0
        ga = np.sum(G * Lf) * a                                   # chain rule through log a
        gb = G.sum(axis=0) if with_bias else np.zeros(KMAX)
        pen = L2 * np.sum(b ** 2) if with_bias else 0.0
        grad = np.concatenate([[ga], gb + (2 * L2 * b if with_bias else 0)])
        return nll + pen, grad if with_bias else grad[:1]

    x0 = np.zeros(1 + KMAX) if with_bias else np.zeros(1)
    res = minimize(f, x0, jac=True, method="L-BFGS-B")
    a = float(np.exp(res.x[0]))
    b = res.x[1:] if with_bias else np.zeros(KMAX)
    return a, b


def fit_variant(name, L, typ, y):
    if name == "T1":
        return {None: fit_type(L, y, False)}
    types = sorted(set(typ))
    return {t: fit_type(L[typ == t], y[typ == t], name == "TBtype") for t in types}


# ---- metrics -------------------------------------------------------------------------------------

def per_item(P, y):
    K = np.isfinite(P).sum(axis=1)
    Pz = np.nan_to_num(P)
    top = Pz.argmax(axis=1)
    conf = Pz.max(axis=1)
    right = (top == y).astype(float)
    onehot = np.zeros_like(Pz); onehot[np.arange(len(y)), y] = 1
    brier = ((Pz - onehot) ** 2).sum(axis=1)
    ll = -np.log(np.clip(Pz[np.arange(len(y)), y], EPS, 1))
    return np.stack([right, conf, brier, ll])


def metrics(x, idx):
    right, conf, brier, ll = x[:, idx]
    b = np.minimum((conf * 10).astype(int), 9)
    ece = sum(abs(right[b == k].mean() - conf[b == k].mean()) * (b == k).sum()
              for k in range(10) if (b == k).any()) / len(b)
    return np.array([right.mean(), ece, brier.mean(), ll.mean()])


def boot(x, base, grp, n=1000):
    units, inv = np.unique(grp, return_inverse=True)
    members = [np.where(inv == u)[0] for u in range(len(units))]
    point, bpoint = metrics(x, np.arange(x.shape[1])), metrics(base, np.arange(x.shape[1]))
    D, M = [], []
    for _ in range(n):
        idx = np.concatenate([members[u] for u in rng.integers(0, len(units), len(units))])
        m = metrics(x, idx); M.append(m); D.append(m - metrics(base, idx))
    lo, hi = np.percentile(M, [2.5, 97.5], axis=0)
    dlo, dhi = np.percentile(D, [2.5, 97.5], axis=0)
    return point, lo, hi, point - bpoint, dlo, dhi


def groups(fmt, tier, ids):
    g = {}
    for line in open(ROOT / "data" / "tiers" / fmt / f"{tier}.jsonl"):
        ex = json.loads(line)
        g[fnv(ex["id"])] = f'{ex["source"]}:{ex["group"]}' if ex.get("group") else ex["id"]
    return np.array([g[i] for i in ids])


VARIANTS = ["raw", "T1", "Ttype", "TBtype", "H2"]
NAMES = ("accuracy", "ece", "brier", "log_loss")


def main():
    out = {}
    for pack in PACKS:
        for fmt in ("general", "stance"):
            key = f"{pack}/{fmt}"
            _, ttyp, ty, tP = read(FIT / pack / fmt / "items-raw-train.tsv")
            params = {v: fit_variant(v, logits(tP), ttyp, ty) for v in ("T1", "Ttype", "TBtype")}

            def probs(v, split_dir, tier):
                src = "h2" if v == "H2" else "raw"
                path = (FIT / pack / fmt if split_dir else ROOT / "results" / pack / fmt) / f"items-{src}-{tier}.tsv"
                ids, typ, y, P = read(path)
                if v in params:
                    P = calibrated(logits(P), typ, params[v])
                return ids, typ, y, P

            res = {"params": {v: {str(t): {"temperature": round(1 / a, 3),
                                           "max_abs_bias": round(float(np.abs(b).max()), 3)}
                                  for t, (a, b) in p.items()} for v, p in params.items()},
                   "train": {}, "validation": {}}
            for split in ("train", "validation"):
                for v in VARIANTS:
                    _, _, y, P = probs(v, True, split)
                    res[split][v] = dict(zip(NAMES, np.round(metrics(per_item(P, y), np.arange(len(y))), 4).tolist()))
            choice = min(VARIANTS[1:], key=lambda v: res["validation"][v]["log_loss"])
            res["preregistered_choice"] = choice
            for tier in HELDOUT[fmt]:
                ids, _, y, Pb = probs("H2", False, tier)
                base = per_item(Pb, y)
                grp = groups(fmt, tier, ids) if tier in GROUPED else np.array(ids)
                res[tier] = {"n": len(ids), "units": len(np.unique(grp))}
                for v in VARIANTS:
                    ids_v, _, yv, P = probs(v, False, tier)
                    assert ids_v == ids and (yv == y).all()
                    p, lo, hi, d, dlo, dhi = boot(per_item(P, y), base, grp)
                    res[tier][v] = {m: [round(p[k], 4), round(lo[k], 4), round(hi[k], 4)] for k, m in enumerate(NAMES)}
                    res[tier][v]["minus_H2"] = {m: [round(d[k], 4), round(dlo[k], 4), round(dhi[k], 4)] for k, m in enumerate(NAMES)}
            out[key] = res
            print(f"\n######## {key}   pre-registered choice (lowest validation log loss): {choice}")
            print("fitted:", json.dumps(res["params"]))
            for split in ("train", "validation"):
                print(f"  {split:<11}" + "  ".join(f"{v} ll {res[split][v]['log_loss']:.3f} ece {res[split][v]['ece']:.3f}" for v in VARIANTS))
            for tier in HELDOUT[fmt]:
                r = res[tier]
                print(f"\n  == {tier} (n {r['n']}, {r['units']} units)")
                print(f"  {'variant':<8}{'accuracy':>22}{'ECE':>22}{'Brier':>22}{'log loss':>22}   minus H2: acc / ECE / Brier")
                for v in VARIANTS:
                    m = r[v]; c = lambda k: f"{m[k][0]:.3f} [{m[k][1]:.3f},{m[k][2]:.3f}]"
                    d = m["minus_H2"]; dd = lambda k: f"{d[k][0]:+.3f} [{d[k][1]:+.3f},{d[k][2]:+.3f}]"
                    print(f"  {v:<8}{c('accuracy'):>22}{c('ece'):>22}{c('brier'):>22}{c('log_loss'):>22}   {dd('accuracy')} / {dd('ece')} / {dd('brier')}")
    json.dump(out, open(Path(__file__).with_name("temperature.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
