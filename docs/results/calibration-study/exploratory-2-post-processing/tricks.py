"""CPU calibration tricks on judgly 0.1.0's cached readouts. Follows PROTOCOL.md (frozen; see
PROTOCOL.sha256). Reads the feature files, rebuilds the raw readout per option order, checks it
against the release's per-item dumps, fits every variant on the train split, chooses on dev,
reports all tiers, and runs the Tdomain sub-study.

    uv run --no-project --with numpy --with scipy --with scikit-learn python tricks.py /Users/timo/code/judgly
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

ROOT = Path(sys.argv[1])
OUT = Path(__file__).parent
PACKS = ["gemma4-12b-q8", "qwen3-4b-q8"]
KMAX, EPS = 26, 1e-12
TIERS = {"general": [("dev", "fitdev", 3), ("final", "final", 3), ("final-flagged", "final-flagged", 3),
                     ("final-seen", "final-seen", 3), ("bench", "bench", 3)],
         "stance": [("dev", "fitdev", 3), ("final", "final", 3), ("final-flagged", "final-flagged", 3),
                    ("final-seen", "final-seen", 3)]}
GROUPED = {"final", "final-flagged", "bench"}
TRAIN, HELDOUT = 0, 3
NAMES = ("accuracy", "ece", "brier", "log_loss")
rng = np.random.default_rng(20260929)


def fnv(text):
    h = 0xcbf29ce484222325
    for c in text.encode():
        h = ((h ^ c) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


# ---- feature files -----------------------------------------------------------------------------

def load_feat(path, split):
    """Items of one split: id (hex), type, K, label (original option index), the averaged raw
    probabilities (KMAX, NaN-padded) and the order disagreement d."""
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    n_embd = int(np.frombuffer(raw[12:16], "<u4")[0])
    n_rec = int(np.frombuffer(raw[24:32], "<u8")[0])
    dt = np.dtype([("id", "<u8"), ("task", "<u4"), ("fam", "<u2"), ("type", "u1"), ("split", "u1"),
                   ("K", "u1"), ("label", "u1"), ("rot", "u1"), ("perm", "u1", KMAX),
                   ("target", "<f4", KMAX), ("z", "<f4", KMAX), ("zc", "<f4", KMAX), ("mass", "<f4"),
                   ("h", "V", 4 * n_embd)])
    assert dt.itemsize == 361 + 4 * n_embd
    rec = np.memmap(path, dtype=dt, mode="r", offset=200, shape=(n_rec,))
    sel = np.where(rec["split"] == split)[0]
    ids, typ, K, lab, fam = [], [], [], [], []
    P, D = [], []
    start = 0
    ids_all = rec["id"][sel]
    while start < len(sel):
        end = start
        while end < len(sel) and ids_all[end] == ids_all[start]:
            end += 1
        r = rec[sel[start:end]]
        k = int(r["K"][0])
        per = np.zeros((len(r), k))
        for j, x in enumerate(r):
            z = x["z"][:k].astype(np.float64)
            e = np.exp(z - z.max())
            p = e / e.sum()
            per[j, x["perm"][:k]] = p          # slot s shows option perm[s]
        mean = per.mean(axis=0)
        d = 0.5 * np.abs(per - mean).sum(axis=1).mean()
        v = np.full(KMAX, np.nan)
        v[:k] = mean
        ids.append(f"{int(r['id'][0]):016x}"); typ.append(int(r["type"][0])); K.append(k)
        lab.append(int(r["perm"][0][r["label"][0]])); fam.append(int(r["fam"][0]))
        P.append(v); D.append(d)
        start = end
    return {"ids": ids, "type": np.array(typ), "K": np.array(K), "label": np.array(lab),
            "fam": np.array(fam), "P": np.array(P), "d": np.array(D)}


def read_dump(path):
    out = {}
    with open(path) as f:
        next(f)
        for line in f:
            h, _, _, t, k, y, p = line.rstrip("\n").split("\t")
            out[h] = (int(y), np.array([float(v) for v in p.split(",")]))
    return out


def check_against_dump(items, path):
    dump = read_dump(path)
    assert set(dump) == set(items["ids"]), f"{path}: item sets differ"
    worst = 0.0
    for i, h in enumerate(items["ids"]):
        y, p = dump[h]
        assert y == items["label"][i], f"{path}: label mismatch"
        worst = max(worst, np.abs(p - items["P"][i][:len(p)]).max())
    return worst


# ---- variants ------------------------------------------------------------------------------------

def logp(P):
    L = np.log(np.clip(P, EPS, 1))
    L[np.isnan(P)] = -np.inf
    return L


def softmax(Z):
    Z = Z - np.nanmax(np.where(np.isfinite(Z), Z, np.nan), axis=1, keepdims=True)
    E = np.exp(Z)
    E[~np.isfinite(Z)] = 0
    return E / E.sum(axis=1, keepdims=True)


def nll(P, y):
    return -np.mean(np.log(np.clip(P[np.arange(len(y)), y], EPS, 1)))


def scale(P, invT):
    return softmax(logp(P) * invT[:, None])


def fit_Ttype(it):
    w = {}
    for t in np.unique(it["type"]):
        m = it["type"] == t
        f = lambda x: nll(scale(it["P"][m], np.full(m.sum(), np.exp(x[0]))), it["label"][m])
        w[int(t)] = float(np.exp(minimize(f, [0.0], method="Nelder-Mead").x[0]))
    return w


def apply_Ttype(it, w):
    return scale(it["P"], np.array([w.get(int(t), 1.0) for t in it["type"]]))


def fit_Tdis(it):
    w = {}
    for t in np.unique(it["type"]):
        m = it["type"] == t
        P, d, y = it["P"][m], it["d"][m], it["label"][m]
        f = lambda x: nll(scale(P, np.exp(-(x[0] + x[1] * d))), y)
        w[int(t)] = minimize(f, [0.0, 0.0], method="Nelder-Mead").x.tolist()
    return w


def apply_Tdis(it, w):
    a = np.array([w[int(t)][0] for t in it["type"]]); b = np.array([w[int(t)][1] for t in it["type"]])
    return scale(it["P"], np.exp(-(a + b * it["d"])))


def top_parts(P):
    Pz = np.nan_to_num(P)
    top = Pz.argmax(axis=1)
    conf = Pz[np.arange(len(P)), top]
    second = np.sort(Pz, axis=1)[:, -2]
    return Pz, top, conf, second


def remap(P, newconf):
    """Top probability set to newconf (kept on top), the others rescaled in proportion."""
    Pz, top, conf, second = top_parts(P)
    floor = second / np.clip(1 - conf + second, EPS, None)
    newconf = np.clip(np.maximum(newconf, floor), 1e-4, 1 - 1e-4)
    rest = (1 - newconf) / np.clip(1 - conf, EPS, None)
    Q = Pz * rest[:, None]
    Q[np.arange(len(P)), top] = newconf
    Q[np.isnan(P)] = np.nan
    return Q


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_top(it, base, kind):
    Pz, top, conf, _ = top_parts(base)
    right = (top == it["label"]).astype(float)
    w = {}
    for t in np.unique(it["type"]):
        m = it["type"] == t
        c, r = conf[m], right[m]
        if kind == "Platt":
            f = lambda x: -np.mean(r * np.log(np.clip(1 / (1 + np.exp(-(x[0] * logit(c) + x[1]))), EPS, 1))
                                   + (1 - r) * np.log(np.clip(1 - 1 / (1 + np.exp(-(x[0] * logit(c) + x[1]))), EPS, 1)))
            w[int(t)] = ("Platt", minimize(f, [1.0, 0.0], method="Nelder-Mead").x.tolist())
        elif kind == "Isotonic":
            w[int(t)] = ("Isotonic", IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(c, r))
        else:
            edges = np.linspace(0, 1, 11)
            b = np.minimum((c * 10).astype(int), 9)
            acc = [((r[b == k].sum() + 1) / ((b == k).sum() + 2)) if (b == k).any() else None for k in range(10)]
            w[int(t)] = ("Hist", acc)
    return w


def apply_top(it, base, w):
    _, _, conf, _ = top_parts(base)
    new = conf.copy()
    for t, (kind, prm) in w.items():
        m = it["type"] == t
        if not m.any():
            continue
        if kind == "Platt":
            new[m] = 1 / (1 + np.exp(-(prm[0] * logit(conf[m]) + prm[1])))
        elif kind == "Isotonic":
            new[m] = prm.predict(conf[m])
        else:
            b = np.minimum((conf[m] * 10).astype(int), 9)
            new[m] = [prm[k] if prm[k] is not None else c for k, c in zip(b, conf[m])]
    return remap(base, new)


# ---- metrics -------------------------------------------------------------------------------------

def per_item(P, y):
    Pz = np.nan_to_num(P)
    top = Pz.argmax(axis=1)
    conf = Pz.max(axis=1)
    onehot = np.zeros_like(Pz); onehot[np.arange(len(y)), y] = 1
    return np.stack([(top == y).astype(float), conf, ((Pz - onehot) ** 2).sum(axis=1),
                     -np.log(np.clip(Pz[np.arange(len(y)), y], EPS, 1))])


def metrics(x, idx):
    right, conf, brier, ll = x[:, idx]
    b = np.minimum((conf * 10).astype(int), 9)
    ece = sum(abs(right[b == k].mean() - conf[b == k].mean()) * (b == k).sum()
              for k in range(10) if (b == k).any()) / len(b)
    return np.array([right.mean(), ece, brier.mean(), ll.mean()])


def boot(x, base, grp, n=1000):
    units, inv = np.unique(grp, return_inverse=True)
    members = [np.where(inv == u)[0] for u in range(len(units))]
    everything = np.arange(x.shape[1])
    point, bpoint = metrics(x, everything), metrics(base, everything)
    M, D = [], []
    for _ in range(n):
        idx = np.concatenate([members[u] for u in rng.integers(0, len(units), len(units))])
        m = metrics(x, idx); M.append(m); D.append(m - metrics(base, idx))
    lo, hi = np.percentile(M, [2.5, 97.5], axis=0)
    dlo, dhi = np.percentile(D, [2.5, 97.5], axis=0)
    return point, lo, hi, point - bpoint, dlo, dhi


def tier_rows(fmt, tier):
    rows = {}
    for line in open(ROOT / "data" / "tiers" / fmt / f"{'fitdev' if tier == 'dev' else tier}.jsonl"):
        ex = json.loads(line)
        rows[f"{fnv(ex['id']):016x}"] = ex
    return rows


# ---- main ----------------------------------------------------------------------------------------

VARIANTS = ["raw", "Ttype", "Tdis", "Platt", "Isotonic", "Hist", "H2"]
N_PARAM = {"raw": 0, "Ttype": 3, "Tdis": 6, "Platt": 9, "Isotonic": 99, "Hist": 99, "H2": 10 ** 6}


def main():
    report = {"protocol_sha256": open(OUT / "PROTOCOL.sha256").read().split()[0]}
    for pack in PACKS:
        for fmt in ("general", "stance"):
            key = f"{pack}/{fmt}"
            rdir = ROOT / "results" / pack / fmt
            train = load_feat(rdir / "fitdev" / "features.feat", TRAIN)
            fitted = {"Ttype": fit_Ttype(train), "Tdis": fit_Tdis(train)}
            tb = apply_Ttype(train, fitted["Ttype"])
            for kind in ("Platt", "Isotonic", "Hist"):
                fitted[kind] = fit_top(train, tb, kind)

            def variant(v, it, tier):
                if v == "raw":
                    return it["P"]
                if v == "Ttype":
                    return apply_Ttype(it, fitted["Ttype"])
                if v == "Tdis":
                    return apply_Tdis(it, fitted["Tdis"])
                if v == "H2":
                    d = read_dump(rdir / f"items-h2-{tier}.tsv")
                    out = np.full_like(it["P"], np.nan)
                    for i, h in enumerate(it["ids"]):
                        out[i, :len(d[h][1])] = d[h][1]
                    return out
                return apply_top(it, apply_Ttype(it, fitted["Ttype"]), fitted[v])

            res = {"fitted": {"Ttype_temperature": {t: round(1 / a, 3) for t, a in fitted["Ttype"].items()},
                              "Tdis_alpha_beta": {t: [round(v, 3) for v in ab] for t, ab in fitted["Tdis"].items()},
                              "Platt": {t: [round(v, 3) for v in p[1]] for t, p in fitted["Platt"].items()}},
                   "raw_check_max_abs_diff": {}}
            items = {}
            for tier, fset, _ in TIERS[fmt]:
                split = HELDOUT
                it = load_feat(rdir / fset / "features.feat", split)
                res["raw_check_max_abs_diff"][tier] = check_against_dump(it, rdir / f"items-raw-{tier}.tsv")
                items[tier] = it
            # choose on dev (protocol rule)
            dev = items["dev"]
            devll = {v: float(metrics(per_item(variant(v, dev, "dev"), dev["label"]), np.arange(len(dev["label"])))[3])
                     for v in VARIANTS if v != "raw"}
            best = min(devll.values())
            choice = min([v for v in devll if devll[v] <= best + 0.002], key=lambda v: N_PARAM[v])
            res["dev_log_loss"] = {v: round(x, 4) for v, x in devll.items()}
            res["choice_by_protocol"] = choice
            for tier, _, _ in TIERS[fmt]:
                it = items[tier]
                rows = tier_rows(fmt, tier)
                grp = np.array([f"{rows[h]['source']}:{rows[h]['group']}" if tier in GROUPED and rows[h].get("group")
                                else h for h in it["ids"]])
                y = it["label"]
                base = per_item(variant("Ttype", it, tier), y)
                res[tier] = {"n": len(y), "units": len(np.unique(grp))}
                for v in VARIANTS:
                    p, lo, hi, dd, dlo, dhi = boot(per_item(variant(v, it, tier), y), base, grp)
                    res[tier][v] = {m: [round(p[k], 4), round(lo[k], 4), round(hi[k], 4)] for k, m in enumerate(NAMES)}
                    res[tier][v]["minus_Ttype"] = {m: [round(dd[k], 4), round(dlo[k], 4), round(dhi[k], 4)]
                                                   for k, m in enumerate(NAMES)}
            # Tdomain sub-study on dev and final
            res["Tdomain"] = {}
            for tier in ("dev", "final"):
                it = items[tier]
                rows = tier_rows(fmt, tier)
                fams = np.array([rows[h]["family"] for h in it["ids"]])
                gkey = np.array([int(hashlib.sha256(str(rows[h].get('group') or h).encode()).hexdigest()[:12], 16)
                                 for h in it["ids"]])
                per_n = {}
                for fam in np.unique(fams):
                    m = np.where(fams == fam)[0]
                    A = m[gkey[m] % 2 == 0]; B = m[gkey[m] % 2 == 1]
                    A = A[np.argsort(gkey[A] // 2)]
                    if len(A) < 25 or len(B) < 25:
                        res.setdefault("Tdomain_skipped", []).append(f"{tier}:{fam} (A {len(A)}, B {len(B)})")
                        continue
                    yB = it["label"][B]
                    sub = lambda idx: {k: (v[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx]) for k, v in it.items()}
                    ref_T = per_item(apply_Ttype(sub(B), fitted["Ttype"]), yB)
                    ref_raw = per_item(sub(B)["P"], yB)
                    for n in (25, 50, 100, 200, "all"):
                        idx = A if n == "all" else A[:n]
                        if n != "all" and len(A) < n:
                            continue
                        sA = sub(idx)
                        f = lambda x: nll(scale(sA["P"], np.full(len(idx), np.exp(x[0]))), sA["label"])
                        invT = float(np.exp(minimize(f, [0.0], method="Nelder-Mead").x[0]))
                        got = per_item(scale(sub(B)["P"], np.full(len(B), invT)), yB)
                        e = np.arange(len(B))
                        per_n.setdefault(str(n), []).append({"family": fam, "nB": len(B), "T": round(1 / invT, 3),
                            "Tdomain": metrics(got, e).round(4).tolist(), "Ttype": metrics(ref_T, e).round(4).tolist(),
                            "raw": metrics(ref_raw, e).round(4).tolist()})
                res["Tdomain"][tier] = per_n
            report[key] = res
            print_block(key, res)
    json.dump(report, open(OUT / "tricks.json", "w"), indent=1, default=float)


def print_block(key, res):
    print(f"\n######## {key}")
    print("raw readout reproduced (max |diff| vs release dumps):", {k: f"{v:.1e}" for k, v in res["raw_check_max_abs_diff"].items()})
    print("fitted:", json.dumps(res["fitted"]))
    print("dev log loss:", res["dev_log_loss"], "-> choice by protocol:", res["choice_by_protocol"])
    for tier in [t for t in res if isinstance(res[t], dict) and "n" in res[t]]:
        r = res[tier]
        print(f"\n  == {tier} (n {r['n']}, {r['units']} units)")
        print(f"  {'variant':<9}{'accuracy':>22}{'ECE':>22}{'Brier':>22}{'log loss':>22}   minus Ttype: ECE / Brier / log loss")
        for v in VARIANTS:
            m = r[v]; c = lambda k: f"{m[k][0]:.3f} [{m[k][1]:.3f},{m[k][2]:.3f}]"
            d = m["minus_Ttype"]; dd = lambda k: f"{d[k][0]:+.3f} [{d[k][1]:+.3f},{d[k][2]:+.3f}]"
            print(f"  {v:<9}{c('accuracy'):>22}{c('ece'):>22}{c('brier'):>22}{c('log_loss'):>22}   {dd('ece')} / {dd('brier')} / {dd('log_loss')}")
    for tier, per_n in res["Tdomain"].items():
        print(f"\n  Tdomain on {tier} (fit on half A of each family, judged on half B; mean over families)")
        for n, fams in per_n.items():
            mean = lambda k, j: np.mean([f[k][j] for f in fams])
            wins = sum(f["Tdomain"][3] < f["Ttype"][3] for f in fams)
            print(f"   n={n:<4} families {len(fams)}  log loss Tdomain {mean('Tdomain',3):.3f} vs Ttype {mean('Ttype',3):.3f} vs raw {mean('raw',3):.3f}"
                  f" | ECE {mean('Tdomain',1):.3f} vs {mean('Ttype',1):.3f} | Brier {mean('Tdomain',2):.3f} vs {mean('Ttype',2):.3f} | Tdomain better in {wins}/{len(fams)}")


if __name__ == "__main__":
    main()
