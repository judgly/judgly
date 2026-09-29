"""Score the confirmation exactly as CONFIRM.md (with amendment 1) says. Written before the
confirmation data was read; smoke-tested on the 0.1.0 dev tier first.

    uv run --no-project --with numpy python score_confirm.py JUDGLY_ROOT RESULTS_DIR TIER_NAME TIERFILE_NAME

RESULTS_DIR holds <pack>/<format>/items-{raw,h2}-<TIER_NAME>.tsv; the tier file is
data/tiers/<format>/<TIERFILE_NAME>.jsonl.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT, RES, TIER, TFILE = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
TEMPS = json.load(open(HERE / "temperatures.json"))
PACKS = ["gemma4-12b-q8", "qwen3-4b-q8"]
EPS = 1e-12
NAMES = ("accuracy", "ece", "brier", "log_loss")
# criteria on the paired difference Ttype - H2 (95% bootstrap interval)
CRIT = {"accuracy": ("lo", -0.01), "brier": ("hi", 0.01), "ece": ("hi", 0.02)}
FLAGGED = ["argument_quality", "humour", "code_outcome"]


def fnv(text):
    h = 0xcbf29ce484222325
    for c in text.encode():
        h = ((h ^ c) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def read_dump(path):
    ids, typ, lab, P = [], [], [], []
    with open(path) as f:
        next(f)
        for line in f:
            h, _, _, t, k, y, p = line.rstrip("\n").split("\t")
            v = np.full(26, np.nan); v[:int(k)] = [float(x) for x in p.split(",")]
            ids.append(h); typ.append(int(t)); lab.append(int(y)); P.append(v)
    return ids, np.array(typ), np.array(lab), np.array(P)


def ttype(P, typ, temps):
    L = np.log(np.clip(P, EPS, 1)); L[np.isnan(P)] = -np.inf
    Z = L / np.array([temps[str(t)] for t in typ])[:, None]
    Z = Z - np.nanmax(np.where(np.isfinite(Z), Z, np.nan), axis=1, keepdims=True)
    E = np.exp(Z); E[~np.isfinite(Z)] = 0
    return E / E.sum(axis=1, keepdims=True)


def per_item(P, y):
    Pz = np.nan_to_num(P); top = Pz.argmax(axis=1); conf = Pz.max(axis=1)
    oh = np.zeros_like(Pz); oh[np.arange(len(y)), y] = 1
    return np.stack([(top == y).astype(float), conf, ((Pz - oh) ** 2).sum(axis=1),
                     -np.log(np.clip(Pz[np.arange(len(y)), y], EPS, 1))])


def metrics(x, idx):
    right, conf, brier, ll = x[:, idx]
    b = np.minimum((conf * 10).astype(int), 9)
    ece = sum(abs(right[b == k].mean() - conf[b == k].mean()) * (b == k).sum() for k in range(10) if (b == k).any()) / len(b)
    return np.array([right.mean(), ece, brier.mean(), ll.mean()])


def boot(xa, xb, grp, n=1000, seed=20260929):
    rng = np.random.default_rng(seed)
    units, inv = np.unique(grp, return_inverse=True)
    members = [np.where(inv == u)[0] for u in range(len(units))]
    everything = np.arange(xa.shape[1])
    pa, pb = metrics(xa, everything), metrics(xb, everything)
    D = []
    for _ in range(n):
        idx = np.concatenate([members[u] for u in rng.integers(0, len(units), len(units))])
        D.append(metrics(xa, idx) - metrics(xb, idx))
    lo, hi = np.percentile(D, [2.5, 97.5], axis=0)
    return pa, pb, pa - pb, lo, hi, len(units)


def verdict(d, lo, hi):
    out = {}
    for m, (side, bound) in CRIT.items():
        k = NAMES.index(m)
        out[m] = bool(lo[k] > bound) if side == "lo" else bool(hi[k] < bound)
    return out


def analyse(ids, typ, y, P_raw, P_h2, temps, grp):
    xt, xh, xr = per_item(ttype(P_raw, typ, temps), y), per_item(P_h2, y), per_item(P_raw, y)
    pt, ph, d, lo, hi, units = boot(xt, xh, grp)
    pr = metrics(xr, np.arange(len(y)))
    return {"n": len(y), "units": units,
            "raw": dict(zip(NAMES, pr.round(4).tolist())), "Ttype": dict(zip(NAMES, pt.round(4).tolist())),
            "H2": dict(zip(NAMES, ph.round(4).tolist())),
            "Ttype_minus_H2": {m: [round(d[k], 4), round(lo[k], 4), round(hi[k], 4)] for k, m in enumerate(NAMES)},
            "criteria_met": verdict(d, lo, hi)}


def main():
    report, confirmed = {}, 0
    for pack in PACKS:
        for fmt in ("general", "stance"):
            d = RES / pack / fmt
            ids, typ, y, P_raw = read_dump(d / f"items-raw-{TIER}.tsv")
            ids2, _, y2, P_h2 = read_dump(d / f"items-h2-{TIER}.tsv")
            assert ids == ids2 and (y == y2).all()
            rows = {fnv(json.loads(l)["id"]): json.loads(l) for l in open(ROOT / "data" / "tiers" / fmt / f"{TFILE}.jsonl")}
            rows = {h: r for h, r in rows.items() if h in set(ids)}
            assert len(rows) == len(ids), "items missing from the tier file"
            key = lambda h, f: str(rows[h].get(f) or rows[h]["id"])
            grp = np.array([key(h, "group") for h in ids])
            temps = TEMPS[pack][fmt]
            r = {"primary": analyse(ids, typ, y, P_raw, P_h2, temps, grp)}
            r["confirmed"] = all(r["primary"]["criteria_met"].values())
            confirmed += r["confirmed"]
            fams = np.array([rows[h]["family"] for h in ids])
            r["per_family"] = {}
            for fam in np.unique(fams):
                m = fams == fam
                r["per_family"][fam] = analyse([i for i, k in zip(ids, m) if k], typ[m], y[m], P_raw[m], P_h2[m], temps, grp[m])
            if fmt == "stance" and any("claim_group" in rows[h] for h in ids):
                r["secondary_claim_groups"] = analyse(ids, typ, y, P_raw, P_h2, temps, np.array([key(h, "claim_group") for h in ids]))
            if fmt == "general":
                r["leave_one_out"] = {}
                for fam in FLAGGED:
                    m = fams != fam
                    if m.all():
                        continue
                    r["leave_one_out"][f"without {fam}"] = analyse([i for i, k in zip(ids, m) if k], typ[m], y[m],
                                                                   P_raw[m], P_h2[m], temps, grp[m])
            report[f"{pack}/{fmt}"] = r
    report["cases_confirmed"] = f"{confirmed} of 4"
    out = HERE / f"result-{TIER}.json"
    json.dump(report, open(out, "w"), indent=1)
    for k, r in report.items():
        if not isinstance(r, dict):
            continue
        p = r["primary"]
        print(f"\n### {k}: n {p['n']} ({p['units']} units)  CONFIRMED: {r['confirmed']}  {p['criteria_met']}")
        for v in ("raw", "Ttype", "H2"):
            print(f"   {v:<6} " + "  ".join(f"{m} {p[v][m]:.3f}" for m in NAMES))
        print("   Ttype-H2 " + "  ".join(f"{m} {a:+.3f} [{b:+.3f},{c:+.3f}]" for m, (a, b, c) in p["Ttype_minus_H2"].items()))
        for sec in ("secondary_claim_groups",):
            if sec in r:
                print(f"   {sec}: {r[sec]['criteria_met']}")
        for name, s in r.get("leave_one_out", {}).items():
            print(f"   {name}: criteria {s['criteria_met']}  acc diff {s['Ttype_minus_H2']['accuracy']}")
        for fam, s in r["per_family"].items():
            dd = s["Ttype_minus_H2"]
            print(f"   family {fam:<18} n {s['n']:>4}  acc T {s['Ttype']['accuracy']:.3f} H2 {s['H2']['accuracy']:.3f} | ECE T {s['Ttype']['ece']:.3f} H2 {s['H2']['ece']:.3f} | Brier T {s['Ttype']['brier']:.3f} H2 {s['H2']['brier']:.3f}")
    print("\ncases confirmed:", report["cases_confirmed"])


if __name__ == "__main__":
    main()
