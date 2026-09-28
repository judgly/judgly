"""Final-tier statistics beyond the calibration records, from the committed snapshot alone.

    uv run --only-group figures python docs/tools/final_tier_stats.py

For each pack and format, on the fresh final tier: the per-family average ECE next to the
pooled ECE, and paired bootstrap differences head minus raw. Intervals are 95% percentiles of
1,000 resamples, items resampled within each family (not by group, so they may be somewhat
narrower than the grouped intervals of the calibration records), seed 20260927. ECE is as in csrc/s1_metrics.c: 10 equal-width bins on the top
probability.
"""

import gzip
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2] / "docs" / "results"
rng = np.random.default_rng(20260927)


def load(pack: str, fmt: str, cond: str) -> list[tuple]:
    rows = []
    with gzip.open(R / pack / fmt / f"items-{cond}-final.tsv.gz", "rt") as f:
        next(f)
        for line in f:
            h, _task, fam, _typ, _k, lab, p = line.rstrip("\n").split("\t")
            rows.append((h, int(fam), int(lab), np.array([float(x) for x in p.split(",")])))
    return sorted(rows, key=lambda r: r[0])


def arrays(rows: list[tuple]) -> tuple[np.ndarray, ...]:
    conf = np.array([r[3].max() for r in rows])
    corr = np.array([r[3].argmax() == r[2] for r in rows], float)
    ll = np.array([-np.log(max(r[3][r[2]], 1e-12)) for r in rows])
    return conf, corr, ll


def ece(conf: np.ndarray, corr: np.ndarray) -> float:
    b = np.minimum((conf * 10).astype(int), 9)
    e = 0.0
    for k in range(10):
        m = b == k
        if m.any():
            e += m.sum() * abs(conf[m].mean() - corr[m].mean())
    return e / len(conf)


def ci(v: list[float]) -> str:
    lo, hi = np.percentile(v, [2.5, 97.5])
    return f"[{lo:.3f}, {hi:.3f}]"


def stats(a: dict, fam: np.ndarray, ix: np.ndarray) -> dict:
    s = {}
    for c in ("raw", "h2"):
        conf, corr, ll = (x[ix] for x in a[c])
        fm = fam[ix]
        s[c] = {"acc": corr.mean(), "ll": ll.mean(), "ece": ece(conf, corr),
                "fam": np.mean([ece(conf[fm == f], corr[fm == f]) for f in np.unique(fm)])}
    return s


def main() -> None:
    for pack in ("gemma4-12b-q8", "qwen3-4b-q8"):
        for fmt in ("general", "stance"):
            raw, h2 = load(pack, fmt, "raw"), load(pack, fmt, "h2")
            assert [r[0] for r in raw] == [r[0] for r in h2]
            a = {"raw": arrays(raw), "h2": arrays(h2)}
            fam = np.array([r[1] for r in raw])
            idx = np.arange(len(raw))
            point = stats(a, fam, idx)
            boot = []
            for _ in range(1000):
                ix = np.concatenate([rng.choice(idx[fam == f], (fam == f).sum())
                                     for f in np.unique(fam)])
                boot.append(stats(a, fam, ix))
            print(f"\n{pack} {fmt}, final tier, n = {len(raw)}, families: {len(np.unique(fam))}")
            for c in ("raw", "h2"):
                p = point[c]
                print(f"  {c:4} pooled ECE {p['ece']:.3f}  per-family average ECE {p['fam']:.3f} "
                      f"{ci([b[c]['fam'] for b in boot])}")
            for m, label in (("acc", "accuracy"), ("ll", "log loss"), ("ece", "ECE")):
                d = point["h2"][m] - point["raw"][m]
                print(f"  head - raw {label}: {d:+.3f} "
                      f"{ci([b['h2'][m] - b['raw'][m] for b in boot])}")


if __name__ == "__main__":
    main()
