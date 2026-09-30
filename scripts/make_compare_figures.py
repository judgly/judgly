"""The external-comparison figures, from the committed record in docs/results/external-comparison.

    make data                      # the tier files (not committed; checked against data/tiers*.sha256)
    uv run --group figures python scripts/make_compare_figures.py        (or: make compare-figures)

Writes docs/assets/results/{compare-tiers,compare-reliability,compare-timing}.{svg,png}, the bench-by-source
intervals docs/assets/results/compare-by-source.json, the control's by-type point values
docs/assets/results/compare-control-by-type.json, and the comparison section of CAPTIONS.md
(the text after the marker below; make_figures.py keeps it).

compare-tiers shows each external model twice: as served, and with the equal-calibration control
(external-comparison/calibrated: a per-type temperature fitted for every system on the same
sample of judgly's training split; the temperatures are read from its result.json).

What is plotted comes from the record, and nothing is written before all of it has been checked:

1. Every item is rescored from the committed answers (answers/<model>/*.jsonl.gz) and judgly's
   committed per-item dumps (docs/results/<pack>/<format>/items-*.tsv.gz), with the definitions of
   the frozen scorer (score_external.py, checked against PROTOCOL.sha256): the item sets, errors,
   units, and every point value of every system, family and bench source must equal each
   result-*.json in final/.
2. The frozen scorer's bootstrap is replayed (one generator seeded 20260929, the tiers in the
   scorer's order, 1,000 resamples of groups), and every interval and paired difference must
   equal the record's. The same resampling then gives what the record does not hold: intervals
   and paired differences for bench split by source (typed-decisions, JevBench), each source
   resampled on its own with a generator seeded 20260930 (written to compare-by-source.json).
3. The reliability bins of judgly's defaults on the confirm tiers must equal those in judgly's
   calibration records (docs/results/<pack>/<format>/record.json).
4. The equal-calibration control (calibrated/, frozen files checked against its PROTOCOL.sha256):
   the per-type temperatures are fitted again as its scorer fits them (scipy's bounded minimiser,
   which is why make compare-figures adds scipy 1.18.1 to the figures group) from the committed
   training answers (calibrated/answers) and judgly's committed train readout
   (calibrated/judgly-train), and must equal result.json's (4 decimals); they are applied to the
   committed test answers and dumps, the control's bootstrap is replayed (seed 20260929, the tiers
   and models in its scorer's order), and every point value, interval and paired difference of
   every system in result.json must be reproduced. The bench points of the calibrated models are
   then split by source as above (seed 20260930), and the external models' point values are split
   by question type (choice, yes/no, score), as served and calibrated, with the temperature
   applied and the mean top probability (compare-control-by-type.json; not in the record).

Output is deterministic (the rcParams of make_figures.py, a fixed SVG hash salt, no dates or
versions in the metadata), so a rerun gives byte-identical files.
"""

import gzip
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "docs" / "results"
REC = SNAP / "external-comparison"
FINAL = REC / "final"
CONTROL = REC / "calibrated"
OUT = ROOT / "docs" / "assets" / "results"
MARKER = "<!-- external comparison: written by scripts/make_compare_figures.py -->"

MODELS = ("nimble:9b", "tev1:4b", "tev1:0.8b")                  # the order of the recorded "all" run
FILES = {"nimble:9b": "nimble_9b", "tev1:4b": "tev1_4b", "tev1:0.8b": "tev1_0.8b", "all": "all"}
RUNS = {"nimble:9b": ["nimble:9b"], "tev1:4b": ["tev1:4b"], "tev1:0.8b": ["tev1:0.8b"], "all": list(MODELS)}
# As in score_external.py:
EPS = 1e-12
CTX = {"tev1:4b": 2050, "tev1:0.8b": 2050, "nimble:9b": 8194}
TIERS = [("general", "confirm"), ("stance", "confirm"), ("general", "final"), ("stance", "final"),
         ("general", "bench"), ("general", "final-flagged"), ("stance", "final-flagged")]
JUDGLY = {"gemma": "gemma4-12b-q8", "qwen": "qwen3-4b-q8"}
DEFAULT = {("gemma", "general"): "h2", ("gemma", "stance"): "temperature",
           ("qwen", "general"): "temperature", ("qwen", "stance"): "temperature"}
NAMES = ("accuracy", "ece", "brier", "log_loss")
SEED, SOURCE_SEED, DRAWS = 20260929, 20260930, 1000

# The test sets of the figures: (label, format/tier in the record, bench source or None).
SETS = [("confirm, general", "general/confirm", None),
        ("confirm, stance", "stance/confirm", None),
        ("final, general", "general/final", None),
        ("final, stance", "stance/final", None),
        ("typed-decisions", "general/bench", "typed_decisions"),
        ("JevBench, public items", "general/bench", "jevbench")]
SHOWN = ("nimble:9b", "tev1:4b", "tev1:0.8b", "judgly-gemma-default", "judgly-qwen-default")
CALIBRATED = tuple(f"{m} calibrated" for m in MODELS)   # the equal-calibration control's rows
FAINT = ("judgly-gemma-raw", "judgly-qwen-raw")
LABEL = {"nimble:9b": "Nimble 9B", "tev1:4b": "Tev1 4B", "tev1:0.8b": "Tev1 0.8B",
         "judgly-gemma-default": "judgly Gemma 4 12B, default", "judgly-qwen-default": "judgly Qwen3-4B, default",
         "judgly-gemma-raw": "judgly Gemma 4 12B, raw", "judgly-qwen-raw": "judgly Qwen3-4B, raw"}
LABEL |= {f"{m} calibrated": f"{LABEL[m]}, equal calibration" for m in MODELS}
SERVED_LABEL = {m: f"{LABEL[m]}, as served" for m in MODELS}
# judgly in blues (Gemma dark, Qwen light), the other models in greys told apart by shade and
# marker: filled with the equal calibration, hollow and faint as served; judgly raw in its
# pack's blue, hollow and faint.
COLOUR = {"nimble:9b": "#3C3C3C", "tev1:4b": "#7A7A7A", "tev1:0.8b": "#ABABAB",
          "judgly-gemma-default": "#0072B2", "judgly-qwen-default": "#56B4E9",
          "judgly-gemma-raw": "#0072B2", "judgly-qwen-raw": "#56B4E9"}
COLOUR |= {f"{m} calibrated": COLOUR[m] for m in MODELS}
HIGHLIGHT = ("judgly-gemma-default", "judgly-qwen-default")
MARK = {"nimble:9b": "D", "tev1:4b": "^", "tev1:0.8b": "v", "judgly-gemma-default": "o",
        "judgly-qwen-default": "s", "judgly-gemma-raw": "o", "judgly-qwen-raw": "s"}
MARK |= {f"{m} calibrated": MARK[m] for m in MODELS}
# As in calibrated/score_calibrated.py:
C_PACKS = {"gemma": "gemma4-12b-q8", "qwen": "qwen3-4b-q8"}
C_TYPES = {0: "choice", 1: "bool", 2: "score"}
RELIABILITY = ("nimble:9b", "tev1:4b", "judgly-gemma-default", "judgly-qwen-default")
GREY = "#7F7F7F"
BINS = 10
TOL = 1e-9

RC = {
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#E6E6E6", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "svg.fonttype": "path", "svg.hashsalt": "judgly-results", "path.simplify": False,
    "lines.linewidth": 1.6, "lines.markersize": 4.5,
}


class Mismatch(SystemExit):
    pass


def same(what: str, mine, theirs) -> None:
    if isinstance(theirs, list):
        if len(mine) != len(theirs):
            raise Mismatch(f"make_compare_figures: {what}: {mine} vs record {theirs}")
        for a, b in zip(mine, theirs):
            same(what, a, b)
    elif abs(float(mine) - float(theirs)) > TOL:
        raise Mismatch(f"make_compare_figures: {what}: recomputed {mine}, record {theirs}")


# ---- the inputs, checked --------------------------------------------------------------------

def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_frozen(where: Path = REC) -> None:
    want = {}
    for line in (where / "PROTOCOL.sha256").read_text().splitlines():
        m = re.match(r"^([0-9a-f]{64})  (\S+)$", line)
        if m:
            want[m.group(2)] = m.group(1)
    for name, sha in want.items():
        if hashlib.sha256((where / name).read_bytes()).hexdigest() != sha:
            raise Mismatch(f"make_compare_figures: {where.name}/{name} does not match PROTOCOL.sha256")


def fnv(text: str) -> str:
    h = 0xcbf29ce484222325
    for c in text.encode():
        h = ((h ^ c) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def keys_of(item: dict) -> list[str]:
    if item["type"] == "choice":
        return list(item["options"])
    if item["type"] == "bool":
        return ["true", "false"]
    return [str(i) for i in range(1, item["levels"] + 1)]


def gold_index(item: dict) -> int:
    k = keys_of(item)
    return k.index(str(item["label"]).lower() if item["type"] == "bool" else str(item["label"]))


def external_probs(item: dict, resp: dict) -> np.ndarray:
    a = resp["answers"]["q"]
    if item["type"] == "choice":
        return np.array([a["probabilities"][k] for k in item["options"]], float)
    if item["type"] == "bool":
        return np.array([a["noul"], 1 - a["noul"]], float)
    pr = a["probabilities"]
    return np.array([pr[str(i)] for i in range(item["levels"])], float)


def judgly_dump(pack: str, fmt: str, cond: str, tier: str) -> dict[str, tuple[np.ndarray, int, int]]:
    """{id hash: (probabilities, gold index, question type)} from a committed per-item dump."""
    out = {}
    with gzip.open(SNAP / pack / fmt / f"items-{cond}-{tier}.tsv.gz", "rt") as f:
        next(f)
        for line in f:
            h, _, _, t, _, y, p = line.rstrip("\n").split("\t")
            out[h] = (np.array([float(v) for v in p.split(",")]), int(y), int(t))
    return out


def per_item(P: list[np.ndarray], y: list[int]) -> np.ndarray:
    top = np.array([int(np.argmax(p)) for p in P])
    conf = np.array([p.max() for p in P])
    brier = np.array([np.sum((p - np.eye(len(p))[g]) ** 2) for p, g in zip(P, y)])
    ll = np.array([-np.log(max(p[g], EPS)) for p, g in zip(P, y)])
    return np.stack([(top == np.array(y)).astype(float), conf, brier, ll])


def metrics(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    right, conf, brier, ll = x[:, idx]
    b = np.minimum((conf * 10).astype(int), 9)
    ece = sum(abs(right[b == k].mean() - conf[b == k].mean()) * (b == k).sum()
              for k in range(10) if (b == k).any()) / len(b)
    return np.array([right.mean(), ece, brier.mean(), ll.mean()])


def boot(xs: dict, grp: np.ndarray, rng: np.random.Generator) -> tuple[dict, int]:
    units, inv = np.unique(grp, return_inverse=True)
    members = [np.where(inv == u)[0] for u in range(len(units))]
    draws = {k: [] for k in xs}
    for _ in range(DRAWS):
        idx = np.concatenate([members[u] for u in rng.integers(0, len(units), len(units))])
        for k, x in xs.items():
            draws[k].append(metrics(x, idx))
    return draws, len(units)


def rows(xs: dict, draws: dict, idx: np.ndarray,
         refs: tuple[str, ...] = ("judgly-gemma-default", "judgly-qwen-default"), prefix: str = "minus_") -> dict:
    """Point values, intervals and paired differences, as the frozen scorers round them."""
    out = {}
    for k, x in xs.items():
        p = metrics(x, idx)
        lo, hi = np.percentile(draws[k], [2.5, 97.5], axis=0)
        row = {m: [round(p[j], 4), round(lo[j], 4), round(hi[j], 4)] for j, m in enumerate(NAMES)}
        for ref in refs:
            if ref in xs and k != ref:
                d = np.array(draws[k]) - np.array(draws[ref])
                pd = p - metrics(xs[ref], idx)
                dlo, dhi = np.percentile(d, [2.5, 97.5], axis=0)
                row[f"{prefix}{ref}"] = {m: [round(pd[j], 4), round(dlo[j], 4), round(dhi[j], 4)]
                                         for j, m in enumerate(NAMES)}
        out[k] = {m: (v if isinstance(v, dict) else [float(a) for a in v]) for m, v in row.items()}
        for m, v in out[k].items():
            if isinstance(v, dict):
                out[k][m] = {a: [float(c) for c in b] for a, b in v.items()}
    return out


def rescore(tiers: dict, judgly: dict, run: str, record: dict) -> tuple[dict, dict]:
    """Rescore one recorded scorer run and check it against its result file; return the per-item
    arrays of each tier and the by-source intervals of bench."""
    rng = np.random.default_rng(SEED)
    where = f"result-{FILES[run]}.json"
    per_tier, by_source = {}, {}
    for fmt, tier in TIERS:
        key = f"{fmt}/{tier}"
        items, answers = tiers[key]["items"], tiers[key]["answers"]
        byid = {x["id"]: x for x in items}
        systems = {m: answers[m] for m in RUNS[run]}
        common = [x["id"] for x in items
                  if all(x["id"] in a and a[x["id"]]["error"] is None for a in systems.values())]
        rec = record[key]
        same(f"{where} {key} n_items", len(items), rec["n_items"])
        same(f"{where} {key} n_common", len(common), rec["n_common"])
        for m, a in systems.items():
            same(f"{where} {key} errors {m}", sum(1 for x in items if x["id"] in a and a[x["id"]]["error"]),
                 rec["errors"][m])
            same(f"{where} {key} missing {m}", sum(1 for x in items if x["id"] not in a), rec["missing"][m])
        grp = np.array([str(byid[i].get("group") or i) for i in common])
        y = [gold_index(byid[i]) for i in common]
        xs, fits = {}, {}
        for m, a in systems.items():
            xs[m] = per_item([external_probs(byid[i], a[i]["response"]) for i in common], y)
            fits[m] = np.array([a[i]["response"]["usage"]["input_tokens"] for i in common]) < CTX[m]
        hs = [fnv(i) for i in common]
        for short, pack in JUDGLY.items():
            for cond in ("raw", "h2", "temperature"):
                d = judgly[(pack, fmt, cond, tier)]
                xs[f"judgly-{short}-{cond}"] = per_item([d[h][0] for h in hs], [d[h][1] for h in hs])
            xs[f"judgly-{short}-default"] = xs[f"judgly-{short}-{DEFAULT[(short, fmt)]}"]
        draws, units = boot(xs, grp, rng)
        same(f"{where} {key} units", units, rec["units"])
        everything = np.arange(len(common))
        mine = rows(xs, draws, everything)
        if sorted(mine) != sorted(rec["systems"]):
            raise Mismatch(f"make_compare_figures: {where} {key}: systems {sorted(mine)} vs {sorted(rec['systems'])}")
        for k, row in mine.items():
            theirs = rec["systems"][k]
            for m, v in row.items():
                if isinstance(v, dict):
                    for a, b in v.items():
                        same(f"{where} {key} {k} {m} {a}", b, theirs[m][a])
                else:
                    same(f"{where} {key} {k} {m}", v, theirs[m])
            if k in fits:
                same(f"{where} {key} {k} items over context", int((~fits[k]).sum()), theirs["items_over_context"])
                secs = np.array([systems[k][i]["seconds"] for i in common])
                same(f"{where} {key} {k} latency median", round(float(np.median(secs)), 3), theirs["latency_s"]["median"])
                same(f"{where} {key} {k} latency p95", round(float(np.percentile(secs, 95)), 3), theirs["latency_s"]["p95"])
        fam = np.array([byid[i]["family"] for i in common])
        for f in np.unique(fam):
            for k, x in xs.items():
                same(f"{where} {key} family {f} {k}", np.round(metrics(x, np.where(fam == f)[0]), 4).tolist(),
                     [rec["by_family"][f][k][n] for n in NAMES])
        if tier == "bench":
            src = np.array([byid[i]["source"] for i in common])
            for s in np.unique(src):
                ix = np.where(src == s)[0]
                for k, x in xs.items():
                    same(f"{where} {key} source {s} {k}", np.round(metrics(x, ix), 4).tolist(),
                         [rec["by_source"][s][k][n] for n in NAMES])
                sub = {k: x[:, ix] for k, x in xs.items()}
                d, u = boot(sub, grp[ix], np.random.default_rng(SOURCE_SEED))
                by_source[str(s)] = {"n_items": int(len(ix)), "units": int(u),
                                     "systems": rows(sub, d, np.arange(len(ix)))}
        per_tier[key] = {"xs": xs, "n": len(common), "units": units}
    return per_tier, by_source


def temper(p: np.ndarray, T: float) -> np.ndarray:
    """As calibrated/score_calibrated.py: probabilities proportional to exp(log p / T), p floored at 1e-12."""
    z = np.log(np.clip(p, EPS, 1)) / T
    e = np.exp(z - z.max())
    return e / e.sum()


def check_rows(where: str, mine: dict, theirs: dict) -> None:
    if sorted(mine) != sorted(theirs):
        raise Mismatch(f"make_compare_figures: {where}: systems {sorted(mine)} vs {sorted(theirs)}")
    for k, row in mine.items():
        if sorted(row) != sorted(theirs[k]):
            raise Mismatch(f"make_compare_figures: {where} {k}: fields {sorted(row)} vs {sorted(theirs[k])}")
        for m, v in row.items():
            if isinstance(v, dict):
                for a, b in v.items():
                    same(f"{where} {k} {m} {a}", b, theirs[k][m][a])
            else:
                same(f"{where} {k} {m}", v, theirs[k][m])


def fit(P: list[np.ndarray], y: list[int], types: list[str]) -> dict[str, float]:
    """As calibrated/score_calibrated.py: one temperature per question type, minimising log loss."""
    from scipy.optimize import minimize_scalar
    out = {}
    for t in sorted(set(types)):
        idx = [i for i, x in enumerate(types) if x == t]
        f = lambda lt: -np.mean([np.log(max(temper(P[i], np.exp(lt))[y[i]], EPS)) for i in idx])  # noqa: E731
        out[t] = float(np.exp(minimize_scalar(f, bounds=(-3, 4), method="bounded").x))
    return out


def fit_control(record: dict) -> dict[str, dict[str, float]]:
    """Fit the control's temperatures again from its committed inputs; check them against result.json."""
    sample = json.loads((CONTROL / "sample.json").read_text())
    temps = {}
    for fmt, ids in sample.items():
        rows_ = {r["id"]: r for r in map(json.loads, open(ROOT / "data" / "tiers" / fmt / "fitdev.jsonl"))}
        for m in MODELS:
            with gzip.open(CONTROL / "answers" / FILES[m] / f"{fmt}-train.jsonl.gz", "rt") as f:
                ans = {r["id"]: r for r in map(json.loads, f)}
            ok = [i for i in ids if ans[i]["error"] is None]
            P = [np.clip(external_probs(rows_[i], ans[i]["response"]), EPS, 1) for i in ok]
            temps[f"{m}/{fmt}"] = fit(P, [gold_index(rows_[i]) for i in ok], [rows_[i]["type"] for i in ok])
            rec = record["temperatures"][f"{m}/{fmt}"]
            same(f"calibrated/result.json {m}/{fmt} n_fit, refused", [len(ok), len(ids) - len(ok)],
                 [rec["n_fit"], rec["refused"]])
        for short, pack in C_PACKS.items():
            tr = {}
            with gzip.open(CONTROL / "judgly-train" / f"{pack}-{fmt}-train.tsv.gz", "rt") as f:
                next(f)
                for line in f:
                    h, _, _, t, _, y, p = line.rstrip("\n").split("\t")
                    tr[h] = (np.array([float(v) for v in p.split(",")]), int(y), C_TYPES[int(t)])
            hs = [fnv(i) for i in ids]
            temps[f"judgly-{short}/{fmt} (refit on the sample)"] = fit(
                [tr[h][0] for h in hs], [tr[h][1] for h in hs], [tr[h][2] for h in hs])
    for k, T in temps.items():
        rec = record["temperatures"][k]["T"]
        if sorted(T) != sorted(rec):
            raise Mismatch(f"make_compare_figures: calibrated/result.json {k}: types {sorted(T)} vs {sorted(rec)}")
        for t, v in T.items():
            same(f"calibrated/result.json temperature {k} {t}", round(v, 4), rec[t])
    return temps


def rescore_control(tiers: dict, judgly: dict, record: dict, served: dict) -> tuple[dict, dict, dict]:
    """Reproduce calibrated/result.json (the equal-calibration control) from the committed answers
    and dumps with the temperatures fitted again, replaying its bootstrap; check every row against
    it and the as-served points against each model's own result file. Return {model: {tier: xs}},
    the bench-by-source intervals of the calibrated models and judgly's defaults, and the external
    models' point values by question type."""
    where = "calibrated/result.json"
    temps = fit_control(record)
    rng = np.random.default_rng(SEED)
    per_model, by_source, by_type = {m: {} for m in MODELS}, {}, {}
    refs = ("judgly-gemma default", "judgly-qwen default")
    for fmt, tier in TIERS:
        key = f"{fmt}/{tier}"
        items, answers = tiers[key]["items"], tiers[key]["answers"]
        byid = {x["id"]: x for x in items}
        for m in MODELS:
            ids = [x["id"] for x in items if answers[m][x["id"]]["error"] is None]
            grp = np.array([str(byid[i].get("group") or i) for i in ids])
            raw, cal, y = [], [], []
            for i in ids:
                p = np.clip(external_probs(byid[i], answers[m][i]["response"]), EPS, 1)
                raw.append(p)
                cal.append(temper(p, temps[f"{m}/{fmt}"].get(byid[i]["type"], 1.0)))
                y.append(gold_index(byid[i]))
            xs = {f"{m} as served": per_item(raw, y), f"{m} calibrated": per_item(cal, y)}
            hs = [fnv(i) for i in ids]
            for short, pack in C_PACKS.items():
                d = judgly[(pack, fmt, "raw", tier)]
                yy = [d[h][1] for h in hs]
                xs[f"judgly-{short} raw"] = per_item([d[h][0] for h in hs], yy)
                refit = temps[f"judgly-{short}/{fmt} (refit on the sample)"]
                xs[f"judgly-{short} refit on the sample"] = per_item(
                    [temper(d[h][0], refit.get(C_TYPES[d[h][2]], 1.0)) for h in hs], yy)
                for cond in ("temperature", "h2"):
                    dc = judgly[(pack, fmt, cond, tier)]
                    xs[f"judgly-{short} {cond}"] = per_item([dc[h][0] for h in hs], [dc[h][1] for h in hs])
                xs[f"judgly-{short} default"] = xs[f"judgly-{short} {DEFAULT[(short, fmt)]}"]
            draws, units = boot(xs, grp, rng)
            rec = record["tiers"][key][m]
            same(f"{where} {key} {m} n", len(ids), rec["n"])
            same(f"{where} {key} {m} units", units, rec["units"])
            check_rows(f"{where} {key} {m}", rows(xs, draws, np.arange(len(ids)), refs, "minus "), rec["systems"])
            same(f"{where} {key} {m} as served = result-{FILES[m]}.json",
                 [rec["systems"][f"{m} as served"][n][0] for n in NAMES],
                 [served[m][key]["systems"][m][n][0] for n in NAMES])
            per_model[m][key] = xs
            kinds = np.array([byid[i]["type"] for i in ids])
            for t in ("choice", "bool", "score"):
                ix = np.where(kinds == t)[0]
                if len(ix):
                    by_type.setdefault(key, {}).setdefault(FILES[m], {})[t] = {
                        "n": int(len(ix)), "temperature": round(temps[f"{m}/{fmt}"].get(t, 1.0), 4),
                        **{f"{label} {n}": round(float(v), 4)
                           for label, x in (("as served", xs[f"{m} as served"]), ("calibrated", xs[f"{m} calibrated"]))
                           for n, v in zip(("accuracy", "ece", "brier", "mean top probability"),
                                           [*metrics(x, ix)[:3], x[1, ix].mean()])}}
            if tier == "bench":
                src = np.array([byid[i]["source"] for i in ids])
                by_source[m] = {}
                for s_ in np.unique(src):
                    ix = np.where(src == s_)[0]
                    sub = {k: xs[k][:, ix] for k in (f"{m} calibrated", "judgly-gemma default", "judgly-qwen default")}
                    dd, u = boot(sub, grp[ix], np.random.default_rng(SOURCE_SEED))
                    by_source[m][str(s_)] = {"n_items": int(len(ix)), "units": int(u),
                                             "systems": rows(sub, dd, np.arange(len(ix)), refs, "minus ")}
    return per_model, by_source, by_type


def reliability(x: np.ndarray) -> list[dict]:
    right, conf = x[0], x[1]
    b = np.minimum((conf * BINS).astype(int), BINS - 1)
    out = []
    for k in range(BINS):
        m = b == k
        n = int(m.sum())
        if n == 0:
            out.append({"bin": k, "n": 0})
            continue
        p, z = right[m].mean(), 1.959963984540054
        centre, half = p + z * z / (2 * n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        out.append({"bin": k, "n": n, "confidence": float(conf[m].mean()), "accuracy": float(p),
                    "wilson_lo": float((centre - half) / (1 + z * z / n)),
                    "wilson_hi": float((centre + half) / (1 + z * z / n))})
    return out


def check_reliability(per_tier: dict) -> None:
    """judgly's default bins on the confirm tiers equal those of its calibration records."""
    for fmt in ("general", "stance"):
        for short, pack in JUDGLY.items():
            cond = DEFAULT[(short, fmt)]
            rec = json.loads((SNAP / pack / fmt / "record.json").read_text())["tiers"]["confirm"]["conditions"][cond]
            mine = reliability(per_tier[f"{fmt}/confirm"]["xs"][f"judgly-{short}-default"])
            for a, b in zip(mine, rec["reliability"], strict=True):
                same(f"record.json {pack}/{fmt} confirm {cond} bin {a['bin']} n", a["n"], b["n"])
                if a["n"]:
                    for k in ("confidence", "accuracy", "wilson_lo", "wilson_hi"):
                        if abs(a[k] - b[k]) > 1e-6:
                            raise Mismatch(f"make_compare_figures: record.json {pack}/{fmt} confirm {cond} "
                                           f"bin {a['bin']} {k}: {a[k]} vs {b[k]}")


# ---- figures --------------------------------------------------------------------------------

def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.svg", metadata={"Date": None, "Creator": None, "Format": None, "Type": None})
    fig.savefig(OUT / f"{name}.png", dpi=150, metadata={"Software": None})
    plt.close(fig)


def point(results: dict, by_source: dict, key: str, source: str | None, system: str) -> tuple[dict, int]:
    """(row, n) for one system on one test set: an external model from its own result file (the items
    it answered), with the equal calibration from the control's result on the same items, judgly
    from Nimble's (every item of the tier)."""
    if system in CALIBRATED:
        m = system.removesuffix(" calibrated")
        if source is None:
            block = results["control"]["tiers"][key][m]
            return block["systems"][system], block["n"]
        s = by_source["control"][m][source]
        return s["systems"][system], s["n_items"]
    run = system if system in MODELS else "nimble:9b"
    if source is None:
        return results[run][key]["systems"][system], results[run][key]["n_common"]
    s = by_source[run][source]
    return s["systems"][system], s["n_items"]


def fig_tiers(results: dict, by_source: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 6.9))
    for ax, (label, key, source) in zip(axes.flat, SETS):
        for m in MODELS:   # as served (hollow, faint, no bars) joined to the same model calibrated
            a0, e0 = (lambda r: (r["accuracy"][0], r["ece"][0]))(point(results, by_source, key, source, m)[0])
            a1, e1 = (lambda r: (r["accuracy"][0], r["ece"][0]))(point(results, by_source, key, source, f"{m} calibrated")[0])
            ax.plot([e0, e1], [a0, a1], color=COLOUR[m], linewidth=0.8, alpha=0.5, zorder=2)
            ax.plot([e0], [a0], color=COLOUR[m], marker=MARK[m], linestyle="none", markersize=5.5, alpha=0.55,
                    markerfacecolor="white", markeredgewidth=1.0, zorder=3, label=SERVED_LABEL[m])
        for system in FAINT:
            row, _ = point(results, by_source, key, source, system)
            ax.plot([row["ece"][0]], [row["accuracy"][0]], color=COLOUR[system], marker=MARK[system], linestyle="none",
                    markersize=5.5, alpha=0.45, markerfacecolor="white", markeredgewidth=1.0, zorder=3,
                    label=LABEL[system])
        for system in CALIBRATED + HIGHLIGHT:
            row, _ = point(results, by_source, key, source, system)
            a, e = row["accuracy"], row["ece"]
            ax.errorbar([e[0]], [a[0]], xerr=[[e[0] - e[1]], [e[2] - e[0]]], yerr=[[a[0] - a[1]], [a[2] - a[0]]],
                        color=COLOUR[system], marker=MARK[system], linestyle="none", capsize=2, elinewidth=0.9,
                        markersize=6.5 if system in HIGHLIGHT else 5.5, zorder=5 if system in HIGHLIGHT else 4,
                        label=LABEL[system])
        n = point(results, by_source, key, source, "judgly-gemma-default")[1]
        n_tev = point(results, by_source, key, source, "tev1:4b")[1]
        ax.set_title(f"{label}\n(n = {n:,}" + (f"; Tev1 {n_tev:,}" if n_tev != n else "") + ")")
        ax.set_xlim(0, 0.45)
        ax.set_ylim(0.3, 0.9)
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    for ax in axes[1]:
        ax.set_xlabel("ECE (lower is better)")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy")
    handles, labels = axes[0][0].get_legend_handles_labels()
    blank = plt.Line2D([], [], linestyle="none")
    served = tuple(SERVED_LABEL[m] for m in MODELS)
    calibrated = tuple(LABEL[s] for s in CALIBRATED)
    judgly = tuple(LABEL[s] for s in HIGHLIGHT + FAINT)
    # columns of the legend (filled column by column): external with the equal calibration |
    # external as served | judgly default | judgly raw
    keys = list(calibrated + served + judgly[:2] + (None,) + judgly[2:] + (None,))
    fig.legend([handles[labels.index(s)] if s else blank for s in keys], [s or "" for s in keys],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0), fontsize=8)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save(fig, "compare-tiers")


def fig_reliability(per_tier: dict, results: dict) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(8.4, 4.9), sharex=True, sharey=True)
    for r, fmt in enumerate(("general", "stance")):
        key = f"{fmt}/confirm"
        for c, system in enumerate(RELIABILITY):
            ax = axes[r][c]
            run = system if system in MODELS else "nimble:9b"
            bins = [b for b in reliability(per_tier[run][key]["xs"][system]) if b["n"] > 0]
            ece = results[run][key]["systems"][system]["ece"][0]
            ax.plot([0, 1], [0, 1], color=GREY, linestyle=":", linewidth=1)
            conf = np.array([b["confidence"] for b in bins])
            acc = np.array([b["accuracy"] for b in bins])
            err = np.array([[b["accuracy"] - b["wilson_lo"] for b in bins], [b["wilson_hi"] - b["accuracy"] for b in bins]])
            ax.errorbar(conf, acc, yerr=err, color=COLOUR[system], marker=MARK[system], linestyle="-", capsize=2,
                        elinewidth=0.9, markersize=4, linewidth=2.0 if system in HIGHLIGHT else 1.4)
            ax.text(0.04, 0.93, f"ECE {ece:.3f}", transform=ax.transAxes, fontsize=8.5, va="top")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            name = LABEL[system].replace(", default", "\ndefault")
            ax.set_title(name if r == 0 else "", fontsize=9)
            if c == 0:
                n = per_tier[run][key]["n"]
                ax.set_ylabel(f"{fmt} (n = {n:,})\naccuracy in bin", fontsize=9)
            if r == 1:
                ax.set_xlabel("confidence", fontsize=9)
    fig.tight_layout()
    save(fig, "compare-reliability")


TIMING = {"tev1:0.8b": "tev1:0.8b", "tev1:4b": "tev1:4b", "judgly-qwen-default": "judgly qwen3-4b-q8",
          "nimble:9b": "nimble:9b", "judgly-gemma-default": "judgly gemma4-12b-q8"}


def fig_timing(timing: dict) -> None:
    """Median single-request time per system, with the 95th percentile as a whisker, straight from
    final/timing.json (checked: every system present, 100 timed requests, no refusals)."""
    rows = []
    for system, key in TIMING.items():
        t = timing["systems"][key]
        if t["n"] != 100 or t.get("refused", 0) != 0 or not t["median_s"] <= t["p95_s"]:
            raise Mismatch(f"make_compare_figures: timing for {key} is not 100 answered requests: {t}")
        rows.append((system, t["median_s"], t["p95_s"]))
    if set(timing["systems"]) != set(TIMING.values()):
        raise Mismatch(f"make_compare_figures: timing.json systems {sorted(timing['systems'])}")
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    for y, (system, median, p95) in enumerate(rows):
        ax.barh(y, median, color=COLOUR[system], height=0.62, zorder=3)
        ax.errorbar([median], [y], xerr=[[0], [p95 - median]], color="#222222", capsize=3, elinewidth=0.9,
                    linestyle="none", zorder=4)
        ax.text(p95 + 0.03, y, f"{median:.2f} s", va="center", fontsize=8, fontfamily="DejaVu Sans")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABEL[r[0]].replace(", default", "") for r in rows], fontsize=8.5)
    ax.set_xlabel("seconds per request (bar: median; whisker: 95th percentile)", fontsize=8.5)
    ax.set_xlim(0, 2.0)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "compare-timing")


# ---- captions -------------------------------------------------------------------------------

def captions(results: dict, by_source: dict, timing: dict) -> str:
    def c(v: list) -> str:
        return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"

    sets = []
    for label, key, source in SETS:
        parts = []
        for m in MODELS:
            row, _ = point(results, by_source, key, source, m)
            cal, _ = point(results, by_source, key, source, f"{m} calibrated")
            parts.append(f"{LABEL[m]} accuracy {c(row['accuracy'])}, ECE as served {c(row['ece'])}, "
                         f"with the equal calibration {c(cal['ece'])}")
        for system in HIGHLIGHT:
            row, _ = point(results, by_source, key, source, system)
            parts.append(f"{LABEL[system]} accuracy {c(row['accuracy'])}, ECE {c(row['ece'])}")
        sets.append(f"{label}: " + "; ".join(parts))
    temps = results["control"]["temperatures"]
    kind = {"choice": "choice", "bool": "yes/no", "score": "score"}
    bound = sorted(f"{LABEL[m]} {kind[t]}" for m in MODELS for f in ("general", "stance")
                   for t, v in temps[f"{m}/{f}"]["T"].items() if v > 54.59)
    rel = []
    for fmt in ("general", "stance"):
        rel.append(f"{fmt} (n = {results['nimble:9b'][f'{fmt}/confirm']['n_common']:,}): " + ", ".join(
            f"{LABEL[s]} {results[s if s in MODELS else 'nimble:9b'][f'{fmt}/confirm']['systems'][s]['ece'][0]:.3f}"
            for s in RELIABILITY))
    cal_rel = "; ".join(
        f"{fmt}: " + ", ".join(
            f"{LABEL[m]} {results['control']['tiers'][f'{fmt}/confirm'][m]['systems'][f'{m} calibrated']['ece'][0]:.3f}"
            for m in RELIABILITY if m in MODELS) for fmt in ("general", "stance"))
    jev = by_source["tev1:4b"]["jevbench"]["n_items"]
    return "\n".join([
        MARKER, "",
        "# External comparison figures", "",
        "Made by `make compare-figures` (scripts/make_compare_figures.py) from the record in",
        "docs/results/external-comparison (with its equal-calibration control in calibrated/) and judgly's",
        "committed per-item dumps; every item is rescored and every plotted number checked against the",
        "record's result files before anything is written. The external models were run as served by",
        "Ollama 0.35.0 (`/v1/systemone`); judgly's calibration was fitted by us on similar question types",
        "(see the caveats in the README and in docs/methods.md).", "",
        "## compare-tiers.svg", "",
        "*What it shows:* accuracy (y) against ECE (x) on six test sets. Each of Nimble 9B, Tev1 4B and",
        "Tev1 0.8B appears twice: as served by Ollama (hollow, faint grey, without bars) and with the equal",
        "calibration (filled grey, with bars), joined by a thin line. The equal calibration is temperature",
        "scaling (Guo et al. 2017): one temperature per question type, fitted for each model by the same",
        "code on the same sample of judgly's training split (1,500 general and 500 stance items; the",
        "record in docs/results/external-comparison/calibrated). A temperature never changes which answer",
        "is on top, so both points of a model share one accuracy. judgly's two packs are shown with their",
        "default calibration (filled blue) and raw, without calibration (hollow, faint blue). Bars are 95%",
        "percentile bootstrap intervals over the tier's groups of related items (1,000 resamples): for the",
        "confirm and final tiers they are the record's (score_external.py for the models as served and",
        "judgly, score_calibrated.py for the calibrated models); for typed-decisions and JevBench, which",
        "the record reports together as bench, each source was resampled on its own by this script (seed",
        "20260930; compare-by-source.json). Each external model is scored on the items it answered and",
        "judgly on every item: Tev1 refused 36 JevBench items longer than its context, so in the JevBench",
        f"panel the Tev1 points are on {jev} of the 231 items and judgly's on all 231, different item sets",
        f"(the README and docs/methods.md also give judgly on Tev1's {jev} items, where judgly Gemma 4 12B",
        "is more accurate than on all 231). On the confirm tiers, judgly's defaults are the ones the",
        "pre-registered confirmation selected from its read of this same tier (the temperature where it was",
        "confirmed; H2, the 0.1.0 default, for Gemma 4 12B on general questions). *How to read it:* up and",
        "to the left is better; points whose intervals overlap are not clearly different, and the paired",
        "differences in docs/methods.md are the sharper test. ECE is never negative, so its intervals lean",
        "upward near 0. *What it says:* accuracy comes from the models, and the temperature does not move",
        "it; judgly Gemma 4 12B is the most accurate or level with the most accurate on every set. Given",
        "the same temperature fitted on the same data, the external models come close to judgly's defaults",
        "in ECE: level with them on general confirm, between judgly's two packs or below both on the stance",
        "tiers (Tev1 0.8B on stance confirm excepted), and still above them on general final. Most of",
        "judgly's calibration lead over the models as served therefore came from its calibration step, not",
        "from its models. The temperature does not always carry over to new kinds of questions: it made",
        "Tev1 4B slightly worse calibrated on the final tiers, where it was already well calibrated as",
        "served, and Nimble 9B and Tev1 4B clearly worse on typed-decisions, mostly on its score questions",
        "(compare-control-by-type.json). Some fitted temperatures reached the search's upper bound,",
        "exp(4) = 54.6 (" + ", ".join(bound) + "); at that temperature the answers of that type are almost",
        "flat, so a low ECE there says little about their usefulness.",
        "The numbers: " + ". ".join(sets) + ".", "",
        "## compare-reliability.svg", "",
        "*What it shows:* reliability diagrams on the two confirm tiers for Nimble 9B and Tev1 4B as served",
        "and judgly's two packs with their default calibration: for ten equal-width bins of the top",
        "probability, the mean confidence (x) against the share of correct answers (y), with 95% Wilson",
        "intervals; empty bins are left out. *How to read it:* on the dotted diagonal, confidence equals",
        "accuracy; points below it are overconfident. The Wilson intervals treat items as independent,",
        "which on stance (1,780 items in 70 linked groups) makes them too narrow. judgly's defaults here",
        "were selected by the pre-registered confirmation from its read of this same tier (H2, the 0.1.0",
        "default, stayed for Gemma 4 12B on general questions); nothing was fitted on it. With H2 instead,",
        "ECE was 0.076 (Qwen3-4B) on general and 0.078 (Gemma 4 12B) and 0.183 (Qwen3-4B) on stance.",
        "*What it says:* Nimble 9B and Tev1 4B, as served, are overconfident on these tiers. judgly's",
        "defaults lie closer to the diagonal, but both are overconfident on stance above 0.5 (Gemma 4 12B",
        "by 0.03 to 0.13 per bin, Qwen3-4B by 0.09 to 0.14), and Gemma 4 12B on general questions in the",
        "0.5 to 0.6 bin (by 0.12, 796 items). ECE: " + "; ".join(rel) + ". With the equal calibration",
        "(compare-tiers), the external models' ECE on these tiers was " + cal_rel + ".", "",
        "## compare-timing.svg", "",
        "*What it shows:* the median time per request (bar) and its 95th percentile (whisker) for each",
        "system, from final/timing.json: 100 items (the first 50 of each confirm tier by the SHA-256 of",
        "their id), one question per request, one request at a time, one uncounted warm-up request per",
        "system, on an Apple M3 Max (64 GB). *How to read it:* shorter is faster. The bars do not measure",
        "the same thing: judgly ran in-process through its Python API and asked each question in up to",
        "four option orders; the Ollama models were asked over HTTP and read each question once. The",
        "timings were taken after the comparison run and were not part of its frozen protocol. *What it",
        "says:* medians: " + "; ".join(f"{LABEL[s].replace(', default', '')} {timing['systems'][k]['median_s']:.3f} s"
                                  for s, k in TIMING.items()) + ".", "",
    ])


def write_captions(text: str) -> None:
    path = OUT / "CAPTIONS.md"
    old = path.read_text() if path.is_file() else ""
    head = old.split(MARKER)[0].rstrip("\n")
    path.write_text((head + "\n\n" if head else "") + text)


def main() -> None:
    check_frozen()
    check_frozen(CONTROL)
    reproduce = load_module("reproduce", REC / "reproduce.py")
    reproduce.check_tiers()
    reproduce.check_tiers(("fitdev",))
    tiers = {}
    for fmt, tier in TIERS:
        items = [json.loads(line) for line in open(ROOT / "data" / "tiers" / fmt / f"{tier}.jsonl")]
        answers = {}
        for m in MODELS:
            with gzip.open(REC / "answers" / FILES[m] / f"{fmt}-{tier}.jsonl.gz", "rt") as f:
                answers[m] = {r["id"]: r for r in map(json.loads, f)}
        tiers[f"{fmt}/{tier}"] = {"items": items, "answers": answers}
    judgly = {(pack, fmt, cond, tier): judgly_dump(pack, fmt, cond, tier)
              for pack in JUDGLY.values() for fmt, tier in TIERS for cond in ("raw", "h2", "temperature")}
    results, per_tier, by_source = {}, {}, {}
    for run in RUNS:
        results[run] = json.loads((FINAL / f"result-{FILES[run]}.json").read_text())
        per_tier[run], by_source[run] = rescore(tiers, judgly, run, results[run])
        print(f"make_compare_figures: result-{FILES[run]}.json reproduced (points, intervals, differences)", flush=True)
    check_reliability(per_tier["nimble:9b"])
    results["control"] = json.loads((CONTROL / "result.json").read_text())
    _, by_source["control"], by_type = rescore_control(tiers, judgly, results["control"], results)
    print("make_compare_figures: calibrated/result.json reproduced (points, intervals, differences)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(RC)
    fig_tiers(results, by_source)
    fig_reliability(per_tier, results)
    timing = json.loads((FINAL / "timing.json").read_text())
    fig_timing(timing)
    (OUT / "compare-by-source.json").write_text(json.dumps(
        {"what": "bench split by source: point values (equal to the record's by_source), 95% percentile "
                 "bootstrap intervals and paired differences against judgly's defaults, 1,000 resamples of "
                 f"each source's groups, numpy default_rng({SOURCE_SEED}); per scorer run (the result file "
                 "whose items are used)",
         "runs": {FILES[r]: by_source[r] for r in RUNS},
         "equal calibration": {
             "what": "the equal-calibration control (external-comparison/calibrated): each external model with its "
                     "per-type temperature fitted on the training sample, and judgly's defaults, on the items that "
                     "model answered; the same resampling (seed 20260930 per source); differences are the model "
                     "minus judgly's default",
             "models": {FILES[m]: by_source["control"][m] for m in MODELS}}}, indent=1) + "\n")
    (OUT / "compare-control-by-type.json").write_text(json.dumps(
        {"what": "the equal-calibration control (external-comparison/calibrated), by question type: for each "
                 "external model on the items it answered, the fitted temperature of the type (1 where the "
                 "training sample had none), and accuracy, ECE, Brier and the mean top probability as served "
                 "and calibrated; point values only, recomputed from the committed answers by "
                 "scripts/make_compare_figures.py after it reproduced calibrated/result.json; not in the record",
         "tiers": by_type}, indent=1) + "\n")
    write_captions(captions(results, by_source, timing))
    print(f"make_compare_figures: checked against the record; wrote compare-tiers, compare-reliability, compare-timing, "
          f"compare-by-source.json, compare-control-by-type.json and CAPTIONS.md in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
