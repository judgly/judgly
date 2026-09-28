"""The results figures, from the committed snapshot in docs/results only.

    uv run --group figures python scripts/make_figures.py        (or: make figures)

Writes docs/assets/results/{reliability,selective,families-<pack>}.{svg,png} and CAPTIONS.md, all
on the fresh final tier (`final` in the record: task families frozen before any head was scored
on them, the reported result).

Every plotted number is checked against record.json before anything is written. For every tier
the record holds (test, dev, final, final-flagged, final-seen, bench) and both conditions, the
per-item dumps (items-*.tsv.gz) are rescored with the definitions of csrc/s1_metrics.c and
scripts/calibration_record.py, and must reproduce the record's overall accuracy, log loss, Brier
score and ECE, its reliability bins (count, mean confidence, accuracy, Wilson interval), its
selective accuracy and coverage at each threshold, and the point values of its per-family
accuracy, log loss, Brier score and ECE. The per-family 95% intervals are the record's own: where
a tier's items carry a group (final, final-flagged, bench), calibration_record.py resamples whole
groups, and the groups live in the tier files, not in the snapshot, so they are read from the
record and checked against the rows of tables.md. The heads' SHA-256 in INPUTS.sha256 must equal
the one each record names. A mismatch stops the script before anything is written.

Output is deterministic: fixed rcParams, a fixed SVG hash salt, glyphs drawn as paths, and no
dates or software versions in the file metadata, so a rerun gives byte-identical files.
"""

import gzip
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "docs" / "results"
OUT = ROOT / "docs" / "assets" / "results"
PACKS = {"gemma4-12b-q8": "Gemma 4 12B", "qwen3-4b-q8": "Qwen3-4B"}
FORMATS = ("general", "stance")
FIG_TIER = "final"                                      # the tier every figure shows
SETS = ("fitdev", "final", "final-flagged", "final-seen", "bench")   # names-<set>.tsv.gz
CONDS = ("raw", "h2")
METRICS = ("accuracy", "log_loss", "brier", "ece")
LABEL = {"raw": "raw (no head)", "h2": "fitted head (H2)"}
BINS = 10                                               # as in scripts/calibration_record.py
TOL = 1e-6
# Okabe-Ito colours (colour-blind safe): condition by colour, pack by line style and marker.
COLOUR = {"raw": "#D55E00", "h2": "#0072B2"}
STYLE = {"gemma4-12b-q8": ("-", "o"), "qwen3-4b-q8": ("--", "s")}
GREY = "#7F7F7F"
MIN_ANSWERED = 50       # selective curves are drawn where at least this many questions are answered

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


def check(what: str, mine: float, theirs: float, tol: float = TOL) -> None:
    if abs(mine - theirs) > tol:
        raise Mismatch(f"make_figures: {what}: recomputed {mine:.8f}, record {theirs:.8f}")


# ---- reading the snapshot -----------------------------------------------------------------

def read_dump(path: Path) -> dict[str, np.ndarray]:
    task, fam, label, probs = [], [], [], []
    with gzip.open(path, "rt") as f:
        next(f)
        for line in f:
            _, t, family, _, _, lab, p = line.rstrip("\n").split("\t")
            task.append(int(t))
            fam.append(int(family))
            label.append(int(lab))
            probs.append(np.array([float(v) for v in p.split(",")]))
    top = np.array([int(np.argmax(p)) for p in probs])      # first maximum, as in C
    conf = np.array([p[t] for p, t in zip(probs, top)])
    label = np.array(label)
    return {"task": np.array(task), "family": np.array(fam), "hit": (top == label).astype(float), "conf": conf,
            "nll": np.array([-np.log(max(p[y], 1e-300)) for p, y in zip(probs, label)]),
            "brier": np.array([float(np.sum((p - np.eye(len(p))[y]) ** 2)) for p, y in zip(probs, label)]),
            "bin": np.minimum((conf * BINS).astype(int), BINS - 1)}


def read_names(path: Path) -> dict[tuple[str, int], str]:
    out = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            kind, ident, name = line.rstrip("\n").split("\t")
            out[(kind, int(ident))] = name
    return out


def load(pack: str, fmt: str) -> dict:
    d = SNAP / pack / fmt
    rec = json.loads((d / "record.json").read_text())
    names: dict = {}
    for s in SETS:
        if (d / f"names-{s}.tsv.gz").is_file():
            names |= read_names(d / f"names-{s}.tsv.gz")
    return {"record": rec, "tables": (d / "tables.md").read_text(), "names": names,
            "items": {(c, t): read_dump(d / f"items-{c}-{t}.tsv.gz") for t in rec["tiers"] for c in CONDS}}


# ---- metrics, as in scripts/calibration_record.py -----------------------------------------

def metrics(x: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, float]:
    hit, conf, b = x["hit"][idx], x["conf"][idx], x["bin"][idx]
    ece = np.abs(np.bincount(b, conf, BINS) - np.bincount(b, hit, BINS)).sum() / len(idx)
    return {"accuracy": float(hit.mean()), "log_loss": float(x["nll"][idx].mean()),
            "brier": float(x["brier"][idx].mean()), "ece": float(ece)}


def wilson(k: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    centre, half = p + z * z / (2 * n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / (1 + z * z / n), (centre + half) / (1 + z * z / n)


def reliability(x: dict[str, np.ndarray]) -> list[dict]:
    out = []
    for b in range(BINS):
        m = x["bin"] == b
        n = int(m.sum())
        if n == 0:
            out.append({"bin": b, "n": 0})
            continue
        lo, hi = wilson(x["hit"][m].sum(), n)
        out.append({"bin": b, "n": n, "confidence": float(x["conf"][m].mean()),
                    "accuracy": float(x["hit"][m].mean()), "wilson_lo": float(lo), "wilson_hi": float(hi)})
    return out


def selective(x: dict[str, np.ndarray], threshold: float) -> tuple[float | None, float]:
    keep = x["conf"] > threshold
    return (float(x["hit"][keep].mean()) if keep.any() else None), float(keep.mean())


def cell(m: dict) -> str:
    return f"{m['value']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]"


def n_items(f: dict) -> str:
    return f"{f['items']} ({f['groups']} groups)" if f.get("groups") else str(f["items"])


def verify(pack: str, fmt: str, run: dict, head_sha: dict[str, str]) -> dict:
    """Rescore every tier and check it against record.json and tables.md; returns, per family of
    the figure tier, its task names."""
    rec, where = run["record"], f"{pack}/{fmt}"
    if rec["pack"] != pack or rec["format"] != fmt or rec["quick"]:
        raise Mismatch(f"make_figures: {where}: record is for {rec['pack']}/{rec['format']} quick={rec['quick']}")
    if FIG_TIER not in rec["tiers"]:
        raise Mismatch(f"make_figures: {where}: the record has no {FIG_TIER} tier")
    if head_sha[f"{pack}/{fmt}"] != rec["head"]["sha256"]:
        raise Mismatch(f"make_figures: {where}: h2.bin in INPUTS.sha256 is not the head the record names")
    tasks: dict[str, set[str]] = {}
    for tier in rec["tiers"]:
        for cond in CONDS:
            x, c = run["items"][(cond, tier)], rec["tiers"][tier]["conditions"][cond]
            if len(x["hit"]) != c["items"]:
                raise Mismatch(f"make_figures: {where} {cond}-{tier}: {len(x['hit'])} items, record {c['items']}")
            for name, v in metrics(x, np.arange(len(x["hit"]))).items():
                check(f"{where} {cond}-{tier} {name}", v, c["metrics"][name]["value"])
            for mine, theirs in zip(reliability(x), c["reliability"], strict=True):
                if mine["n"] != theirs["n"]:
                    raise Mismatch(f"make_figures: {where} {cond}-{tier} bin {mine['bin']}: n {mine['n']} vs {theirs['n']}")
                for k in ("confidence", "accuracy", "wilson_lo", "wilson_hi"):
                    if mine["n"]:
                        check(f"{where} {cond}-{tier} bin {mine['bin']} {k}", mine[k], theirs[k])
            for s in c["selective"]:
                a, cov = selective(x, s["threshold"])
                check(f"{where} {cond}-{tier} coverage at {s['threshold']}", cov, s["coverage"])
                if a is None or s["accuracy"] is None:
                    if a != s["accuracy"]:
                        raise Mismatch(f"make_figures: {where} {cond}-{tier} selective at {s['threshold']}")
                else:
                    check(f"{where} {cond}-{tier} selective accuracy at {s['threshold']}", a, s["accuracy"])
            groups: dict[str, list[int]] = {}
            for i, f in enumerate(x["family"]):
                groups.setdefault(run["names"][("family", int(f))], []).append(i)
            if sorted(groups) != sorted(c["by_family"]):
                raise Mismatch(f"make_figures: {where} {cond}-{tier}: families {sorted(groups)} vs "
                               f"{sorted(c['by_family'])}")
            for fam, ix in groups.items():
                theirs = c["by_family"][fam]
                if len(ix) != theirs["items"]:
                    raise Mismatch(f"make_figures: {where} {cond}-{tier} family {fam}: {len(ix)} items, "
                                   f"record {theirs['items']}")
                for name, v in metrics(x, np.array(ix)).items():
                    check(f"{where} {cond}-{tier} family {fam} {name}", v, theirs[name]["value"])
                if tier == FIG_TIER:
                    tasks.setdefault(fam, set()).update(run["names"][("task", int(t))] for t in x["task"][ix])
        raw, h2 = (rec["tiers"][tier]["conditions"][c]["by_family"] for c in CONDS)
        for fam, h in h2.items():
            row = (f"| {tier} | {fam} | {n_items(h)} | {cell(raw[fam]['accuracy'])} | {cell(h['accuracy'])} | "
                   f"{cell(raw[fam]['ece'])} | {cell(h['ece'])} |")
            if row not in run["tables"]:
                raise Mismatch(f"make_figures: {where}: tables.md lacks the row {row}")
    return {fam: sorted(t) for fam, t in tasks.items()}


# ---- figures ------------------------------------------------------------------------------

def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.svg", metadata={"Date": None, "Creator": None, "Format": None, "Type": None})
    fig.savefig(OUT / f"{name}.png", dpi=150, metadata={"Software": None})
    plt.close(fig)


def fig_reliability(runs: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.0), sharex=True, sharey=True)
    for r, pack in enumerate(PACKS):
        for c, fmt in enumerate(FORMATS):
            ax, rec = axes[r][c], runs[(pack, fmt)]["record"]["tiers"][FIG_TIER]
            ax.plot([0, 1], [0, 1], color=GREY, linestyle=":", linewidth=1, label="perfect calibration")
            for k, cond in enumerate(CONDS):
                bins = [b for b in rec["conditions"][cond]["reliability"] if b["n"] > 0]
                conf = np.array([b["confidence"] for b in bins])
                acc = np.array([b["accuracy"] for b in bins])
                err = np.array([[b["accuracy"] - b["wilson_lo"] for b in bins],
                                [b["wilson_hi"] - b["accuracy"] for b in bins]])
                ece = rec["conditions"][cond]["metrics"]["ece"]["value"]
                ax.errorbar(conf, acc, yerr=err, color=COLOUR[cond], marker="os"[k], linestyle="-",
                            capsize=2, elinewidth=0.9, label=f"{LABEL[cond]}, ECE {ece:.3f}")
            ax.set_title(f"{PACKS[pack]}, {fmt}, fresh final tier (n = {rec['items']:,})")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.legend(loc="upper left", frameon=False)
            if r == 1:
                ax.set_xlabel("mean confidence in bin (top probability)")
            if c == 0:
                ax.set_ylabel("accuracy in bin (share correct)")
    fig.tight_layout()
    save(fig, "reliability")


def fig_selective(runs: dict) -> None:
    grid = np.linspace(0, 1, 1001)[:-1]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), sharey=True)
    for c, fmt in enumerate(FORMATS):
        ax = axes[c]
        for pack in PACKS:
            run = runs[(pack, fmt)]
            for cond in CONDS:
                x = run["items"][(cond, FIG_TIER)]
                pts = [selective(x, t) for t in grid]
                pts = [p for p in pts if p[1] * len(x["hit"]) >= MIN_ANSWERED]
                cov = np.array([p[1] for p in pts])
                acc = np.array([p[0] for p in pts])
                ls, mk = STYLE[pack]
                ax.plot(100 * cov, 100 * acc, color=COLOUR[cond], linestyle=ls,
                        label=f"{PACKS[pack]}, {LABEL[cond]}")
                sel = [s for s in run["record"]["tiers"][FIG_TIER]["conditions"][cond]["selective"]
                       if s["coverage"] * len(x["hit"]) >= MIN_ANSWERED]
                ax.plot([100 * s["coverage"] for s in sel], [100 * s["accuracy"] for s in sel],
                        color=COLOUR[cond], marker=mk, linestyle="none")
        n = runs[(next(iter(PACKS)), fmt)]["record"]["tiers"][FIG_TIER]["items"]
        ax.set_title(f"{fmt}, fresh final tier (n = {n:,} per model)")
        ax.set_xlabel("share of questions answered (%)")
        ax.set_xlim(0, 100)
        if c == 0:
            ax.set_ylabel("accuracy on answered questions (%)")
    axes[0].set_ylim(30, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save(fig, "selective")


def fig_families(pack: str, runs: dict, tasks: dict) -> None:
    rows = []   # (label, stats per cond): general families, then the stance family
    for fmt in FORMATS:
        fams = runs[(pack, fmt)]["record"]["tiers"][FIG_TIER]["conditions"]
        for fam in sorted(fams["h2"]["by_family"]):
            names = ", ".join(tasks[(pack, fmt)][fam])
            label = f"{fam}: {names}" if fmt == "general" else f"stance: {names}"
            rows.append((f"{label} (n = {fams['h2']['by_family'][fam]['items']:,})",
                         {c: fams[c]["by_family"][fam] for c in CONDS}))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 0.42 * len(rows) + 1.5), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, metric, xlabel in ((axes[0], "accuracy", "accuracy (share correct)"),
                               (axes[1], "ece", "expected calibration error (ECE)")):
        for k, cond in enumerate(CONDS):
            v = np.array([s[cond][metric]["value"] for _, s in rows])
            lo = np.array([s[cond][metric]["lo"] for _, s in rows])
            hi = np.array([s[cond][metric]["hi"] for _, s in rows])
            ax.errorbar(v, y + (0.15 if k == 0 else -0.15), xerr=[v - lo, hi - v], color=COLOUR[cond],
                        marker="os"[k], linestyle="none", capsize=2, elinewidth=1, label=LABEL[cond])
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlim(0, 1)
    top = max(s[c]["ece"]["hi"] for _, s in rows for c in CONDS)
    axes[1].set_xlim(0, max(0.5, np.ceil(top * 10 + 0.2) / 10))
    axes[0].set_yticks(y, [label for label, _ in rows])
    axes[0].set_ylim(-0.7, len(rows) - 0.3)
    fig.suptitle(f"{PACKS[pack]}, fresh final tier by family, 95% bootstrap intervals", fontsize=10.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, f"families-{pack}")


def captions(runs: dict) -> str:
    def tier(pack: str, fmt: str) -> dict:
        return runs[(pack, fmt)]["record"]["tiers"][FIG_TIER]

    def n(pack: str, fmt: str) -> str:
        t = tier(pack, fmt)
        return f"n = {t['items']:,}" + (f" in {t['groups']:,} groups" if t.get("groups") else "")

    def ece(pack: str, fmt: str, cond: str) -> str:
        return cell(tier(pack, fmt)["conditions"][cond]["metrics"]["ece"])

    first = next(iter(PACKS))
    out = ["# Results figures", "",
           "Made by `make figures` (scripts/make_figures.py) from the snapshot in docs/results only; every",
           "plotted number is checked against record.json. All figures show the fresh final tier: task",
           "families frozen before any head was scored on them, and never used to fit or choose a head.",
           "Brackets are 95% intervals; on this tier the bootstrap resamples whole groups of items that",
           "share a claim, a table, a template or a query.", "",
           "## reliability.svg", "",
           "Reliability diagrams on the fresh final tier: for ten equal-width bins of the top probability, the",
           "mean confidence (x) against the share of correct answers (y), with 95% Wilson intervals; empty",
           "bins are left out (the Wilson intervals treat items as independent). On the dotted diagonal, confidence equals accuracy. "
           + "; ".join(f"{PACKS[p]} {f} ({n(p, f)}): ECE raw {ece(p, f, 'raw')}, head {ece(p, f, 'h2')}"
                       for p in PACKS for f in FORMATS) + ".", "",
           "## selective.svg", "",
           "Selective accuracy on the fresh final tier: answer only when the top probability is above a",
           "threshold, swept from 0 to 0.999; x is the share of questions answered, y the accuracy on those.",
           "Markers are the record's thresholds 0.5, 0.6, 0.7, 0.8, 0.9, 0.95 and 0.99. Curves stop where",
           f"fewer than {MIN_ANSWERED} questions are answered. The raw curves stop far from 0%: without a head the",
           "models give many answers a top probability above 0.999, so no threshold can hold them back. "
           + "; ".join(f"{f}: {n(first, f)} per model" for f in FORMATS) + ".", ""]
    for pack in PACKS:
        out += [f"## families-{pack}.svg", "",
                f"{PACKS[pack]}, fresh final tier by task family: accuracy and ECE, raw and with the fitted",
                "head, with the calibration record's 95% percentile bootstrap intervals (1,000 resamples of",
                "whole groups). Each label names the family, its tasks and its number of items. The stance",
                "format's fresh final tier is one family (the Check-COVID task). The flagged families",
                "(politeness, HealthFC) and the families seen during development are not shown; their",
                "numbers are in the tables of docs/results.", ""]
    return "\n".join(out)


def main() -> None:
    head_sha = {}
    for line in (SNAP / "INPUTS.sha256").read_text().splitlines():
        sha, path = line.split("  ", 1)
        parts = Path(path).parts
        if parts[-1] == "h2.bin":
            head_sha[f"{parts[-3]}/{parts[-2]}"] = sha
    runs = {(p, f): load(p, f) for p in PACKS for f in FORMATS}
    tasks = {(p, f): verify(p, f, runs[(p, f)], head_sha) for p in PACKS for f in FORMATS}
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(RC)
    fig_reliability(runs)
    fig_selective(runs)
    for pack in PACKS:
        fig_families(pack, runs, tasks)
    (OUT / "CAPTIONS.md").write_text(captions(runs))
    print(f"make_figures: checked against record.json and tables.md; wrote {len(list(OUT.iterdir()))} files "
          f"in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
