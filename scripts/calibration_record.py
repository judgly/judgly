"""The calibration record of one pack and format, and its tables, from s1-eval's output.

    uv run --group pipeline python scripts/calibration_record.py FORMAT_DIR DATA_DIR PACK FORMAT QUICK

FORMAT_DIR is RESULTS/<pack>/<format> as the Makefile writes it: h2.bin, train-h2.log,
temperature.bin, train-temperature.log, and per condition (raw, h2, temperature) and tier (test,
dev, final, final-flagged, confirm, final-seen, bench; a tier the format does not have is skipped,
and a QUICK run never reads final, final-flagged or confirm) the s1-eval report
<cond>-<tier>.json and its per-item dump items-<cond>-<tier>.tsv; fitdev/, final/,
final-flagged/, confirm/, final-seen/ and bench/ hold the feature files and their names sidecars.
Writes FORMAT_DIR/record.json (machine-readable) and FORMAT_DIR/tables.md.

The two calibration options are H2 (the head, `head` in the record) and the per-type temperature
(`temperature`); both are fitted on the fit tier's train split and scored on the same items.
confirm is the untouched tier of the pre-registered comparison of the two (docs/calibration.md):
read once, by both, after both were frozen; the numbers here are recomputed from those same
readouts with the same frozen head and temperatures.

final is the fresh final tier (the reported result); final-flagged holds fresh families with a
recorded caveat (the registry's `caveat`), reported in their own table beside final and never
pooled into it; final-seen is an earlier held-out tier whose families were read during development, and is reported
in its own table as a secondary evaluation of previously seen families; bench holds external
benchmarks, which are also scored as each benchmark defines against its own gold (`benchmark` in
the record; see bench_metrics).

Overall metrics are s1-eval's own. Their 95% bootstrap intervals (1,000 resamples) are s1-eval's
where items are independent (test, dev, final-seen); where the tier file gives each item a
`group` (final, final-flagged, bench: items sharing a claim, an abstract, a table, a template or
a case), the intervals of accuracy, log loss, Brier score and ECE are recomputed here by
resampling whole groups, and s1-eval's item-level intervals are kept beside them
(`metrics_item_resampled`). The per-family numbers are recomputed here from the per-item dumps
with the same definitions (csrc/s1_metrics.c) and their own percentile bootstrap (by group where
there are groups); as a check, the overall numbers are recomputed the same way and must equal
s1-eval's, or the script fails. The record has no
timestamps (tiers are identified by their files' SHA-256, not the manifest, which holds the build
date), so the same inputs give the same file. Provenance: the model and template SHA-256 and engine
settings (from RESULTS/<pack>/pack-info.mk), the self-test gate's output, the judgly and llama.cpp
commits (dirty if any tracked file differs from the commit) and the platform.
"""

import json
import platform
import subprocess
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import registry  # noqa: E402

BINS, RESAMPLES, SEED = 10, 1000, 20260926
TIERS = {"test": ("fitdev", "test"), "dev": ("fitdev", "heldout"), "final": ("final", "heldout"),
         "final-flagged": ("final-flagged", "heldout"), "confirm": ("confirm", "heldout"),
         "final-seen": ("final-seen", "heldout"), "bench": ("bench", "heldout")}
FRESH = ("final", "final-flagged", "confirm")       # never read by a QUICK run
METRICS = ("accuracy", "log_loss", "brier", "ece")
TIER_NOTE = {"test": "in-distribution: the fit tier's test split",
             "dev": "held-out families scored during development",
             "final": "fresh held-out families, frozen before any head was scored on them: the reported result",
             "final-flagged": "fresh held-out families with a recorded caveat (see `caveats`): reported beside "
                              "the final result, never pooled into it, judging no bar",
             "confirm": "untouched families (general) and ClimateCheck (stance), built after every other tier and "
                        "read once, after both calibration options were frozen, for the pre-registered comparison "
                        "of the temperature with H2 (docs/calibration.md); `caveats` lists the families the "
                        "review flagged",
             "final-seen": "secondary evaluation: an earlier held-out tier, its families seen during development",
             "bench": "external benchmarks, evaluation only (also scored against their own gold in `benchmark`)"}
CONDITIONS = {"raw": "no head (H0): letter probabilities with the pack's engine settings (model.engine)",
              "h2": "the fitted H2 head, with the pack's engine settings (model.engine)",
              "temperature": "the per-type temperature applied to the raw probabilities averaged over the option "
                             "orders, with the pack's engine settings (model.engine)"}
FITTED = ("h2", "temperature")


def fnv(text: str) -> int:
    h = 0xcbf29ce484222325
    for c in text.encode():
        h = ((h ^ c) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def read_dump(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            h, task, fam, typ, k, label, p = line.rstrip("\n").split("\t")
            rows.append({"id_hash": int(h, 16), "task_id": int(task), "family_id": int(fam), "type": int(typ),
                         "label": int(label), "p": np.array([float(v) for v in p.split(",")])})
    return rows


def names(path: Path) -> dict[tuple[str, int], str]:
    out = {}
    for line in open(path):
        kind, ident, name = line.rstrip("\n").split("\t")
        out[(kind, int(ident))] = name
    return out


def per_item(rows: list[dict]) -> dict[str, np.ndarray]:
    top = np.array([int(np.argmax(r["p"])) for r in rows])     # first maximum, as in C
    conf = np.array([r["p"][t] for r, t in zip(rows, top)])
    label = np.array([r["label"] for r in rows])
    return {"hit": (top == label).astype(float), "conf": conf,
            "nll": np.array([-np.log(max(r["p"][r["label"]], 1e-300)) for r in rows]),
            "brier": np.array([float(np.sum((r["p"] - np.eye(len(r["p"]))[r["label"]]) ** 2)) for r in rows]),
            "bin": np.minimum((conf * BINS).astype(int), BINS - 1)}


def metrics(x: dict[str, np.ndarray], idx: np.ndarray) -> tuple[float, float, float, float]:
    hit, conf, b = x["hit"][idx], x["conf"][idx], x["bin"][idx]
    ece = np.abs(np.bincount(b, conf, BINS) - np.bincount(b, hit, BINS)).sum() / len(idx)
    return float(hit.mean()), float(x["nll"][idx].mean()), float(x["brier"][idx].mean()), float(ece)


def with_interval(x: dict[str, np.ndarray], idx: np.ndarray, rng: np.random.Generator,
                  group: list[str] | None = None) -> dict:
    """Point values and 95% percentile intervals over `idx`, resampling items, or whole groups
    when `group` (the group of every item, by position) is given."""
    point = metrics(x, idx)
    if group is None:
        draws = np.array([metrics(x, rng.choice(idx, size=len(idx), replace=True)) for _ in range(RESAMPLES)])
    else:
        members: dict[str, list[int]] = defaultdict(list)
        for i in idx:
            members[group[i]].append(int(i))
        blocks = [np.array(v) for _, v in sorted(members.items())]
        draws = np.array([metrics(x, np.concatenate([blocks[k] for k in rng.choice(len(blocks), len(blocks))]))
                          for _ in range(RESAMPLES)])
    lo, hi = np.quantile(draws, 0.025, axis=0), np.quantile(draws, 0.975, axis=0)
    return {m: {"value": round(point[i], 6), "lo": round(float(lo[i]), 6), "hi": round(float(hi[i]), 6)}
            for i, m in enumerate(METRICS)}


def bench_metrics(rows: list[dict], examples: dict, rng: np.random.Generator) -> dict:
    """Each benchmark (bench family) scored against its own gold, as the benchmark defines it:
    accuracy (argmax against the gold label; for a JevBench score question, the rounded
    expected level, as JevBench scores ordinal questions), multi-class Brier score (the sum over
    options of the squared difference from the gold distribution: typed-decisions' soft
    teacher distribution, JevBench's expected label or published gold probabilities), KL
    divergence from the gold distribution to the prediction, ECE (top label, 10 equal-width
    bins) and, for score questions, the mean absolute difference between the predicted and the
    gold expected level. 95% percentile intervals from 1,000 bootstrap resamples of cases
    (items sharing a state or a paraphrase group are resampled together)."""
    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        ex = examples[r["id_hash"]]
        by_family[ex["family"]].append({"p": r["p"], "ex": ex})

    def one(items: list[dict]) -> dict:
        hit, brier, kl, conf, mae = [], [], [], [], []
        for x in items:
            p, ex = np.clip(x["p"], 1e-12, 1.0), x["ex"]
            g = np.array(ex["gold"], dtype=float)
            top = int(np.argmax(x["p"]))
            if ex["type"] == "score":
                ev, gev = float(np.dot(np.arange(len(p)), x["p"])), float(np.dot(np.arange(len(g)), g))
                mae.append(abs(ev - gev))
                right = round(ev) == int(ex["label"]) - 1 if ex["source"] == "jevbench" else top == int(ex["label"]) - 1
            else:
                keys = list(ex["options"]) if ex["type"] == "choice" else ["true", "false"]
                right = keys[top] == ex["label"]
            hit.append(float(right))
            brier.append(float(np.sum((x["p"] - g) ** 2)))
            kl.append(float(np.sum(np.where(g > 0, g * np.log(np.where(g > 0, g, 1.0) / p), 0.0))))
            conf.append(float(x["p"][top]))
        hit_a, conf_a = np.array(hit), np.array(conf)
        b = np.minimum((conf_a * BINS).astype(int), BINS - 1)
        ece = float(np.abs(np.bincount(b, conf_a, BINS) - np.bincount(b, hit_a, BINS)).sum() / len(items))
        return {"accuracy": float(hit_a.mean()), "brier": float(np.mean(brier)), "kl": float(np.mean(kl)), "ece": ece,
                "score_mae": float(np.mean(mae)) if mae else None}

    out = {}
    for fam, items in sorted(by_family.items()):
        groups: dict[str, list[dict]] = defaultdict(list)
        for x in items:
            groups[x["ex"].get("group") or x["ex"]["id"]].append(x)
        keys = sorted(groups)
        point = one(items)
        draws = [one([x for k in rng.choice(len(keys), len(keys)) for x in groups[keys[k]]]) for _ in range(RESAMPLES)]
        out[fam] = {"items": len(items), "cases": len(keys)}
        for m, v in point.items():
            vals = [d[m] for d in draws if d[m] is not None]
            out[fam][m] = None if v is None else {"value": round(v, 6), "lo": round(float(np.quantile(vals, 0.025)), 6),
                                                   "hi": round(float(np.quantile(vals, 0.975)), 6)}
        out[fam]["by_type"] = {typ: round(float(np.mean([one([x])["accuracy"] for x in items if x["ex"]["type"] == typ])), 6)
                               for typ in sorted({x["ex"]["type"] for x in items})}
    return out


def licence_used(src: dict) -> list[str]:
    """The licence the source is used under: the recorded decision's `treat_as` where there is
    one (the card is wrong or incomplete), else the card's licence tokens."""
    decision = src.get("licence_decision")
    return [decision["treat_as"]] if isinstance(decision, dict) and decision.get("treat_as") else src["licence"]


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True,
                              cwd=registry.ROOT).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sysctl(name: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_info(out: Path) -> dict:
    """The model and engine settings of the run, from RESULTS/<pack>/pack-info.mk, with the
    template's SHA-256 (paths are left out: they differ between machines)."""
    info = {}
    path = out / "pack-info.mk"
    if path.is_file():
        for line in path.read_text().splitlines():
            key, _, value = line.partition(" := ")
            info[key] = value
    template = Path(info["TEMPLATE"]) if "TEMPLATE" in info else None
    return {"sha256": info.get("MODEL_SHA256"), "file": Path(info["MODEL_PATH"]).name if "MODEL_PATH" in info else None,
            "template_sha256": registry.sha256(template) if template and template.is_file() else None,
            "engine": {"rotations": info.get("ENGINE_ROTATIONS") == "1",
                       "content_free": info.get("ENGINE_CONTENT_FREE") == "1",
                       "max_rotations": int(info.get("ENGINE_MAX_ROTATIONS", 0))}}


def main(fdir: Path, data: Path, pack: str, fmt: str, quick: bool) -> None:
    reg = registry.load()
    manifest = json.loads((data / "manifest.json").read_text())
    examples = {}
    for part in sorted({part for part, _ in TIERS.values()}):
        if not (data / fmt / f"{part}.jsonl").is_file():
            continue
        for line in open(data / fmt / f"{part}.jsonl", encoding="utf-8"):
            ex = json.loads(line)
            examples[fnv(ex["id"])] = ex
    main_rng = np.random.default_rng(SEED)
    tiers = {}
    for tier, (part, split) in TIERS.items():
        if not (fdir / f"raw-{tier}.json").is_file() or (quick and tier in FRESH):
            continue
        nm = names(fdir / part / "features.feat.names.tsv")
        t: dict = {"note": TIER_NOTE[tier], "conditions": {}}
        for cond in CONDITIONS:
            # raw and h2 on the 0.1.0 tiers draw from the one stream they drew from in 0.1.0, in the
            # same order, so their intervals are unchanged; every other pair has its own stream
            rng = main_rng if cond in ("raw", "h2") and tier != "confirm" else \
                np.random.default_rng([SEED, zlib.crc32(f"{tier}/{cond}".encode())])
            report = json.loads((fdir / f"{cond}-{tier}.json").read_text())
            rows = read_dump(fdir / f"items-{cond}-{tier}.tsv")
            x = per_item(rows)
            everything = np.arange(len(rows))
            mine = metrics(x, everything)
            for i, m in enumerate(METRICS):
                if abs(mine[i] - report["metrics"][m]["value"]) > 1e-6:
                    raise SystemExit(f"calibration_record: {cond}-{tier} {m}: recomputed {mine[i]:.8f}, "
                                     f"s1-eval {report['metrics'][m]['value']:.8f}")
            groups = [f'{ex["source"]}:{ex["group"]}' if ex.get("group") else None
                      for ex in (examples[r["id_hash"]] for r in rows)]
            group = groups if all(groups) else None
            if any(groups) and group is None:
                raise SystemExit(f"calibration_record: {cond}-{tier}: some items have a group and some do not")
            fams = defaultdict(list)
            for i, r in enumerate(rows):
                fams[nm[("family", r["family_id"])]].append(i)
            by_family = {f: {"items": len(ix), **({"groups": len({group[i] for i in ix})} if group else {}),
                             **with_interval(x, np.array(ix), rng, group)} for f, ix in sorted(fams.items())}
            metrics_ = report["metrics"]
            entry = {"items": report["items"], "interval_unit": "group" if group else "item"}
            if group:
                grouped = with_interval(x, everything, rng, group)
                metrics_ = {m: (v | {"lo": grouped[m]["lo"], "hi": grouped[m]["hi"]} if m in grouped else v)
                            for m, v in report["metrics"].items()}
                entry |= {"groups": len(set(group)),
                          "metrics_item_resampled": {m: report["metrics"][m] for m in METRICS}}
            entry |= {"metrics": metrics_, "reliability": report["reliability"],
                      "selective": report["risk_coverage"], "by_family": by_family}
            if tier == "bench":
                entry["benchmark"] = bench_metrics(rows, examples, rng)
            if fmt == "stance":
                conf = defaultdict(Counter)
                for r in rows:
                    ex = examples[r["id_hash"]]
                    keys = list(ex["options"])
                    conf[ex["label"]][keys[int(np.argmax(r["p"]))]] += 1
                entry["confusion"] = {g: dict(sorted(c.items())) for g, c in sorted(conf.items())}
            t["conditions"][cond] = entry
            t["items"] = report["items"]
            t["groups"] = entry.get("groups")
        srcs = Counter(ex["source"] for ex in examples.values()
                       if ex["split"] == split and (ex["split"] != "heldout" or
                                                    registry.tier_of(reg, ex["source"]) == tier))
        t["sources"] = dict(sorted(srcs.items()))
        if tier == "final-flagged":
            t["caveats"] = {s: reg["sources"][s]["caveat"] for s in t["sources"]}
        if tier == "confirm":
            t["caveats"] = {s: reg["sources"][s]["confirm_caveat"] for s in t["sources"]
                            if reg["sources"][s].get("confirm_caveat")}
        tiers[tier] = t

    fit_sources = sorted({ex["source"] for ex in examples.values() if ex["split"] in ("train", "validation")})
    head = fdir / "h2.bin"
    sidecar = json.loads((fdir / "h2.bin.json").read_text())   # per type: fitted, or the identity and why
    temp = fdir / "temperature.bin"
    temp_sidecar = json.loads((fdir / "temperature.bin.json").read_text())
    fitted_on = [{"source": s, "licence": licence_used(reg["sources"][s]), "card_licence": reg["sources"][s]["licence"],
                  "citation": reg["sources"][s]["citation"],
                  "items": sum(1 for ex in examples.values() if ex["source"] == s and ex["split"] == "train")}
                 for s in fit_sources]
    record = {
        "schema": 3, "pack": pack, "format": fmt, "quick": quick,
        "head": {"kind": "H2", "sha256": registry.sha256(head), "licence": reg["policy"]["head_licence"][fmt],
                 "fitted_on": fitted_on,
                 "engine": sidecar["engine"],
                 "types": sidecar["types"],
                 "train_log": (fdir / "train-h2.log").read_text().splitlines()[-12:]},
        "temperature": {"kind": "temperature", "sha256": registry.sha256(temp),
                        "licence": reg["policy"]["head_licence"][fmt],
                        "applied": "p_T = softmax(log(max(p, 1e-12)) / T), T of the question's type, p the raw "
                                   "probabilities averaged over the option orders",
                        "fitted": "per question type, the T in [0.05, 100] minimising the mean log loss of the fit "
                                  "tier's train items (each item once, unweighted), rounded to three decimals; the "
                                  "validation split only for the verdict (fallback to T = 1)",
                        "fitted_on": "the same train items as head",
                        "engine": temp_sidecar["engine"],
                        "types": temp_sidecar["types"],
                        "train_log": (fdir / "train-temperature.log").read_text().splitlines()},
        "conditions": CONDITIONS,
        "data": {"registry_sha256": manifest["registry_sha256"], "files": manifest["formats"][fmt]["files"],
                 "seed": manifest["seed"], "settings": manifest["formats"][fmt]["settings"]},
        "model": run_info(fdir.parent),
        "software": {"judgly_commit": git("rev-parse", "HEAD"),
                     "judgly_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
                     "llama_cpp_commit": git("rev-parse", "HEAD:third_party/llama.cpp")},
        "platform": {"system": platform.platform(), "machine": platform.machine(),
                     "cpu": sysctl("machdep.cpu.brand_string"), "python": platform.python_version()},
        "selftest": (fdir.parent / "selftest.txt").read_text().splitlines()
                    if (fdir.parent / "selftest.txt").is_file() else None,
        "metrics_note": "accuracy, log loss, Brier score and ECE (10 equal-width bins on the top probability) with "
                        "95% percentile bootstrap intervals, resampling items, or whole groups where a tier's items "
                        "carry one (interval_unit; groups is their number); 'selective' is accuracy and coverage "
                        "when only answers whose top probability is above the threshold (strictly greater) are kept",
        "tiers": tiers,
    }
    (fdir / "record.json").write_text(json.dumps(record, indent=1) + "\n")
    (fdir / "tables.md").write_text(tables(record))
    print(tables(record))


def cell(m: dict) -> str:
    return f"{m['value']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]"


def n_items(t: dict) -> str:
    """Items, with the number of groups the intervals resample where there are groups."""
    return f"{t['items']} ({t['groups']} groups)" if t.get("groups") else str(t["items"])


def tables(r: dict) -> str:
    out = [f"# {r['pack']} / {r['format']}" + (" (QUICK smoke run: not results)" if r["quick"] else ""), ""]
    main_tiers = [t for t in ("test", "dev", "final") if t in r["tiers"]]
    out += ["Conditions: raw (no calibration), h2 (the fitted head) and temperature (one temperature per question",
            "type, applied after the mean over option orders). Both are fitted on the fit tier's train split.", ""]
    out += ["| tier | items | condition | accuracy | log loss | ECE |", "|---|---|---|---|---|---|"]
    for tier in main_tiers:
        t = r["tiers"][tier]
        for cond, c in t["conditions"].items():
            m = c["metrics"]
            out.append(f"| {tier} | {n_items(t)} | {cond} | {cell(m['accuracy'])} | {cell(m['log_loss'])} | {cell(m['ece'])} |")
    if "final-flagged" in r["tiers"]:
        t = r["tiers"]["final-flagged"]
        out += ["", "## Fresh families with a recorded caveat (final-flagged), reported beside final", "",
                "Held out like the final tier and read with it, but not pooled into its numbers and judging no bar:", ""]
        out += [f"- {s}: {c}" for s, c in t["caveats"].items()]
        out += ["", "| tier | items | condition | accuracy | log loss | ECE |", "|---|---|---|---|---|---|"]
        for cond, c in t["conditions"].items():
            m = c["metrics"]
            out.append(f"| final-flagged | {n_items(t)} | {cond} | {cell(m['accuracy'])} | {cell(m['log_loss'])} | "
                       f"{cell(m['ece'])} |")
    if "confirm" in r["tiers"]:
        t = r["tiers"]["confirm"]
        out += ["", "## Confirmation tier (confirm): untouched families, read once", "",
                "Built after every other tier and read once, by both calibration options, after both were frozen",
                "(the pre-registered comparison in docs/calibration.md). Families the review flagged:", ""]
        out += [f"- {s}: {c}" for s, c in t.get("caveats", {}).items()]
        out += ["", "| tier | items | condition | accuracy | log loss | ECE |", "|---|---|---|---|---|---|"]
        for cond, c in t["conditions"].items():
            m = c["metrics"]
            out.append(f"| confirm | {n_items(t)} | {cond} | {cell(m['accuracy'])} | {cell(m['log_loss'])} | "
                       f"{cell(m['ece'])} |")
    if "final-seen" in r["tiers"]:
        t = r["tiers"]["final-seen"]
        out += ["", "## Secondary evaluation: held-out families seen during development (final-seen)", "",
                "An earlier held-out tier whose families were read while judgly was developed, so these numbers are",
                "not a held-out result; they are reported as a secondary evaluation.", "",
                "| tier | items | condition | accuracy | log loss | ECE |", "|---|---|---|---|---|---|"]
        for cond, c in t["conditions"].items():
            m = c["metrics"]
            out.append(f"| final-seen | {t['items']} | {cond} | {cell(m['accuracy'])} | {cell(m['log_loss'])} | {cell(m['ece'])} |")
    if "bench" in r["tiers"]:
        t = r["tiers"]["bench"]
        out += ["", "## External benchmarks (bench), scored against their own gold", "",
                "| benchmark | items (cases) | condition | accuracy | Brier | KL | ECE | score MAE |",
                "|---|---|---|---|---|---|---|---|"]
        for cond, c in t["conditions"].items():
            for fam, b in c["benchmark"].items():
                out.append(f"| {fam} | {b['items']} ({b['cases']}) | {cond} | {cell(b['accuracy'])} | {cell(b['brier'])} | {cell(b['kl'])} | "
                           f"{cell(b['ece'])} | {cell(b['score_mae']) if b['score_mae'] else 'n/a'} |")
    sel_tier = "final" if "final" in r["tiers"] else main_tiers[-1]
    for cond in FITTED:
        thresholds = r["tiers"][sel_tier]["conditions"][cond]["selective"]
        out += ["", f"## Selective accuracy ({cond}): accuracy / share answered", "",
                "| tier | " + " | ".join(f"{s['threshold']:.2f}" for s in thresholds) + " |",
                "|---|" + "---|" * len(thresholds)]
        for tier, t in r["tiers"].items():
            out.append(f"| {tier} | " + " | ".join("n/a / 0%" if s["accuracy"] is None else
                                                   f"{s['accuracy']:.2f} / {s['coverage']:.0%}"
                                                   for s in t["conditions"][cond]["selective"]) + " |")
    out += ["", "## By family", "",
            "| tier | family | items | raw accuracy | h2 accuracy | temperature accuracy | raw ECE | h2 ECE | "
            "temperature ECE |", "|---|---|---|---|---|---|---|---|---|"]
    for tier, t in r["tiers"].items():
        for fam, h in t["conditions"]["h2"]["by_family"].items():
            raw, tt = (t["conditions"][c]["by_family"][fam] for c in ("raw", "temperature"))
            out.append(f"| {tier} | {fam} | {n_items(h)} | {cell(raw['accuracy'])} | {cell(h['accuracy'])} | "
                       f"{cell(tt['accuracy'])} | {cell(raw['ece'])} | {cell(h['ece'])} | {cell(tt['ece'])} |")
    if r["format"] == "stance":
        for cond in FITTED:
            out += ["", f"## Confusion ({cond}): gold (rows) by predicted (columns)", ""]
            for tier, t in r["tiers"].items():
                conf = t["conditions"][cond]["confusion"]
                keys = sorted({k for c in conf.values() for k in c} | set(conf))
                out += [f"**{tier}**", "", "| gold | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
                out += [f"| {g} | " + " | ".join(str(conf.get(g, {}).get(k, 0)) for k in keys) + " |" for g in keys]
                out.append("")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5] == "1")
