"""Rerun the calibration study's analyses from a judgly checkout and compare their output with
the committed output.

The analysis scripts are kept exactly as they were run: they take the root of a judgly checkout
on the command line, read uncompressed per-item dumps under <root>/results/<pack>/<format>/ and
write their output next to themselves. This wrapper does not edit them. It builds a scratch
workspace instead:

    <work>/root/data                        -> the checkout's data/ (the tier files, from make data)
    <work>/root/results/<pack>/<format>/     the committed per-item dumps (docs/results, gunzipped)
                                             and links to the feature directories of a finished
                                             pack run (results/<pack>/<format>/<set>/, CPU only)
    <work>/fit/<pack>/<format>/              train and validation dumps for temperature.py, written
                                             by build/cli/s1-eval with the shipped H2 head
    <work>/<analysis>/                       a copy of the analysis directory; the script runs here

then runs the script with the documented arguments, writes its printed output to the .txt file
where the committed directory has one, and compares every output file with the committed one
(byte for byte; for JSON also the largest numeric difference).

    uv run --no-project --with numpy --with scipy --with scikit-learn \
        python docs/results/calibration-study/rerun.py all [--work DIR] [--results results]

Analyses: combine (exploratory 0), temperature (exploratory 1), tricks (exploratory 2),
confirm (the confirmation's scorer). combine and confirm need only the committed dumps and the
tier files; temperature also needs build/cli/s1-eval and the fit tier's features; tricks needs
the feature files of every tier it reads. Nothing here runs the model.
"""

import argparse
import gzip
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SNAPSHOT = REPO / "docs" / "results"
PACKS = ("gemma4-12b-q8", "qwen3-4b-q8")
FORMATS = ("general", "stance")
FEATURE_SETS = ("fitdev", "final", "final-flagged", "final-seen", "bench")

# analysis -> (directory, script, extra arguments after ROOT, printed output file, output files)
ANALYSES = {
    "combine": ("exploratory-0-combine", "combine.py", lambda w: [], None, ["combine.json"]),
    "temperature": ("exploratory-1-temperature", "temperature.py", lambda w: [str(w / "fit")],
                    "temperature.txt", ["temperature.json"]),
    "tricks": ("exploratory-2-post-processing", "tricks.py", lambda w: [], "tricks.txt", ["tricks.json"]),
    "confirm": ("confirmation", "score_confirm.py", lambda w: [str(w / "root" / "results"), "confirm", "confirm"],
                "result-confirm.txt", ["result-confirm.json"]),
}


def gunzip(src: Path, dst: Path) -> None:
    if not dst.exists():
        with gzip.open(src, "rb") as f, open(dst, "wb") as g:
            shutil.copyfileobj(f, g)


def build_root(work: Path, results: Path) -> Path:
    root = work / "root"
    (root / "results").mkdir(parents=True, exist_ok=True)
    if not (root / "data").exists():
        (root / "data").symlink_to(REPO / "data")
    for pack in PACKS:
        for fmt in FORMATS:
            d = root / "results" / pack / fmt
            d.mkdir(parents=True, exist_ok=True)
            for gz in sorted((SNAPSHOT / pack / fmt).glob("items-*.tsv.gz")):
                gunzip(gz, d / gz.name[:-3])
            for s in FEATURE_SETS:
                src = results / pack / fmt / s
                if src.is_dir() and not (d / s).exists():
                    (d / s).symlink_to(src.resolve())
    return root


def build_fit(work: Path, results: Path) -> None:
    """Train and validation dumps (raw and H2) for temperature.py, as the analysis made them:
    s1-eval --features <fitdev features> --split train|validation --rotations [--head h2] --dump-items."""
    s1eval = REPO / "build" / "cli" / "s1-eval"
    for pack in PACKS:
        for fmt in FORMATS:
            feats = results / pack / fmt / "fitdev" / "features.feat"
            head = REPO / "src" / "judgly" / "packs" / pack / "heads" / ("h2.bin" if fmt == "general" else "h2-stance.bin")
            d = work / "fit" / pack / fmt
            d.mkdir(parents=True, exist_ok=True)
            for split in ("train", "validation"):
                for cond, extra in (("raw", []), ("h2", ["--head", str(head)])):
                    out = d / f"items-{cond}-{split}.tsv"
                    if out.exists():
                        continue
                    subprocess.run([str(s1eval), "--features", str(feats), "--split", split, "--rotations", *extra,
                                    "--dump-items", str(out)], check=True, stdout=subprocess.DEVNULL)


def numeric_diff(a, b, path=""):
    """Largest absolute difference between the numbers of two JSON values, and the first path
    whose structure or non-numeric value differs (None if none)."""
    if isinstance(a, bool) or isinstance(b, bool) or not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                return 0.0, f"{path}: keys differ"
            worst, first = 0.0, None
            for k in a:
                w, f = numeric_diff(a[k], b[k], f"{path}/{k}")
                worst, first = max(worst, w), first or f
            return worst, first
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return 0.0, f"{path}: lengths differ"
            worst, first = 0.0, None
            for i, (x, y) in enumerate(zip(a, b)):
                w, f = numeric_diff(x, y, f"{path}/{i}")
                worst, first = max(worst, w), first or f
            return worst, first
        return 0.0, (None if a == b else f"{path}: {a!r} != {b!r}")
    return abs(a - b), None


def run(name: str, work: Path, root: Path) -> bool:
    directory, script, extra, printed, outputs = ANALYSES[name]
    committed = HERE / directory
    here = work / directory
    if here.exists():
        shutil.rmtree(here)
    shutil.copytree(committed, here, ignore=shutil.ignore_patterns("__pycache__", *outputs, *([printed] if printed else [])))
    cmd = [sys.executable, str(here / script), str(root), *extra(work)]
    print(f"== {name}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=here, check=True, capture_output=True, text=True)
    if printed:
        (here / printed).write_text(proc.stdout)
    same = True
    for f in outputs + ([printed] if printed else []):
        mine, theirs = here / f, committed / f
        if mine.read_bytes() == theirs.read_bytes():
            print(f"   {f}: identical to the committed file")
            continue
        same = False
        if f.endswith(".json"):
            worst, first = numeric_diff(json.loads(mine.read_text()), json.loads(theirs.read_text()))
            print(f"   {f}: DIFFERS; largest numeric difference {worst:.3g}; first other difference: {first}")
        else:
            print(f"   {f}: DIFFERS (see {mine})")
    return same


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("analysis", choices=[*ANALYSES, "all"])
    ap.add_argument("--work", type=Path, help="scratch directory (default: a new temporary directory)")
    ap.add_argument("--results", type=Path, default=REPO / "results",
                    help="the results directory of finished pack runs, for feature files (default: results)")
    a = ap.parse_args()
    work = (a.work or Path(tempfile.mkdtemp(prefix="judgly-calibration-study-"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    print(f"workspace: {work}")
    root = build_root(work, a.results.resolve())
    names = list(ANALYSES) if a.analysis == "all" else [a.analysis]
    if "temperature" in names:
        build_fit(work, a.results.resolve())
    ok = all([run(n, work, root) for n in names])
    print("all outputs identical to the committed ones" if ok else "some outputs differ (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
