"""Rescore or rerun the external comparison without editing its frozen files.

The runner (run_external.py) and the scorer (score_external.py) are kept exactly as they were
frozen (PROTOCOL.sha256). Both read and write next to themselves: the runner appends to
answers/<model>/<format>-<tier>.jsonl, the scorer reads those files and the uncompressed
per-item dumps under <root>/results/<pack>/<format>/ and writes result-external.json. This
wrapper therefore builds a workspace and runs hash-checked copies there:

    <work>/root/data                     -> the checkout's data/ (the tier files, from make data)
    <work>/root/results/<pack>/<format>/  judgly's committed per-item dumps (docs/results, gunzipped)
    <work>/score-<name>/                  a copy of score_external.py and its answers/, one per
                                          scorer run (<name>: a model, or "all")

score   rescore the committed answers (answers/<model>/*.jsonl.gz) with the frozen scorer, once
        per model and once with all three models, and check that every result-*.json and
        score-*.txt equals the committed one in final/, byte for byte. Then rescore the
        equal-calibration control (calibrated/, frozen with calibrated/PROTOCOL.sha256) from its
        committed files and check that its result.json and result.txt are rebuilt byte for byte
        (--part comparison or --part control runs one of the two). CPU only.

run     ask the Ollama models again (needs Ollama >= 0.35.0 and the models pulled by the
        recorded tags), with a copy of the frozen runner in OUT (default: results-compare/),
        then score OUT's answers the same way into OUT/final/ and compare them with the record.
        Never writes into this directory. The frozen runner treats every line it has written as
        done, including a failed request; before each resume this wrapper moves lines whose
        error is not an HTTP 400 refusal (a connection error, a timeout, a server error) to
        <format>-<tier>.transient.jsonl next to the answers, so that those items are asked again.

The control's frozen scorer (calibrated/score_calibrated.py) reads the external models' training
answers uncompressed next to itself, the comparison's answers and scorer one directory up, and
judgly's raw readout of the train split from `s1-eval --dump-items` on judgly's cached feature
files (results/<pack>/<format>/fitdev/features.feat, 2.8 GB, not committed). The rows of that
readout for the 2,000 sampled items are committed (calibrated/judgly-train/, gzipped); in the
workspace a stand-in for s1-eval writes them where the scorer asks for its dump:

    <work>/control/score_external.py      a copy of the frozen comparison scorer
    <work>/control/answers               -> this directory's answers/ (the test-tier answers)
    <work>/control/calibrated/           a copy of score_calibrated.py and sample.json, and the
                                          training answers gunzipped
    <work>/root/build/cli/s1-eval         the stand-in (writes calibrated/judgly-train/<pack>-<format>-train.tsv.gz)

When judgly's own s1-eval (build/cli) and the cached feature files are present in the checkout,
the committed rows are first checked against a fresh s1-eval dump (CPU, seconds); otherwise
that check is reported as not made.

    uv run --no-project --python 3.13 --with numpy==2.5.3 --with scipy==1.18.1 python docs/results/external-comparison/reproduce.py score
    uv run --no-project --python 3.13 --with numpy==2.5.3 python docs/results/external-comparison/reproduce.py run [--out DIR]

The byte-for-byte check was made with numpy 2.5.3 and scipy 1.18.1 on Python 3.13 (the versions
above, which the Makefile pins). The scorers' bootstrap uses numpy's default_rng, whose draws numpy
does not promise to keep across versions, so another numpy may give other intervals and fail the
check; the control's temperatures come from scipy's bounded scalar minimiser.
"""

import argparse
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAL = HERE / "calibrated"
REPO = HERE.parents[2]
SNAPSHOT = REPO / "docs" / "results"
MODELS = ("nimble:9b", "tev1:4b", "tev1:0.8b")  # the order of the recorded "all" run
PACKS = ("gemma4-12b-q8", "qwen3-4b-q8")
FORMATS = ("general", "stance")
TIERS = ("confirm", "final", "bench", "final-flagged")
# The Ollama model IDs (the first 12 hex digits of the manifest's SHA-256) the record was made with.
RECORDED_IDS = {"nimble:9b": "aa4a79f08ae0", "tev1:4b": "d18e9174f4db", "tev1:0.8b": "c0099a86fcbd"}
OLLAMA_MIN = (0, 35, 0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen(name: str, where: Path = HERE) -> Path:
    """The frozen file, after checking it against PROTOCOL.sha256 (of `where`)."""
    want = {}
    for line in (where / "PROTOCOL.sha256").read_text().splitlines():
        m = re.match(r"^([0-9a-f]{64})  (\S+)$", line)
        if m:
            want[m.group(2)] = m.group(1)
    path = where / name
    if sha256(path) != want[name]:
        sys.exit(f"{path} does not match PROTOCOL.sha256; it must stay as frozen")
    return path


def check_tiers(tiers: tuple[str, ...] = TIERS) -> None:
    """The tier files the scorer reads equal the release ones (data/tiers.sha256 and
    data/tiers-confirm.sha256)."""
    want = {}
    for manifest in ("tiers.sha256", "tiers-confirm.sha256"):
        for line in (REPO / "data" / manifest).read_text().splitlines():
            if line.strip():
                h, name = line.split(maxsplit=1)
                want[name.strip()] = h
    bad = []
    for fmt in FORMATS:
        for tier in tiers:
            f = REPO / "data" / "tiers" / fmt / f"{tier}.jsonl"
            if fmt == "stance" and tier == "bench":
                continue
            key = next((k for k in want if k.endswith(f"{fmt}/{tier}.jsonl")), None)
            if not f.exists():
                bad.append(f"{f} missing (make data)")
            elif key is None or sha256(f) != want[key]:
                bad.append(f"{f} differs from the recorded SHA-256")
    if bad:
        sys.exit("the tier files are not the release ones:\n  " + "\n  ".join(bad))


def gunzip(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src, "rb") as f, open(dst, "wb") as g:
        shutil.copyfileobj(f, g)


def build_root(work: Path) -> Path:
    root = work / "root"
    (root / "results").mkdir(parents=True, exist_ok=True)
    if not (root / "data").exists():
        (root / "data").symlink_to(REPO / "data")
    for pack in PACKS:
        for fmt in FORMATS:
            for gz in sorted((SNAPSHOT / pack / fmt).glob("items-*.tsv.gz")):
                if gz.name[:-7].split("-", 2)[2] in TIERS:
                    gunzip(gz, root / "results" / pack / fmt / gz.name[:-3])
    return root


def score_all(answers: Path, work: Path, root: Path) -> dict[str, tuple[bytes, bytes]]:
    """Run the frozen scorer once per model and once with all models on the answers under
    `answers` (<model>/<format>-<tier>.jsonl or .jsonl.gz); return {name: (json, printed)}."""
    scorer = frozen("score_external.py")
    runs = {m.replace(":", "_"): [m] for m in MODELS} | {"all": list(MODELS)}

    def one(name: str) -> tuple[str, bytes, bytes]:
        d = work / f"score-{name}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        shutil.copy2(scorer, d / "score_external.py")
        for m in runs[name]:
            src = answers / m.replace(":", "_")
            for f in sorted(src.glob("*.jsonl*")):
                if ".transient." in f.name:
                    continue
                dst = d / "answers" / src.name / f.name.removesuffix(".gz")
                if f.suffix == ".gz":
                    gunzip(f, dst)
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst)
        cmd = [sys.executable, str(d / "score_external.py"), str(root), *runs[name]]
        print(f"== {name}: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, cwd=d, check=True, capture_output=True)
        return name, (d / "result-external.json").read_bytes(), proc.stdout

    with ThreadPoolExecutor(max_workers=len(runs)) as pool:
        return {n: (j, t) for n, j, t in pool.map(one, runs)}


def compare(results: dict[str, tuple[bytes, bytes]], out: Path | None) -> bool:
    same = True
    for name, (js, txt) in results.items():
        for fname, mine in ((f"result-{name}.json", js), (f"score-{name}.txt", txt)):
            if out is not None:
                (out / fname).write_bytes(mine)
            if mine == (HERE / "final" / fname).read_bytes():
                print(f"   {fname}: identical to the committed file")
            else:
                same = False
                print(f"   {fname}: DIFFERS from final/{fname}")
    return same


S1_EVAL_STANDIN = """#!{python}
# Stands in for judgly's s1-eval in the control's rescoring workspace (reproduce.py): writes the
# committed train-split readout of the sampled items to --dump-items. Nothing else is supported.
import gzip, shutil, sys
a = sys.argv
feat = a[a.index("--features") + 1].split("/")  # <root>/results/<pack>/<format>/fitdev/features.feat
assert a[a.index("--split") + 1] == "train" and "--rotations" in a and feat[-2:] == ["fitdev", "features.feat"]
with gzip.open("{src}/" + feat[-4] + "-" + feat[-3] + "-train.tsv.gz", "rb") as f, \\
        open(a[a.index("--dump-items") + 1], "wb") as g:
    shutil.copyfileobj(f, g)
"""


def check_train_readout(work: Path) -> None:
    """The committed judgly-train rows equal those of a fresh s1-eval dump of the cached features,
    when judgly's s1-eval and the feature files are present (else: reported as not checked)."""
    s1 = REPO / "build" / "cli" / "s1-eval"
    for pack in PACKS:
        for fmt in FORMATS:
            feat = REPO / "results" / pack / fmt / "fitdev" / "features.feat"
            name = f"{pack}-{fmt}-train.tsv.gz"
            if not (s1.exists() and feat.exists()):
                print(f"   judgly-train/{name}: not checked against s1-eval (no build/cli/s1-eval or no {feat.relative_to(REPO)})")
                continue
            dump = work / f"fresh-{pack}-{fmt}-train.tsv"
            subprocess.run([str(s1), "--features", str(feat), "--split", "train", "--rotations",
                            "--dump-items", str(dump)], check=True, stdout=subprocess.DEVNULL)
            with gzip.open(CAL / "judgly-train" / name, "rt") as f:
                mine = f.read().splitlines()
            keep = set(mine[1:])
            fresh = dump.read_text().splitlines()
            same = fresh[0] == mine[0] and [x for x in fresh[1:] if x in keep] == mine[1:]
            if not same:
                sys.exit(f"judgly-train/{name} differs from a fresh s1-eval dump of {feat}")
            print(f"   judgly-train/{name}: equals the rows of a fresh s1-eval dump ({len(mine) - 1} items)")


def score_control(work: Path, root: Path) -> bool:
    """Rescore the equal-calibration control (calibrated/) with its frozen scorer and compare its
    result.json and printed output with the committed ones."""
    try:
        import scipy  # noqa: F401  (the frozen scorer's minimiser)
    except ImportError:
        sys.exit("the control's scorer needs scipy (make compare-score pins scipy==1.18.1)")
    check_tiers(("fitdev",))
    check_train_readout(work)
    d = work / "control"
    if d.exists():
        shutil.rmtree(d)
    (d / "calibrated").mkdir(parents=True)
    shutil.copy2(frozen("score_external.py"), d / "score_external.py")
    (d / "answers").symlink_to(HERE / "answers")
    for name in ("score_calibrated.py", "sample.json"):
        shutil.copy2(frozen(name, CAL), d / "calibrated" / name)
    for gz in sorted((CAL / "answers").glob("*/*-train.jsonl.gz")):
        gunzip(gz, d / "calibrated" / "answers" / gz.parent.name / gz.name.removesuffix(".gz"))
    stub = root / "build" / "cli" / "s1-eval"
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_text(S1_EVAL_STANDIN.format(python=sys.executable, src=CAL / "judgly-train"))
    stub.chmod(0o755)
    cmd = [sys.executable, str(d / "calibrated" / "score_calibrated.py"), str(root)]
    print(f"== control: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=d / "calibrated", check=True, capture_output=True)
    same = True
    for fname, mine in (("result.json", (d / "calibrated" / "result.json").read_bytes()), ("result.txt", proc.stdout)):
        if mine == (CAL / fname).read_bytes():
            print(f"   calibrated/{fname}: identical to the committed file")
        else:
            same = False
            print(f"   calibrated/{fname}: DIFFERS from the committed file")
    return same


def cmd_score(a) -> int:
    check_tiers()
    work = (a.work or Path(tempfile.mkdtemp(prefix="judgly-external-"))).resolve()
    print(f"workspace: {work}")
    root = build_root(work)
    ok = True
    if a.part in ("all", "comparison"):
        ok = compare(score_all(HERE / "answers", work, root), None) and ok
    if a.part in ("all", "control"):
        ok = score_control(work, root) and ok
    print("all rebuilt results identical to the committed ones" if ok else "some results differ (see above)")
    if a.work is None:
        if ok:
            shutil.rmtree(work)  # a temporary workspace is removed; one given with --work is kept
        else:
            print(f"workspace kept for inspection: {work}")
    return 0 if ok else 1


def set_aside_transient(answers: Path) -> int:
    """Move answer lines that record a failed request other than an HTTP 400 refusal (for example
    a connection error or a timeout) to <format>-<tier>.transient.jsonl, so that the frozen runner,
    which skips every id it has written, asks those items again. Returns how many were moved."""
    moved = 0
    for f in sorted(answers.glob("*/*.jsonl")):
        if f.name.endswith(".transient.jsonl"):
            continue
        keep, aside = [], []
        for line in f.read_text().splitlines(keepends=True):
            if not line.strip():
                continue
            err = json.loads(line).get("error")
            (aside if err is not None and not err.startswith("HTTP 400") else keep).append(line)
        if aside:
            with open(f.with_suffix(".transient.jsonl"), "a") as g:
                g.writelines(aside)
            f.write_text("".join(keep))
            moved += len(aside)
            print(f"  {f}: {len(aside)} failed request(s) set aside, to be asked again")
    return moved


def ollama_ids() -> dict[str, str]:
    try:
        v = subprocess.run(["ollama", "--version"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        sys.exit(f"ollama is not available: {e}")
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", v)
    if not m or tuple(int(x) for x in m.groups()) < OLLAMA_MIN:
        sys.exit(f"needs Ollama >= {'.'.join(map(str, OLLAMA_MIN))} (/v1/systemone); found: {v.strip()}")
    print(v.strip())
    listing = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True).stdout
    ids = {}
    for line in listing.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            ids[parts[0]] = parts[1]
    return ids


def cmd_run(a) -> int:
    out = a.out.resolve()
    if out == HERE or HERE in out.parents:
        sys.exit(f"--out must be outside {HERE}: the record is never written over")
    check_tiers()
    ids = ollama_ids()
    missing = [m for m in MODELS if m not in ids]
    if missing:
        sys.exit(f"models not pulled: {', '.join(missing)} (ollama pull {' '.join(missing)})")
    print("WARNING: Ollama tags can move; a tag pulled today may name other weights than the record's.")
    for m in MODELS:
        note = "same as the record" if ids[m] == RECORDED_IDS[m] else f"DIFFERS from the record's {RECORDED_IDS[m]}"
        print(f"  {m}: ID {ids[m]} ({note})")
    out.mkdir(parents=True, exist_ok=True)
    (out / "ollama-ids.json").write_text(json.dumps({"recorded": RECORDED_IDS, "used": {m: ids[m] for m in MODELS}},
                                                    indent=1) + "\n")
    runner = out / "run_external.py"
    shutil.copy2(frozen("run_external.py"), runner)
    for m in MODELS:  # resumable: items already in OUT/answers are skipped
        set_aside_transient(out / "answers")
        subprocess.run([sys.executable, str(runner), str(REPO), m], check=True)
    left = sum(1 for f in (out / "answers").glob("*/*.jsonl") if not f.name.endswith(".transient.jsonl")
               for line in f.read_text().splitlines()
               if line.strip() and (e := json.loads(line).get("error")) and not e.startswith("HTTP 400"))
    if left:
        print(f"WARNING: {left} request(s) failed other than by an HTTP 400 refusal; "
              "run make compare-run again to ask them again before comparing")
    work = out / "work"
    root = build_root(work)
    (out / "final").mkdir(exist_ok=True)
    ok = compare(score_all(out / "answers", work, root), out / "final")
    print(f"results in {out / 'final'}; "
          + ("identical to the record" if ok else "they differ from the record (see above)"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score", help="rescore the committed answers and compare (CPU)")
    s.add_argument("--work", type=Path, help="scratch directory (default: a new temporary directory)")
    s.add_argument("--part", choices=("all", "comparison", "control"), default="all",
                   help="the comparison's eight result files, the control's two, or both (default)")
    r = sub.add_parser("run", help="ask the Ollama models again into OUT and score (needs Ollama)")
    r.add_argument("--out", type=Path, default=REPO / "results-compare", help="default: results-compare")
    a = ap.parse_args()
    return cmd_score(a) if a.command == "score" else cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
