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
        score-*.txt equals the committed one in final/, byte for byte. CPU only.

run     ask the Ollama models again (needs Ollama >= 0.35.0 and the models pulled by the
        recorded tags), with a copy of the frozen runner in OUT (default: results-compare/),
        then score OUT's answers the same way into OUT/final/ and compare them with the record.
        Never writes into this directory.

    uv run --no-project --with numpy python docs/results/external-comparison/reproduce.py score
    uv run --no-project --with numpy python docs/results/external-comparison/reproduce.py run [--out DIR]
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


def frozen(name: str) -> Path:
    """The frozen file, after checking it against PROTOCOL.sha256."""
    want = {}
    for line in (HERE / "PROTOCOL.sha256").read_text().splitlines():
        m = re.match(r"^([0-9a-f]{64})  (\S+)$", line)
        if m:
            want[m.group(2)] = m.group(1)
    path = HERE / name
    if sha256(path) != want[name]:
        sys.exit(f"{path} does not match PROTOCOL.sha256; it must stay as frozen")
    return path


def check_tiers() -> None:
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
        for tier in TIERS:
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


def cmd_score(a) -> int:
    check_tiers()
    work = (a.work or Path(tempfile.mkdtemp(prefix="judgly-external-"))).resolve()
    print(f"workspace: {work}")
    root = build_root(work)
    ok = compare(score_all(HERE / "answers", work, root), None)
    print("all rebuilt results identical to the committed ones" if ok else "some results differ (see above)")
    return 0 if ok else 1


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
        subprocess.run([sys.executable, str(runner), str(REPO), m], check=True)
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
    r = sub.add_parser("run", help="ask the Ollama models again into OUT and score (needs Ollama)")
    r.add_argument("--out", type=Path, default=REPO / "results-compare", help="default: results-compare")
    a = ap.parse_args()
    return cmd_score(a) if a.command == "score" else cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
