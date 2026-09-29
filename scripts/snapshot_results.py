"""Copy what the published numbers depend on from RESULTS into docs/results, with checksums.

    uv run python scripts/snapshot_results.py [RESULTS] [PACK ...]

RESULTS defaults to results/ and the packs to gemma4-12b-q8 and qwen3-4b-q8. For each pack and
format (general, stance) it copies record.json, tables.md, train-h2.log, train-temperature.log,
the per-item dumps items-{raw,h2,temperature}-{test,dev,final,final-flagged,confirm,final-seen,bench}.tsv
and the names sidecars of the fitdev, final, final-flagged, confirm, final-seen and bench feature
files (family ids to names; a tier the format does not have is skipped), the last two gzipped
without a timestamp so the same inputs give the same bytes; per pack it copies selftest.txt. It
then writes

  docs/results/MANIFEST        SHA-256 of every snapshot file   (cd docs/results && shasum -a 256 -c MANIFEST)
  docs/results/INPUTS.sha256   SHA-256 of the large inputs that are not committed: the merged
                               features.feat per format and tier, the h2 and temperature
                               heads and the tier files                            (shasum -a 256 -c docs/results/INPUTS.sha256)

scripts/make_figures.py reads only docs/results.
"""

import gzip
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "results"
FORMATS = ("general", "stance")
PLAIN = ("record.json", "tables.md", "train-h2.log", "train-temperature.log")
TIERS = ("test", "dev", "final", "final-flagged", "confirm", "final-seen", "bench")
PARTS = ("fitdev", "final", "final-flagged", "confirm", "final-seen", "bench")
DUMPS = [f"items-{c}-{t}.tsv" for c in ("raw", "h2", "temperature") for t in TIERS]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def gz(src: Path, dst: Path) -> None:
    with open(src, "rb") as f, open(dst, "wb") as raw, \
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as g:
        shutil.copyfileobj(f, g)


def main(results: Path, packs: list[str]) -> None:
    inputs = []
    for pack in packs:
        src_pack, dst_pack = results / pack, OUT / pack
        dst_pack.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_pack / "selftest.txt", dst_pack / "selftest.txt")
        for fmt in FORMATS:
            src, dst = src_pack / fmt, dst_pack / fmt
            dst.mkdir(parents=True, exist_ok=True)
            for name in PLAIN:
                shutil.copyfile(src / name, dst / name)
            for name in DUMPS:
                if (src / name).is_file():
                    gz(src / name, dst / f"{name}.gz")
            for part in PARTS:
                if not (src / part / "features.feat").is_file():
                    continue
                gz(src / part / "features.feat.names.tsv", dst / f"names-{part}.tsv.gz")
                inputs.append(src / part / "features.feat")
            inputs += [src / "h2.bin", src / "temperature.bin"]
    for fmt in FORMATS:
        for part in PARTS:
            if (ROOT / "data" / "tiers" / fmt / f"{part}.jsonl").is_file():
                inputs.append(ROOT / "data" / "tiers" / fmt / f"{part}.jsonl")
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in ("MANIFEST", "INPUTS.sha256"))
    (OUT / "MANIFEST").write_text("".join(f"{sha256(p)}  {p.relative_to(OUT)}\n" for p in files))
    (OUT / "INPUTS.sha256").write_text(
        "".join(f"{sha256(p)}  {p.resolve().relative_to(ROOT)}\n" for p in inputs))
    size = sum(p.stat().st_size for p in files)
    print(f"snapshot: {len(files)} files, {size / 1e6:.1f} MB in {OUT.relative_to(ROOT)}; "
          f"{len(inputs)} large inputs hashed")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(Path(args[0]) if args else ROOT / "results", args[1:] or ["gemma4-12b-q8", "qwen3-4b-q8"])
