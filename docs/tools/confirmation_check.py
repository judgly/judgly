"""Check that the shipped temperature option is the one the pre-registered confirmation tested.

    uv run --no-project --with numpy python docs/tools/confirmation_check.py

1. The frozen files of the study (docs/results/calibration-study) still have the SHA-256 recorded
   when they were frozen (confirmation/CONFIRM.sha256, exploratory-2-post-processing/PROTOCOL.sha256).
2. Every shipped temperature head (src/judgly/packs/*/heads/temperature*.bin) holds exactly the
   temperatures of confirmation/temperatures.json (1 for a question type the format does not
   have).
3. The engine's temperature output reproduces the confirmation's: for the confirm tier and, as a
   second check, the dev tier of every pack and format, the per-item probabilities that s1-eval
   wrote with the shipped temperature head (docs/results/<pack>/<format>/items-temperature-*.tsv.gz)
   equal those that the frozen scorer's ttype() (confirmation/score_confirm.py) computes from the
   raw per-item dumps (items-raw-*.tsv.gz), within 1e-9.

Prints the largest difference per pack, format and tier; exits 1 if any check fails.
"""

import gzip
import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "docs" / "results" / "calibration-study"
PACKS = ("gemma4-12b-q8", "qwen3-4b-q8")
PARAMS_AT = 8 + 4 + 4 + 4 + 65 + 65 + 4 * 26   # csrc/s1_headfile.c
TOL = 1e-9
failures: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)
    print("FAIL", msg)


def frozen_files() -> None:
    for sums in (STUDY / "confirmation" / "CONFIRM.sha256",
                 STUDY / "exploratory-2-post-processing" / "PROTOCOL.sha256"):
        for line in sums.read_text().splitlines():
            parts = line.split()
            if len(parts) != 2 or len(parts[0]) != 64:
                continue                                  # the freeze-time notes
            got = hashlib.sha256((sums.parent / parts[1]).read_bytes()).hexdigest()
            if got != parts[0]:
                fail(f"{sums.parent.name}/{parts[1]} differs from its frozen SHA-256")
            else:
                print(f"ok   {sums.parent.name}/{parts[1]} has its frozen SHA-256")


def scorer():
    """The frozen scorer as a module (it reads four command-line arguments when imported)."""
    argv, sys.argv = sys.argv, ["score_confirm.py", str(ROOT), "-", "-", "-"]
    try:
        spec = importlib.util.spec_from_file_location("score_confirm", STUDY / "confirmation" / "score_confirm.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.argv = argv
    return module


def main() -> int:
    frozen_files()
    temps = json.loads((STUDY / "confirmation" / "temperatures.json").read_text())
    sc = scorer()
    with tempfile.TemporaryDirectory() as tmp:
        for pack in PACKS:
            for fmt, suffix in (("general", ""), ("stance", "-stance")):
                head = ROOT / "src" / "judgly" / "packs" / pack / "heads" / f"temperature{suffix}.bin"
                shipped = struct.unpack_from("<3d", head.read_bytes(), PARAMS_AT)
                wanted = tuple(temps[pack][fmt].get(str(t), 1.0) for t in range(3))
                if shipped != wanted:
                    fail(f"{head.relative_to(ROOT)} holds {shipped}, not the confirmed {wanted}")
                else:
                    print(f"ok   {head.relative_to(ROOT)} holds the confirmed temperatures {wanted}")
                for tier in ("confirm", "dev"):
                    dumps = {}
                    for cond in ("raw", "temperature"):
                        src = ROOT / "docs" / "results" / pack / fmt / f"items-{cond}-{tier}.tsv.gz"
                        dst = Path(tmp) / f"{pack}-{fmt}-{cond}-{tier}.tsv"
                        dst.write_bytes(gzip.decompress(src.read_bytes()))
                        dumps[cond] = sc.read_dump(dst)
                    ids, typ, y, raw = dumps["raw"]
                    ids2, typ2, y2, ours = dumps["temperature"]
                    if ids != ids2 or not (typ == typ2).all() or not (y == y2).all():
                        fail(f"{pack}/{fmt}/{tier}: the raw and temperature dumps hold different items")
                        continue
                    diff = float(np.nanmax(np.abs(sc.ttype(raw, typ, temps[pack][fmt]) - ours)))
                    line = f"{pack}/{fmt}/{tier}: {len(ids)} items, max |engine - frozen ttype()| = {diff:.1e}"
                    if diff < TOL:
                        print(f"ok   {line}")
                    else:
                        fail(line)
    print("PASS confirmation check" if not failures else f"FAIL confirmation check: {len(failures)} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
