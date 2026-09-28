"""Assemble a pack directory: the source pack's pack.json and template, the fitted heads and
their calibration records. Every head must pass scripts/check_heads.py under the pack's engine
settings, or no pack is written.

    uv run --group pipeline python scripts/build_pack.py PACK OUT_DIR RESULTS_DIR FORMAT [FORMAT ...]

For each FORMAT, RESULTS_DIR/<format>/h2.bin becomes OUT_DIR/heads/<file> (h2.bin for the
general format, which answers every question without a format of its own, "*"; h2-<format>.bin
otherwise), with the trainer's provenance sidecar h2.bin.json beside it as <file>.json, and
RESULTS_DIR/<format>/record.json becomes OUT_DIR/calibration/<format>.json. The pack.json entry of each head gains its SHA-256, licence and record. The directory is written
under OUT_DIR.tmp and renamed, so an OUT_DIR that exists is complete.
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_heads  # noqa: E402
import registry  # noqa: E402
from judgly.packs import Pack  # noqa: E402


def main(pack: str, out: Path, results: Path, formats: list[str]) -> None:
    src = Pack.find(pack)
    reg = registry.load()
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "heads").mkdir(parents=True)
    (tmp / "calibration").mkdir()
    spec = json.loads((src.directory / "pack.json").read_text())
    shutil.copy(src.template, tmp / spec["template"])
    heads = {}
    quick = False
    for fmt in formats:
        record = json.loads((results / fmt / "record.json").read_text())
        quick |= record["quick"]
        key, name = ("*", "h2.bin") if fmt == "general" else (fmt, f"h2-{fmt}.bin")
        bad = check_heads.problems(results / fmt / "h2.bin", spec["engine"])
        if bad:
            sys.exit(f"build_pack: {results / fmt / 'h2.bin'}: " + "; ".join(bad))
        shutil.copy(results / fmt / "h2.bin", tmp / "heads" / name)
        if (results / fmt / "h2.bin.json").is_file():   # the trainer's sidecar: features SHA-256, lambdas, losses
            shutil.copy(results / fmt / "h2.bin.json", tmp / "heads" / f"{name}.json")
        shutil.copy(results / fmt / "record.json", tmp / "calibration" / f"{fmt}.json")
        heads[key] = {"file": f"heads/{name}", "status": "available", "sha256": record["head"]["sha256"],
                      "licence": reg["policy"]["head_licence"][fmt], "calibration": f"calibration/{fmt}.json"}
        if not record["quick"]:          # a QUICK run never reads the final tier
            final = record["tiers"]["final"]["conditions"]["h2"]["metrics"]
            heads[key]["final_tier"] = {m: final[m]["value"] for m in ("accuracy", "log_loss", "ece")}
    spec["heads"] = heads
    if quick:
        spec["description"] = "QUICK SMOKE-TEST PACK: heads fitted on a few hundred items. Not for use. " + spec["description"]
        spec["quick"] = True
    (tmp / "pack.json").write_text(json.dumps(spec, indent=1) + "\n")
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    print(f"build_pack: {out} with heads {', '.join(heads)}")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:])
