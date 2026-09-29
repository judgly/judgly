"""Assemble a pack directory: the source pack's pack.json and template, both calibration options
of every format (the H2 head and the per-type temperature) and their calibration records. Every
head file must pass scripts/check_heads.py under the pack's engine settings, or no pack is
written.

    uv run --group pipeline python scripts/build_pack.py PACK OUT_DIR RESULTS_DIR FORMAT [FORMAT ...]

For each FORMAT, RESULTS_DIR/<format>/h2.bin and temperature.bin become OUT_DIR/heads/<file>
(h2.bin and temperature.bin for the general format, which answers every question without a format
of its own, "*"; h2-<format>.bin and temperature-<format>.bin otherwise), each with the trainer's
provenance sidecar beside it as <file>.json, and RESULTS_DIR/<format>/record.json becomes
OUT_DIR/calibration/<format>.json (one record holds both options). pack.json (schema 2) names,
per format, both options with their SHA-256, licence and record, and the option the pack uses by
default (DEFAULTS). The directory is written under OUT_DIR.tmp and renamed, so an OUT_DIR that
exists is complete.
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_heads  # noqa: E402
import registry  # noqa: E402
from judgly.packs import Pack  # noqa: E402

OPTIONS = ("h2", "temperature")

# The option a pack uses when the caller does not choose one. The pre-registered confirmation on
# untouched data (docs/calibration.md) compared the temperature with H2 per pack and format: the
# temperature met all three criteria in three cases and is the default there; for gemma4-12b-q8
# general it did not, and H2 stays the default. A pack or format not listed keeps H2.
DEFAULTS = {("qwen3-4b-q8", "general"): "temperature", ("qwen3-4b-q8", "stance"): "temperature",
            ("gemma4-12b-q8", "stance"): "temperature", ("gemma4-12b-q8", "general"): "h2"}
BASIS = "pre-registered confirmation on untouched data, temperature against H2 (docs/calibration.md): {}"


def summary(record: dict, tier: str, cond: str) -> dict | None:
    t = record["tiers"].get(tier)
    if t is None:
        return None
    m = t["conditions"][cond]["metrics"]
    return {k: m[k]["value"] for k in ("accuracy", "log_loss", "ece")}


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
        key = "*" if fmt == "general" else fmt
        default = DEFAULTS.get((pack, fmt), "h2")
        verdict = ("confirmed, so the temperature is the default" if default == "temperature" else
                   "not confirmed, so H2 stays the default" if (pack, fmt) in DEFAULTS else
                   "not tested, so H2 is the default")
        entry = {"default": default, "default_basis": BASIS.format(verdict), "options": {}}
        for option in OPTIONS:
            name = f"{option}.bin" if fmt == "general" else f"{option}-{fmt}.bin"
            head = results / fmt / f"{option}.bin"
            bad = check_heads.problems(head, spec["engine"])
            if bad:
                sys.exit(f"build_pack: {head}: " + "; ".join(bad))
            sha = (record["head"] if option == "h2" else record["temperature"])["sha256"]
            if sha != registry.sha256(head):
                sys.exit(f"build_pack: {head} is not the file that {results / fmt / 'record.json'} describes")
            shutil.copy(head, tmp / "heads" / name)
            shutil.copy(results / fmt / f"{option}.bin.json", tmp / "heads" / f"{name}.json")
            entry["options"][option] = {"file": f"heads/{name}", "status": "available", "sha256": sha,
                                        "licence": reg["policy"]["head_licence"][fmt],
                                        "calibration": f"calibration/{fmt}.json"}
            if not record["quick"]:          # a QUICK run never reads the final or confirm tiers
                for tier in ("final", "confirm"):
                    if (s := summary(record, tier, option)) is not None:
                        entry["options"][option][f"{tier}_tier"] = s
        shutil.copy(results / fmt / "record.json", tmp / "calibration" / f"{fmt}.json")
        heads[key] = entry
    spec["schema"] = 2
    spec["heads"] = heads
    if quick:
        spec["description"] = "QUICK SMOKE-TEST PACK: heads fitted on a few hundred items. Not for use. " + spec["description"]
        spec["quick"] = True
    (tmp / "pack.json").write_text(json.dumps(spec, indent=1) + "\n")
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    print(f"build_pack: {out}: " + ", ".join(f"{k} ({v['default']} by default)" for k, v in heads.items()))


if __name__ == "__main__":
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:])
