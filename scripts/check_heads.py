"""Check a fitted head file against its sidecar before it goes into a pack.

    uv run --group pipeline python scripts/check_heads.py HEAD.bin [--engine ROTATIONS MAX_ROTATIONS CONTENT_FREE]

The sidecar HEAD.bin.json (written by s1-train) must record the engine settings the head was
fitted under (and equal --engine, given as 0/1, N, 0/1, when given) and, per question type,
either a fitted head or an explicit fallback to the identity (H0) with its reason. A fitted type
must depend on its input (the largest standard deviation of any option's probability across the
validation records at least MIN_PROB_SD), have a validation loss below the raw readout's, and a
temperature within [TEMP_MIN, TEMP_MAX] that matches the head file. A type that falls back must
be all zeros in the head file (H1, H2) or have temperature 1 (a temperature head). A temperature
head's file must hold exactly the sidecar's (three-decimal) temperature. Exits 1 and names every
failure otherwise.
"""

import json
import math
import struct
import sys
from pathlib import Path

TYPES = ("choice", "bool", "score")
K_MAX = 26
TEMP_MIN, TEMP_MAX, MIN_PROB_SD = 0.05, 100.0, 1e-3  # csrc/s1.h: S1_TEMP_MIN, S1_TEMP_MAX, S1_MIN_PROB_SD
MAGIC = b"S1HEAD\0\0"


def head_kind(path: Path) -> int:
    """1 (H1), 2 (H2) or 3 (per-type temperature)."""
    return struct.unpack_from("<I", path.read_bytes(), 12)[0]


def head_params(path: Path) -> dict[str, list[float]]:
    """The parameter array per question type of a head file (csrc/s1_headfile.c); for a
    temperature head, [T] per type."""
    raw = path.read_bytes()
    if raw[:8] != MAGIC:
        raise ValueError(f"{path} is not a head file")
    _, kind, n_embd = struct.unpack_from("<III", raw, 8)
    n = 1 if kind == 3 else 2 + K_MAX + (K_MAX * n_embd if kind == 2 else 0)
    at = 20 + 65 + 65 + 4 * K_MAX
    if len(raw) != at + 3 * n * 8:
        raise ValueError(f"{path} has {len(raw)} bytes, expected {at + 3 * n * 8}")
    return {t: list(struct.unpack_from(f"<{n}d", raw, at + i * n * 8)) for i, t in enumerate(TYPES)}


def problems(head: Path, engine: dict | None = None) -> list[str]:
    sidecar = Path(f"{head}.json")
    if not sidecar.is_file():
        return [f"{sidecar} is missing"]
    side = json.loads(sidecar.read_text())
    found = []
    fitted_engine = side.get("engine")
    if not isinstance(fitted_engine, dict) or set(fitted_engine) != {"rotations", "max_rotations", "content_free"}:
        found.append("the sidecar does not record the engine settings (rotations, max_rotations, content_free)")
    elif engine is not None and fitted_engine != engine:
        found.append(f"engine settings {fitted_engine}, expected {engine}")
    params = head_params(head)
    temperature_head = head_kind(head) == 3
    for t in TYPES:
        entry = side.get("types", {}).get(t)
        if entry is None:
            found.append(f"{t}: no entry in the sidecar")
            continue
        if entry.get("fallback") is not None:
            if entry["fallback"] != "identity" or not entry.get("reason"):
                found.append(f"{t}: fallback {entry['fallback']!r} without a reason, or not the identity")
            if temperature_head and params[t] != [1.0]:
                found.append(f"{t}: falls back to the identity, but its temperature is not 1")
            if not temperature_head and any(v != 0.0 for v in params[t]):
                found.append(f"{t}: falls back to the identity, but its parameters are not all zero")
            continue
        sd, loss, h0, temp = (entry.get(k) for k in ("prob_sd", "validation_loss", "h0_validation_loss", "temperature"))
        if None in (sd, loss, h0, temp):
            found.append(f"{t}: the sidecar lacks prob_sd, validation_loss, h0_validation_loss or temperature")
            continue
        if sd < MIN_PROB_SD:
            found.append(f"{t}: input-independent (probability sd {sd:.3g} < {MIN_PROB_SD}) and no fallback recorded")
        if not loss < h0:
            found.append(f"{t}: validation loss {loss:.6f} does not beat the raw readout's {h0:.6f}")
        if not TEMP_MIN * (1 - 1e-9) <= temp <= TEMP_MAX * (1 + 1e-9):
            found.append(f"{t}: temperature {temp:.6g} outside [{TEMP_MIN}, {TEMP_MAX}]")
        if temperature_head and params[t][0] != temp:
            found.append(f"{t}: the head file's temperature {params[t][0]!r} is not the sidecar's {temp!r}")
        if not temperature_head and not math.isclose(math.exp(-params[t][0]), temp, rel_tol=1e-4):
            found.append(f"{t}: the head file's temperature {math.exp(-params[t][0]):.6g} is not the sidecar's {temp:.6g}")
    return found


def main(argv: list[str]) -> int:
    if len(argv) not in (1, 5) or (len(argv) == 5 and argv[1] != "--engine"):
        sys.exit(__doc__)
    engine = None
    if len(argv) == 5:
        engine = {"rotations": argv[2] == "1", "max_rotations": int(argv[3]), "content_free": argv[4] == "1"}
    bad = problems(Path(argv[0]), engine)
    for b in bad:
        print(f"check_heads: {argv[0]}: {b}")
    if not bad:
        side = json.loads(Path(f"{argv[0]}.json").read_text())["types"]
        print(f"check_heads: {argv[0]}: ok (" + ", ".join(
            f"{t} {'identity: ' + side[t]['reason'] if side[t].get('fallback') else 'fitted'}" for t in TYPES) + ")")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
