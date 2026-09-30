"""The data registry (data/registry.yaml) and its licence policy, shared by the pipeline scripts.

    from registry import load, tier_of, fit_allowed
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "data" / "registry.yaml"
RAW = ROOT / "data" / "raw"


def load(path: Path = REGISTRY) -> dict:
    reg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if reg.get("schema") != 1:
        raise SystemExit(f"{path}: unsupported registry schema")
    for name, src in reg["sources"].items():
        fams = reg["formats"][src["format"]]["families"]
        if src["family"] not in fams:
            raise SystemExit(f"{path}: {name}: family {src['family']!r} has no tier in format {src['format']!r}")
    for fmt, spec in reg["formats"].items():
        for fam, tier in spec["families"].items():
            if tier not in TIERS:
                raise SystemExit(f"{path}: {fmt} family {fam}: unknown tier {tier!r}")
    for name, src in reg["sources"].items():
        if (tier_of(reg, name) == "final-flagged") != bool(src.get("caveat")):
            raise SystemExit(f"{path}: {name}: a final-flagged source, and only one, must record its `caveat`")
    return reg


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def registry_sha256() -> str:
    return sha256(REGISTRY)


# fit: fitting data; dev: held-out families scored during development; final: the fresh held-out
# families, read once for the reported numbers; final-flagged: fresh held-out families with a
# recorded caveat (a source's `caveat`), read with final but reported beside it, never pooled
# into it; final-seen: an earlier held-out tier, its families read during development
# (reported separately as previously seen); bench: external benchmarks, evaluation only;
# confirm: untouched families for a pre-registered confirmation, evaluation only, built after
# every other tier and cleaned against all of them; reserved: in no tier, kept unseen; none: not
# used.
TIERS = ("fit", "dev", "final", "final-flagged", "final-seen", "bench", "confirm", "reserved", "none")
EVAL_TIERS = ("dev", "final", "final-flagged", "final-seen", "bench", "confirm")


def tier_of(reg: dict, name: str) -> str:
    src = reg["sources"][name]
    return reg["formats"][src["format"]]["families"][src["family"]]


def used(reg: dict, name: str) -> bool:
    return reg["sources"][name].get("use", True) and tier_of(reg, name) != "none"


def fit_allowed(reg: dict, name: str) -> tuple[bool, str]:
    """Whether the policy lets this source be fitted on, and why (or why not).

    Without a recorded decision, every licence token on the card must be one the policy allows
    for this source's format. With one (`licence_decision: {treat_as, reason}`), the source is
    treated as `treat_as` alone, whatever the card says: the decision must quote its evidence,
    either the card's own licensing text (`licence_text`) or a primary source (`source`, a URL,
    and `quote`, text found there), and `treat_as` must itself be a licence the policy allows
    for this format, so that a decision can never route share-alike or unknown data into the
    general head. A decision can also make a source stricter than its card (a card that says
    Apache-2.0 for share-alike data)."""
    src = reg["sources"][name]
    allowed = set(reg["policy"]["permissive"])
    if src["format"] in reg["policy"]["share_alike_formats"]:
        allowed |= set(reg["policy"]["share_alike"])
    tokens = src.get("licence") or []
    if not tokens:
        return False, "no licence recorded"
    decision = src.get("licence_decision")
    if decision is None:
        unknown = [t for t in tokens if t not in allowed]
        if not unknown:
            return True, "allowed"
        return False, f"licence {', '.join(unknown)} is not allowed for fitting the {src['format']} head"
    quoted = src.get("licence_text") or (isinstance(decision, dict) and decision.get("source") and decision.get("quote"))
    if not (isinstance(decision, dict) and decision.get("reason") and quoted):
        return False, "the recorded licence decision names no reason or quotes no evidence"
    if decision.get("treat_as") not in allowed:
        return False, (f"recorded decision treats {', '.join(tokens)} as {decision.get('treat_as')!r}, "
                       f"which is not allowed for fitting the {src['format']} head")
    return True, f"allowed by recorded decision: treated as {decision['treat_as']} ({decision['reason']})"


def check_policy(reg: dict) -> list[str]:
    """Every problem with the tier assignment: a fit source the policy does not allow, or a
    fit-family source with a disallowed licence that is not marked `use: false`."""
    problems = []
    for name, src in reg["sources"].items():
        if tier_of(reg, name) != "fit":
            continue
        ok, why = fit_allowed(reg, name)
        if src.get("use", True) and not ok:
            problems.append(f"{name}: in the fit tier but {why}")
        if not src.get("use", True) and ok:
            problems.append(f"{name}: marked use: false but its licence is allowed; say why in a note or use it")
    return problems
