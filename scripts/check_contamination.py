"""Contamination check for the tiers written by scripts/prep_tiers.py.

    uv run --group pipeline python scripts/check_contamination.py DATA_DIR

Recomputes everything from the tier files themselves (and, for reserved sources, from their
pinned releases) with the text matching of scripts/texthash.py. Per format it fails (exit 1)
when:

  - an item of a reserved source (its claims and passages) appears in any tier
  - a final-seen item appears in the fit or dev tier, or a fit item in the dev tier
  - an item of the fresh final or final-flagged tier matches the fit, dev, final-seen or
    reserved texts (a final-flagged item also the final texts): equal after normalisation,
    sharing a sentence of 60 or more normalised characters, sharing a passage (5 or more
    distinct word 8-grams that are not boilerplate) or a near duplicate (word Jaccard of at
    least 0.6)
  - a fit item shares a sentence or a passage with, or is a near duplicate of, a dev,
    final-seen or reserved text
  - a bench item matches the fit tier in any of these ways
  - an item of the confirm tier matches any other tier (fit, dev, final, final-flagged,
    final-seen, bench) or the texts it is built against (prep_tiers.confirm_texts: the
    reserved sources and, for stance, every evaluation source as a whole) in any of these ways
  - a family appears in more than one tier
  - a fit-tier item comes from a source the licence policy does not allow for fitting, or
    any item from a source the registry puts in another tier
  - two items in the tiers have the same id

Two items match exactly when any identifying text is equal after normalisation: the claim or
the evidence passage of a stance item, the state or the item's own question of a general item
(texts under texthash.MIN_CHARS characters are not compared). Reported, not failed: overlap
between the train, validation and test splits of the fit tier (they share a distribution by
design, and items sharing a passage are already kept in one split); bench items matching dev,
final, final-flagged or final-seen (all evaluation only); the fresh final and final-flagged
tiers of each format against the fit tier of the other format, which its head never sees; and
dev and final-seen items sharing a sentence or a passage with, or nearly duplicating, a
reserved text (both were read before the reserved sources were chosen, so a reserved source
must drop what overlaps them before it is promoted to a final tier).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import registry  # noqa: E402
from prep_tiers import OUT_FILE, STRICT, confirm_texts, reserved_texts  # noqa: E402  the same reading of the sources
from texthash import (NearIndex, ShingleIndex, digest, digests_of, segment_digests, segments_of,  # noqa: E402
                      texts_of)

SPLITS = ("train", "validation", "test")


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def index(examples: list[dict], key=digests_of) -> dict[str, str]:
    out: dict[str, str] = {}
    for ex in examples:
        for d in key(ex):
            out.setdefault(d, ex["id"])
    return out


def overlap(a: dict[str, str], b: dict[str, str]) -> list[tuple[str, str]]:
    return [(a[k], b[k]) for k in sorted(a.keys() & b.keys())]


def near_index(examples: list[dict]) -> NearIndex:
    idx = NearIndex()
    for ex in examples:
        for name, text in texts_of(ex).items():
            if text:
                idx.add(f"{ex['id']}#{name}", text)
    return idx


def near_pairs(examples: list[dict], idx: NearIndex) -> list[tuple[str, str]]:
    out = []
    for ex in examples:
        for name, text in texts_of(ex).items():
            for key, j in idx.near(text) if text else ():
                out.append((f"{ex['id']}#{name}", f"{key} (Jaccard {j:.2f})"))
    return out


def shingle_index(examples: list[dict]) -> ShingleIndex:
    idx = ShingleIndex()
    for ex in examples:
        for name, text in texts_of(ex).items():
            if text:
                idx.add(f"{ex['id']}#{name}", text)
    return idx


def shingle_pairs(examples: list[dict], idx: ShingleIndex) -> list[tuple[str, str]]:
    out = []
    for ex in examples:
        for name, text in texts_of(ex).items():
            for key, n in idx.shared(text) if text else ():
                out.append((f"{ex['id']}#{name}", f"{key} ({n} shared 8-grams)"))
    return out


FINALS = ("final", "final-flagged")


def tiers_of(data: Path, fmt: str) -> dict[str, list[dict]]:
    fitdev = read(data / fmt / f"{OUT_FILE['fit']}.jsonl")
    return {"fit": [e for e in fitdev if e["split"] in SPLITS],
            "dev": [e for e in fitdev if e["split"] == "heldout"],
            **{t: read(data / fmt / f"{OUT_FILE[t]}.jsonl") for t in ("final", "final-flagged", "final-seen", "bench",
                                                                      "confirm")}}


def check_format(reg: dict, data: Path, fmt: str, other_fit: list[dict] | None = None) -> int:
    tiers = tiers_of(data, fmt)
    idx = {t: index(v) for t, v in tiers.items()}
    seg = {t: index(v, segments_of) for t, v in tiers.items()}
    texts = reserved_texts(reg, fmt)
    reserved = {d: "reserved" for d in (digest(t) for t in texts) if d}
    reserved_seg = {d: "reserved" for t in texts for d in segment_digests(t)}
    near = {t: near_index(tiers[t]) for t in ("fit", "dev", "final-seen", "final", "final-flagged", "bench")}
    shing = {t: shingle_index(tiers[t]) for t in tiers}
    near["reserved"], shing["reserved"] = NearIndex(), ShingleIndex()
    for i, t in enumerate(texts):
        near["reserved"].add(f"reserved-{i}", t)
        shing["reserved"].add(f"reserved-{i}", t)
    seg["reserved"] = reserved_seg

    def joined(names) -> ShingleIndex:
        """One index over several tiers, as the tier builder searches them (boilerplate is
        counted across all of them)."""
        out = ShingleIndex()
        for n in names:
            out.update(shing[n])
        return out

    def against(name: str, a: str, b: str) -> dict[str, list]:
        """a (a tier) against b (a tier, or reserved): shared sentence, near duplicate."""
        return {f"{name} shares a sentence with {b}": overlap(seg[a], seg[b]),
                f"{name} near duplicate of {b}": near_pairs(tiers[a], near[b])}

    failures = {f"reserved in {t}": overlap(reserved, idx[t]) for t in tiers if t != "confirm"}
    failures |= {"final-seen in fit": overlap(idx["final-seen"], idx["fit"]),
                 "final-seen in dev": overlap(idx["final-seen"], idx["dev"]),
                 "fit in dev": overlap(idx["fit"], idx["dev"])}
    for f in FINALS:
        for t in ("fit", "dev", "final-seen", "reserved") + (("final",) if f == "final-flagged" else ()):
            if t != "reserved":
                failures[f"{f} in {t}"] = overlap(idx[f], idx[t])
            failures |= against(f, f, t)
        failures[f"{f} shares a passage with an earlier tier ({', '.join(STRICT[f])})"] = \
            shingle_pairs(tiers[f], joined(STRICT[f]))
    for t in ("dev", "final-seen", "reserved"):
        failures |= against("fit", "fit", t)
    failures[f"fit shares a passage with an evaluation tier ({', '.join(STRICT['fit'])})"] = \
        shingle_pairs(tiers["fit"], joined(STRICT["fit"]))
    failures["bench in fit"] = overlap(idx["bench"], idx["fit"])
    failures |= against("bench", "bench", "fit")
    failures["bench shares a passage with fit"] = shingle_pairs(tiers["bench"], shing["fit"])
    if tiers["confirm"]:
        # The confirm tier against every other tier and against its source texts (not the
        # was_reserved sources, one of which it is built from).
        extra = confirm_texts(reg, fmt)
        idx["sources"] = {d: "sources" for d in (digest(t) for t in extra) if d}
        seg["sources"] = {d: "sources" for t in extra for d in segment_digests(t)}
        near["sources"], shing["sources"] = NearIndex(), ShingleIndex()
        for i, t in enumerate(extra):
            near["sources"].add(f"sources-{i}", t)
            shing["sources"].add(f"sources-{i}", t)
        for t in ("fit", "dev", "final", "final-flagged", "final-seen", "bench", "sources"):
            failures[f"confirm in {t}"] = overlap(idx["confirm"], idx[t])
            failures |= against("confirm", "confirm", t)
        failures[f"confirm shares a passage with another tier ({', '.join(STRICT['confirm'])})"] = \
            shingle_pairs(tiers["confirm"], joined(STRICT["confirm"]))

    families: dict[str, set[str]] = {}
    for tier, exs in tiers.items():
        for e in exs:
            families.setdefault(e["family"], set()).add(tier)
    failures["family in two tiers"] = [(f, ",".join(sorted(t))) for f, t in families.items() if len(t) > 1]
    bad_source = []
    for tier, exs in tiers.items():
        for src in sorted({e["source"] for e in exs}):
            if src not in reg["sources"] or registry.tier_of(reg, src) != tier or not registry.used(reg, src):
                said = registry.tier_of(reg, src) if src in reg["sources"] else "nothing"
                bad_source.append((src, f"in {tier}, registry says {said}"))
            elif tier == "fit" and not registry.fit_allowed(reg, src)[0]:
                bad_source.append((src, registry.fit_allowed(reg, src)[1]))
    failures["source not allowed in its tier"] = bad_source
    ids = [e["id"] for exs in tiers.values() for e in exs]
    failures["duplicate ids"] = ([(i, "") for i in sorted({i for i in ids if ids.count(i) > 1})]
                                 if len(set(ids)) != len(ids) else [])

    for name, pairs in failures.items():
        print(f"{fmt}: {name}: {len(pairs)}" + (f"  e.g. {pairs[:3]}" if pairs else ""))
    splits = {s: index([e for e in tiers["fit"] if e["split"] == s]) for s in SPLITS}
    print(f"{fmt}: fit splits sharing a text (reported, allowed): train/test {len(overlap(splits['train'], splits['test']))}, "
          f"train/validation {len(overlap(splits['train'], splits['validation']))}")
    reported = [f"bench equal to {t} {len(overlap(idx['bench'], idx[t]))}" for t in ("dev", "final", "final-flagged", "final-seen")]
    if tiers["confirm"]:
        reported.append(f"confirm texts in the 0.1.0 reserved set (was_reserved) {len(overlap(reserved, idx['confirm']))}")
    if other_fit is not None:
        other_idx, other_seg = index(other_fit), index(other_fit, segments_of)
        reported += [f"{f} equal to the other format's fit {len(overlap(idx[f], other_idx))}, "
                     f"sharing a sentence {len(overlap(seg[f], other_seg))}" for f in FINALS]
    print(f"{fmt}: reported, allowed: " + "; ".join(reported))
    for t in ("dev", "final-seen"):
        pairs_by = against(t, t, "reserved") | {f"{t} shares a passage with reserved": shingle_pairs(tiers[t], shing["reserved"])}
        for name, pairs in pairs_by.items():
            print(f"{fmt}: reported, allowed: {name}: {len(pairs)}" + (f"  e.g. {pairs[:3]}" if pairs else ""))
    print(f"{fmt}: items " + ", ".join(f"{t} {len(v)}" for t, v in tiers.items()) + f"; reserved texts {len(reserved)}")
    print(f"{fmt}: families " + ", ".join(f"{f}:{next(iter(t))}" for f, t in sorted(families.items())))
    return sum(len(p) for p in failures.values())


def main(data: Path) -> int:
    reg = registry.load()
    fits = {fmt: tiers_of(data, fmt)["fit"] for fmt in reg["formats"]}
    bad = 0
    for fmt in reg["formats"]:
        bad += check_format(reg, data, fmt, [e for f, exs in fits.items() if f != fmt for e in exs])
    print("PASS contamination" if bad == 0 else f"FAIL contamination: {bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(Path(sys.argv[1])))
