"""Check every licence in data/registry.yaml against the dataset card at the pinned revision,
and the tier assignment against the licence policy.

    uv run --group pipeline python scripts/verify_licences.py [--offline]

For each Hugging Face source: the card's `license` field at the pinned revision (or at
`licence_revision`, for a source pinned to a converted-parquet branch) must equal the
registry's `licence` ("none on card" when the card has no field), and each quoted
`licence_text` must appear in the card's README. For a recorded `licence_decision` that quotes
a primary source, the quote must appear in the page at `source`. Then the policy: no fit-tier
source may have a licence the policy excludes. Sources from `raw` files (not on the Hub) have
no card to check; their licence evidence is in docs/licences.md. --offline skips the card and
source checks. Exit 1 on any problem.
"""

import json
import re
import sys
import urllib.parse
import urllib.request

from registry import check_policy, fit_allowed, load, tier_of

API = "https://huggingface.co/api/datasets/{id}/revision/{rev}"
README = "https://huggingface.co/datasets/{id}/raw/{rev}/README.md"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def words(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def card_check(name: str, src: dict) -> list[str]:
    rev = src.get("licence_revision") or src["revision"]
    info = json.loads(fetch(API.format(id=src["hub"], rev=urllib.parse.quote(rev, safe=""))))
    card = (info.get("cardData") or {}).get("license")
    card = [card] if isinstance(card, str) else (card or ["none on card"])
    problems = []
    if sorted(card) != sorted(src["licence"]):
        problems.append(f"{name}: card says {card}, registry says {src['licence']}")
    if src.get("licence_text"):
        readme = words(fetch(README.format(id=src["hub"], rev=rev)).decode("utf-8", "replace"))
        for part in src["licence_text"].split("[...]"):
            if words(part) not in readme:
                problems.append(f"{name}: quoted licence text not found in the card: {part[:60]!r}...")
    return problems


def decision_check(name: str, src: dict) -> list[str]:
    decision = src.get("licence_decision")
    if not (isinstance(decision, dict) and decision.get("source") and decision.get("quote")):
        return []
    page = words(re.sub(r"<[^>]+>", " ", fetch(decision["source"]).decode("utf-8", "replace")))
    if words(decision["quote"]) not in page:
        return [f"{name}: the licence decision's quote was not found at {decision['source']}"]
    return []


def main(offline: bool) -> int:
    reg = load()
    problems = check_policy(reg)
    for name, src in reg["sources"].items():
        tier = tier_of(reg, name)
        fit_ok, why = fit_allowed(reg, name)
        status = ""
        if src.get("hub") and src.get("revision") and not offline:
            found = card_check(name, src) + decision_check(name, src)
            problems += found
            status = "card ok" if not found else "CARD MISMATCH"
        use = "not used" if not src.get("use", True) or tier == "none" else tier
        print(f"{name:22s} {src['format']:8s} {use:10s} {','.join(src.get('licence') or []):32s} "
              f"{'fit-ok' if fit_ok else 'eval-only':9s} {status}")
    for p in problems:
        print("PROBLEM", p)
    print("PASS licences" if not problems else f"FAIL licences: {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["--offline"]):
        sys.exit(__doc__)
    sys.exit(main(bool(sys.argv[1:])))
