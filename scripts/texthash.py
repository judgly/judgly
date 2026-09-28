"""Normalised text matching, the one definition shared by the tier builder and the contamination
checker. Four levels:

  exact      two texts match when they are equal after lower-casing, dropping the Penn Treebank
             bracket tokens of tokenised Wikipedia text (-LRB- and the like) and collapsing every
             run of characters other than a-z and 0-9 into one space (`digest`). Texts shorter
             than MIN_CHARS after that carry too little to identify an item ("thank you so much")
             and are not hashed.
  segment    two passages match when they share a sentence (or title) of at least SEGMENT_CHARS
             normalised characters (`segment_digests`): the same abstract under another title,
             or a passage quoted inside a longer one. A sentence that starts with a short
             "Title: " prefix (FEVER evidence starts each page's sentences with the page title)
             is also hashed without it.
  passage    two texts match when they share at least SHINGLE_MIN distinct word SHINGLE-grams
             that are not boilerplate (`ShingleIndex`): a reworded or re-segmented copy of the
             same passage (a preprint and the published abstract, a Wikipedia paragraph and
             FEVER's tokenised sentences of it). An n-gram found in more than SHINGLE_DF distinct
             indexed texts is boilerplate ("severe acute respiratory syndrome coronavirus 2")
             and is not counted.
  near       two texts are near duplicates when the Jaccard similarity of their sets of
             normalised words is at least NEAR_JACCARD, both having at least NEAR_MIN_WORDS
             distinct words (`NearIndex`: candidates from MinHash banding, 40 bands of 3,
             which misses a pair at the threshold with probability below 1e-4; the exact
             Jaccard of every candidate is then computed, so a reported pair always meets
             the threshold).
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

MIN_CHARS = 24
PASSAGE_CHARS = 100
SEGMENT_CHARS = 60
TITLE_CHARS = 100                   # a "Title: " prefix is at most this long
SHINGLE, SHINGLE_MIN, SHINGLE_DF = 8, 5, 3
NEAR_JACCARD = 0.6
NEAR_MIN_WORDS = 6
_BANDS, _ROWS = 40, 3               # a pair at the threshold is missed with probability below 1e-4
_PRIME = (1 << 31) - 1              # (a * h + b) mod p over 32-bit word hashes: below 2^63, no overflow
_rng = np.random.default_rng(20260927)
_A = _rng.integers(1, _PRIME, _BANDS * _ROWS, dtype=np.uint64)
_B = _rng.integers(0, _PRIME, _BANDS * _ROWS, dtype=np.uint64)


_PTB = re.compile(r"-(?:LRB|RRB|LSB|RSB|LCB|RCB|COLON)-")


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _PTB.sub(" ", text).lower()).strip()


def digest(text: str) -> str | None:
    n = norm(text)
    return hashlib.sha256(n.encode()).hexdigest()[:24] if len(n) >= MIN_CHARS else None


def segment_digests(text: str) -> set[str]:
    """Digests of the sentences (split after . ! ? or at a line break) of at least
    SEGMENT_CHARS normalised characters."""
    out = set()
    for part in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        titled = re.match(r"[^:.!?]{1,%d}:\s+(.+)" % TITLE_CHARS, part, re.S)
        for p in (part, titled.group(1)) if titled else (part,):
            n = norm(p)
            if len(n) >= SEGMENT_CHARS:
                out.add(hashlib.sha256(n.encode()).hexdigest()[:24])
    return out


def shingles(text: str) -> frozenset[str]:
    w = norm(text or "").split()
    return frozenset(" ".join(w[i:i + SHINGLE]) for i in range(len(w) - SHINGLE + 1))


class ShingleIndex:
    """Texts to be searched for a shared passage: add(key, text), then shared(text) gives the
    keys of indexed texts sharing at least SHINGLE_MIN distinct non-boilerplate word n-grams
    with it, with that number. Identical texts are indexed once, so a passage repeated across
    items (one abstract behind several claims) is not taken for boilerplate."""

    def __init__(self) -> None:
        self.keys: dict[frozenset[str], str] = {}
        self.where: dict[str, list[str]] = {}

    def __len__(self) -> int:
        return len(self.keys)

    def add(self, key: str, text: str) -> None:
        g = shingles(text)
        if not g or g in self.keys:
            return
        self.keys[g] = key
        for s in g:
            self.where.setdefault(s, []).append(key)

    def update(self, other: "ShingleIndex") -> None:
        for g, key in other.keys.items():
            if g not in self.keys:
                self.keys[g] = key
                for s in g:
                    self.where.setdefault(s, []).append(key)

    def shared(self, text: str) -> list[tuple[str, int]]:
        count: dict[str, int] = {}
        for s in shingles(text):
            keys = self.where.get(s, ())
            if 0 < len(keys) <= SHINGLE_DF:
                for k in keys:
                    count[k] = count.get(k, 0) + 1
        return sorted((k, n) for k, n in count.items() if n >= SHINGLE_MIN)


def words(text: str) -> frozenset[str]:
    return frozenset(norm(text).split())


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def _signature(ws: frozenset[str]) -> np.ndarray:
    h = np.array([int.from_bytes(hashlib.blake2b(w.encode(), digest_size=4).digest(), "little") for w in ws],
                 dtype=np.uint64)
    return ((np.outer(_A, h) + _B[:, None]) % _PRIME).min(axis=1)


class NearIndex:
    """Texts to be searched for near duplicates: add(key, text), then near(text) gives the keys
    of indexed texts whose word Jaccard with it is at least NEAR_JACCARD, with the Jaccard."""

    def __init__(self) -> None:
        self.words: dict[str, frozenset[str]] = {}
        self.buckets: dict[tuple, list[str]] = {}

    def __len__(self) -> int:
        return len(self.words)

    def add(self, key: str, text: str) -> None:
        ws = words(text)
        if len(ws) < NEAR_MIN_WORDS or key in self.words:
            return
        self.words[key] = ws
        sig = _signature(ws)
        for b in range(_BANDS):
            self.buckets.setdefault((b, *sig[b * _ROWS:(b + 1) * _ROWS].tolist()), []).append(key)

    def near(self, text: str) -> list[tuple[str, float]]:
        ws = words(text)
        if len(ws) < NEAR_MIN_WORDS or not self.words:
            return []
        sig = _signature(ws)
        seen: set[str] = set()
        for b in range(_BANDS):
            seen.update(self.buckets.get((b, *sig[b * _ROWS:(b + 1) * _ROWS].tolist()), ()))
        hits = [(k, jaccard(ws, self.words[k])) for k in seen]
        return sorted((k, j) for k, j in hits if j >= NEAR_JACCARD)


def passage_of(question: str) -> str:
    """The passage a question carries before its last line (MMLU-style "This question refers to
    the following information. <passage> <question>"), so that two items asking different
    questions about the same passage match. Empty when there is no such passage of at least
    PASSAGE_CHARS normalised characters (a short lead-in alone does not identify an item)."""
    head, sep, _ = question.strip().rpartition("\n")
    return head if sep and len(norm(head)) >= PASSAGE_CHARS else ""


def split_stance(state: str) -> tuple[str, str]:
    claim, _, evidence = state.partition("\n\nEvidence: ")
    return claim.removeprefix("Claim: "), evidence


def texts_of(example: dict) -> dict[str, str]:
    """The texts of a rendered example that identify it: the claim and the evidence passage of
    a stance item; the state, the item's own question (not the task's fixed wording) and the
    passage inside that question, if any, of a general item."""
    if example.get("format") == "stance":
        claim, evidence = split_stance(example["state"])
        return {"claim": claim, "evidence": evidence}
    question = example.get("source_question") or ""
    return {"state": example["state"], "question": question, "question_passage": passage_of(question)}


def digests_of(example: dict) -> set[str]:
    return {d for d in (digest(t) for t in texts_of(example).values()) if d}


def segments_of(example: dict) -> set[str]:
    out: set[str] = set()
    for t in texts_of(example).values():
        out |= segment_digests(t)
    return out
