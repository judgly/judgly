"""Build the data tiers of every format from data/registry.yaml.

    uv run --group pipeline python scripts/prep_tiers.py OUT_DIR [--quick]

Writes, per format (general, stance):
  OUT_DIR/<format>/fitdev.jsonl      fit tier (split train / validation / test) and the dev tier
                                     (split heldout), extracted together
  OUT_DIR/<format>/final.jsonl       the fresh final tier (split heldout)
  OUT_DIR/<format>/final-flagged.jsonl  fresh families with a recorded caveat (the registry's
                                     `caveat`), reported beside the final tier, never pooled
                                     into it (split heldout)
  OUT_DIR/<format>/final-seen.jsonl  an earlier held-out tier, a secondary evaluation of
                                     previously seen families (split heldout)
  OUT_DIR/<format>/bench.jsonl       external benchmarks, evaluation only (general format;
                                     split heldout; each item carries the benchmark's gold)
  OUT_DIR/<format>/confirm.jsonl     untouched families for the pre-registered confirmation,
                                     evaluation only (split heldout), built last (below)
and OUT_DIR/manifest.json: settings, counts per source, task and split, labels, every drop
and why, the registry's SHA-256 and the SHA-256 of every file written.

Tiers and licences come from the registry, and the licence policy is enforced here (a fit
source whose licence the policy does not allow stops the build). Tiers are built in the order
reserved, final-seen, dev, final, final-flagged, bench, fit, and an item of a later tier is
dropped when it matches an earlier one: always when its claim, passage, state or own question is
equal after normalisation (scripts/texthash.py); for the final, final-flagged and fit tiers also
when a passage shares a sentence with an earlier tier, shares a passage with one (word n-grams),
or a text is a near duplicate of one (word Jaccard). final-seen
and dev are built from their sources alone; the bench tier
drops nothing for overlap (a benchmark is scored as published), only questions longer than the
extraction can take (MAX_CHARS). Exact duplicates are dropped within a tier;
fit items are split 70/15/15 by a hash of their passage (or recorded group), so items sharing
one stay in one split. Items of the final, final-flagged and confirm tiers carry a `group` (items
that share a claim, an abstract, a table, a story or a template), which the calibration record
resamples together for its intervals. scripts/check_contamination.py checks the result.

The confirm tier is built after all the others and drops an item that matches any of them in any
of the four ways, or any text of a reserved source (reserved now, not `was_reserved`), or, for
stance, any claim or passage of an evaluation source (dev, final, final-flagged, final-seen) as
a whole, not only the items its tier holds; an item that matches a confirm source built before
it is dropped too, and a source marked `distinct` keeps no two items that nearly duplicate each
other. The tiers built before it are still checked against every `was_reserved` source, so they
stay byte-identical to the 0.1.0 release.

Everything is deterministic: items are ordered by a hash of their id and the first N kept
(evenly across labels or strata where a source asks for it), so a rerun gives byte-identical
files (the manifest records their SHA-256).

--quick writes small tiers for a smoke run of the whole pipeline in minutes (QUICK below).
"""

from __future__ import annotations

import csv
import datetime
import gzip
import json
import platform
import random
import re
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import registry
from texthash import NearIndex, ShingleIndex, digest, digests_of, norm, passage_of, segment_digests

SEED = 20260926
FULL = {
    # Tier sizes: general fit 9,000 and dev 250 per task (calibration curves flatten after a
    # few hundred labelled items); final 1,000 per task unless the registry's `pool` says
    # otherwise; final-seen 700 per task (as built for that tier). Stance: 4,500 per fit source
    # unless `pool` says otherwise, 700 per dev source, 2,100 per final and final-seen source.
    # The bench tier is every item of every benchmark.
    "general": {"fit_pool": 7000, "dev_pool": 250, "final_pool": 1000, "seen_pool": 700, "bench_pool": None,
                "confirm_pool": 500, "caps": {"train": 6300, "validation": 1350, "test": 1350}},
    "stance": {"fit_pool": 4500, "dev_pool": 700, "final_pool": 2100, "seen_pool": 2100, "bench_pool": None,
               "confirm_pool": None, "caps": {"train": 10 ** 9, "validation": 10 ** 9, "test": 10 ** 9}},
}
QUICK = {
    "general": {"fit_pool": 60, "dev_pool": 8, "final_pool": 8, "seen_pool": 8, "bench_pool": 16,
                "confirm_pool": 8, "caps": {"train": 192, "validation": 64, "test": 32}},
    "stance": {"fit_pool": 90, "dev_pool": 16, "final_pool": 48, "seen_pool": 48, "bench_pool": 16,
               "confirm_pool": 16, "caps": {"train": 96, "validation": 32, "test": 32}},
}
POOL_KEY = {"fit": "fit_pool", "dev": "dev_pool", "final": "final_pool", "final-flagged": "final_pool",
            "final-seen": "seen_pool", "bench": "bench_pool", "confirm": "confirm_pool"}
BUILD_ORDER = ("final-seen", "dev", "final", "final-flagged", "bench", "fit", "confirm")  # after reserved; see the docstring
STRICT = {"final": ("reserved", "final-seen", "dev"),                          # tiers checked for shared
          "final-flagged": ("reserved", "final-seen", "dev", "final"),          # sentences, shared passages
          "fit": ("reserved", "final-seen", "dev", "final", "final-flagged", "bench"),  # and near duplicates
          # "sources": the confirm tier's reserved and (stance) whole evaluation sources (confirm_texts)
          "confirm": ("sources", "final-seen", "dev", "final", "final-flagged", "bench", "fit")}
OUT_FILE = {"fit": "fitdev", "dev": "fitdev", "final": "final", "final-flagged": "final-flagged",
            "final-seen": "final-seen", "bench": "bench", "confirm": "confirm"}
GROUPED = ("final", "final-flagged", "confirm")    # tiers whose rows carry their resampling group
# Longer items are left out, never truncated: s1-features reads 32 examples at a time in one
# context of 32,768 tokens (tools/s1-features.c), so an example must stay far below 1/32 of it.
MAX_CHARS = {"general": 2500, "stance": 4000, "bench": 20000}
MAX_SHOWN = 10          # a task with more classes shows the right one and MAX_SHOWN - 1 others
P_DISTRACTOR, P_NONE = 0.2, 0.1
NONE_KEY, NONE_TEXT = "none", "None of these"
SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2, "heldout": 3}


@dataclass
class Item:
    id: str
    state: str
    question: str | None                     # the item's own question, if any
    options: list[tuple[str, str]]           # empty for bool and score
    label: str
    task: str = ""                           # set for generated items (several tasks in one source)
    type: str = ""
    levels: int = 0
    wordings: tuple[str, ...] = ()
    claim: str = ""                          # stance only
    evidence: str = ""
    group: str = ""                          # items of one group share a fit split (and, with
                                             # Spec.one_per_group, at most one is kept)
    stratum: str = ""                        # sampled evenly across strata (Spec.balance)
    row: dict | None = None                  # a bench item, rendered as published
    context: str = ""                        # stance: the document the evidence was taken from, if
                                             # only part of it is shown (checked, never shown)
    cluster: str = ""                        # the resampling group of an evaluation item, where it
                                             # differs from `group` (default: group, else the id)


@dataclass
class Spec:
    convert: Callable | None                 # (row, item_id) -> Item | None
    wordings: tuple[str, ...] = ("{q}",)
    open_options: bool = False
    needs_names: str = ""
    levels: int = 0
    balance: bool = False                    # sample evenly across labels (or Item.stratum)
    one_per_group: bool = False              # keep at most one item per Item.group
    distinct: bool = False                   # confirm: no two kept items nearly duplicate each other
    setup: Callable | None = None            # (reg) -> convert, for converters that need other data


def unit(text: str) -> float:
    import hashlib
    return int(hashlib.sha256(text.encode()).hexdigest()[:13], 16) / 16 ** 13


# ---- converters -------------------------------------------------

def fixed(labels: dict[str, str], state_col, label_col: str, names: list[str]):
    def convert(row, item_id):
        value = row[label_col]
        if value is None or value < 0 or value >= len(names):
            return None
        state = state_col(row) if callable(state_col) else row[state_col]
        right = names[value]
        options = list(labels.items())
        if len(options) > MAX_SHOWN:
            rng = random.Random(f"{SEED}:{item_id}")
            others = [o for o in options if o[0] != right]
            options = [(right, labels[right])] + rng.sample(others, MAX_SHOWN - 1)
        return Item(item_id, state, None, options, right)
    return convert


def lettered(state_col, question_col, choices, answer):
    def convert(row, item_id):
        texts, right = choices(row), answer(row)
        if right is None or not 0 <= right < len(texts) or len(set(texts)) != len(texts):
            return None
        if len(texts) > MAX_SHOWN:
            rng = random.Random(f"{SEED}:{item_id}")
            others = rng.sample([t for i, t in enumerate(texts) if i != right], MAX_SHOWN - 1)
            texts, right = [texts[right]] + others, 0
        options = [(f"o{i + 1}", str(t)) for i, t in enumerate(texts)]
        return Item(item_id, row[state_col] if state_col else "", row[question_col], options, options[right][0])
    return convert


def key_index(row):
    labels = row["choices"]["label"]
    return labels.index(row["answerKey"]) if row["answerKey"] in labels else None


def from_names(state_col, label_col):
    def build(names):
        return fixed({n: n.replace("_", " ").replace(".", " ") for n in names}, state_col, label_col, names)
    return build


def pair(state_of, claim_col, options, label_of):
    def convert(row, item_id):
        k = label_of(row)
        if k is None or not 0 <= k < len(options):
            return None
        state = state_of(row) if callable(state_of) else row[state_of]
        return Item(item_id, state, row[claim_col], options, options[k][0])
    return convert


def boolean(state_of, statement_of, truth_of):
    def convert(row, item_id):
        truth = truth_of(row)
        if truth is None:
            return None
        return Item(item_id, state_of(row), statement_of(row), [], "true" if truth else "false")
    return convert


def scored(state_col, level_of, question_col=None):
    def convert(row, item_id):
        level = level_of(row)
        if level is None:
            return None
        return Item(item_id, row[state_col], row[question_col] if question_col else None, [], str(level))
    return convert


def preference(row, item_id):
    chosen, rejected = row["chosen"], row["rejected"]
    cut = chosen.rfind("\n\nAssistant:")
    if cut < 0 or not rejected.startswith(chosen[:cut]):
        return None
    tail = len("\n\nAssistant:")
    a, b = chosen[cut + tail:].strip(), rejected[cut + tail:].strip()
    if not a or not b or a == b:
        return None
    return Item(item_id, chosen[:cut].strip(), None, [("o1", a), ("o2", b)], "o1")


def go_emotions(names):
    """New in judgly: single-label GoEmotions items; the class set is sampled down to MAX_SHOWN."""
    labels = {n: ("No particular emotion" if n == "neutral" else f"The writer expresses {n}") for n in names}

    def convert(row, item_id):
        if len(row["labels"]) != 1:
            return None
        return fixed(labels, "text", "_label", names)(row | {"_label": row["labels"][0]}, item_id)
    return convert


def toxicity_band(row):
    """New in judgly: the Civil Comments toxicity fraction in five bands."""
    t = row["toxicity"]
    return 1 if t < 0.1 else 2 if t < 0.3 else 3 if t < 0.5 else 4 if t < 0.7 else 5


def helpsteer(row, item_id):
    """New in judgly: HelpSteer2 helpfulness 0-4 as levels 1-5; items sharing a prompt share a
    fit split."""
    h = row.get("helpfulness")
    if h is None or not 0 <= h <= 4 or not row["prompt"].strip() or not row["response"].strip():
        return None
    state = f"User request:\n{row['prompt'].strip()}\n\nResponse:\n{row['response'].strip()}"
    return Item(item_id, state, None, [], str(h + 1), group="prompt:" + norm(row["prompt"]))


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def ledgar(names):
    """New in judgly: the provision type, shown with three other types of the 100."""
    def convert(row, item_id):
        k = row["label"]
        if k is None or not 0 <= k < len(names):
            return None
        rng = random.Random(f"{SEED}:{item_id}")
        shown = [names[k]] + rng.sample([n for i, n in enumerate(names) if i != k], 3)
        return Item(item_id, row["text"], None, [(slug(n), n) for n in shown], slug(names[k]))
    return convert


def bbq(row, item_id):
    texts = [row["ans0"], row["ans1"], row["ans2"]]
    if row["label"] not in (0, 1, 2) or len(set(texts)) != 3:
        return None
    options = [(f"o{i + 1}", t) for i, t in enumerate(texts)]
    return Item(item_id, row["context"], row["question"], options, options[row["label"]][0],
                group=f'{row["category"]}|{row["question_index"]}|{row["question_polarity"]}|{row["context_condition"]}',
                stratum=f'{row["category"]}|{row["context_condition"]}',
                cluster=f'{row["category"]}|{row["question_index"]}')       # one template


def blimp(row, item_id):
    """New in judgly: the two sentences of a minimal pair in a random order; which one is
    acceptable. Stratified by paradigm."""
    good, bad = row["sentence_good"], row["sentence_bad"]
    if not good or not bad or good == bad:
        return None
    first_good = random.Random(f"{SEED}:{item_id}").random() < 0.5
    a, b = (good, bad) if first_good else (bad, good)
    return Item(item_id, f"1. {a}\n2. {b}", None, [("first", "The first sentence"), ("second", "The second sentence")],
                "first" if first_good else "second", stratum=row["_tag"])


def fig_qa(row, item_id):
    if row["labels"] not in ("0", "1") or row.get("valid", "1") != "1":
        return None
    return Item(item_id, row["startphrase"], None, [("o1", row["ending1"]), ("o2", row["ending2"])],
                "o1" if row["labels"] == "0" else "o2")


CIRCA = {0: ("yes", "Yes"), 1: ("no", "No"), 3: ("conditional", "Yes, subject to some conditions"),
         2: ("middle", "In the middle, neither yes nor no")}


def circa(row, item_id):
    g = row["goldstandard2"]
    if g not in CIRCA:
        return None
    state = f'{row["context"]}\nX asks: "{row["question-X"]}"\nY answers: "{row["answer-Y"]}"'
    return Item(item_id, state, None, list(CIRCA.values()), CIRCA[g][0], group="q:" + norm(row["question-X"]))


def tabfact(row, item_id):
    if row["label"] not in (0, 1):
        return None
    table = "\n".join(" | ".join(c.strip() for c in line.split("#")) for line in row["table_text"].strip().split("\n"))
    return Item(item_id, f'Table: {row["table_caption"]}\n{table}', row["statement"], [],
                "true" if row["label"] == 1 else "false", group=row["table_id"])


ESCI = {"Irrelevant": 1, "Complement": 2, "Substitute": 3, "Exact": 4}


def esci(row, item_id):
    """New in judgly: ESCI relevance as four levels; the query, the product title and the
    product's first bullet point; one product per query."""
    if row["product_locale"] != "us":
        return "not the US locale"
    if row["esci_label"] not in ESCI or not row["product_title"]:
        return None
    bullet = (row["product_bullet_point"] or "").strip().split("\n")[0].strip()
    state = f'Search query: "{row["query"]}"\nProduct: {row["product_title"].strip()}' + (f"\n{bullet}" if bullet else "")
    return Item(item_id, state, None, [], str(ESCI[row["esci_label"]]), group=str(row["query_id"]))


CEFR = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
POLITENESS = {"impolite": 1, "neutral": 2, "polite": 3}


def short(prefix: str, text: str) -> str:
    """A group name for items sharing a text: the prefix and a hash of the normalised text."""
    import hashlib
    return prefix + hashlib.sha256(norm(text).encode()).hexdigest()[:16]


# ---- confirm tier converters (families new on 2026-09-29) -----------------------------------

FEMALE = {"aunt", "daughter", "daughter-in-law", "granddaughter", "grandmother", "mother", "mother-in-law", "niece",
          "sister"}


def clutrr(names):
    """The gold relation and three other relations of the same gender (so that a pronoun in the
    story does not give the answer away), as "<tail> is <head>'s <relation>"."""
    def convert(row, item_id):
        k = row["label"]
        if k is None or not 0 <= k < len(names):
            return None
        gold = names[k]
        same = [n for n in names if n != gold and (n in FEMALE) == (gold in FEMALE)]
        shown = [gold] + random.Random(f"{SEED}:{item_id}").sample(same, 3)
        options = [(slug(n), f"{row['tail']} is {row['head']}'s {n}") for n in shown]
        return Item(item_id, row["story"], row["query"], options, slug(gold), group=short("story:", row["story"]),
                    stratum=f"hops{int(row['hops']):02d}")
    return convert


def spartqa(row, item_id):
    if row["answer"] not in ("Yes", "No"):
        return "answer DK (not a yes/no answer)"
    return Item(item_id, row["story"], row["question"].replace(" ,", ","), [], "true" if row["answer"] == "Yes" else "false",
                group=short("story:", row["story"]))


def argument_quality(row, item_id):
    wa = float(row["WA"])
    state = f"Topic: {row['topic'].strip()}\nArgument: {row['argument'].strip()}"
    return Item(item_id, state, None, [], str(1 if wa < 0.6 else 2 if wa < 0.85 else 3), group=short("topic:", row["topic"]))


def humicroedit(row, item_id):
    g = float(row["meanGrade"])
    level = 1 if g <= 0.4 + 1e-9 else 2 if 0.6 - 1e-9 <= g <= 1.2 + 1e-9 else 3 if g >= 1.4 - 1e-9 else None
    if level is None:
        return "mean grade between two levels"
    state = f"Original headline: {row['headline'].strip()}\nEdited headline: {row['edited'].strip()}"
    return Item(item_id, state, None, [], str(level), group=short("headline:", row["headline"]))


VERDICT = {"Compile Error": "Compile Error", "Runtime Error": "Runtime Error", "Time Limit Exceeded": "Time Limit Exceeded",
           "Memory Limit Exceeded": "Memory Limit Exceeded", "Internal error": "Internal error",
           "No abnormally found": "Runs without error"}


def code_outcome(row, item_id):
    choices, k = list(row["choices"]), "ABCD".find(row["answer"] or "")
    if not 0 <= k < len(choices) or len(set(choices)) != len(choices) or any(c not in VERDICT for c in choices):
        return None
    if choices[k] == "Memory Limit Exceeded":
        return "verdict Memory Limit Exceeded (too rare to balance)"
    if len(row["question"]) > 1500:
        return "program longer than 1,500 characters"
    options = [(slug(VERDICT[c]), VERDICT[c]) for c in choices]
    return Item(item_id, row["question"], None, options, options[k][0])


NLI3 = [("entailment", "The text supports the claim"),
        ("neutral", "The text neither supports nor contradicts the claim"),
        ("contradiction", "The text contradicts the claim")]
CLAIM_OPTIONS = [("supports", "The evidence supports the claim"), ("refutes", "The evidence refutes the claim"),
                 ("not_enough_info", "The evidence is not enough to decide")]
SENTIMENT = {"positive": "The review is positive", "negative": "The review is negative"}
NEWS = {"World": "World news and politics", "Sports": "Sport", "Business": "Business and finance",
        "Sci/Tech": "Science and technology"}
TREC = {"ABBR": "An abbreviation or its expansion", "ENTY": "An entity such as an animal, product or term",
        "DESC": "A description, definition or reason", "HUM": "A person or a group of people",
        "LOC": "A place", "NUM": "A number, date or quantity"}
QW = ("{q}", "Question: {q}", "Answer this question. {q}")
nli_label = lambda r: r["label"] if r["label"] in (0, 1, 2) else None

GENERAL: dict[str, Spec] = {
    # fit
    "mmlu": Spec(lettered(None, "question", lambda r: r["choices"], lambda r: r["answer"]), QW, True),
    "commonsense_qa": Spec(lettered(None, "question", lambda r: r["choices"]["text"], key_index), QW, True),
    "qasc": Spec(lettered(None, "question", lambda r: r["choices"]["text"], key_index), QW, True),
    "strategyqa": Spec(boolean(lambda r: f'{r["term"]}: {r["description"]}',
                               lambda r: f'The answer to "{r["question"]}" is yes.', lambda r: bool(r["answer"]))),
    "cosmos_qa": Spec(lettered("context", "question", lambda r: [r[f"answer{i}"] for i in range(4)],
                               lambda r: r["label"]), QW, True),
    "civil_comments": Spec(boolean(lambda r: r["text"], lambda r: "This comment is toxic.",
                                   lambda r: None if 0.3 < r["toxicity"] < 0.7 else r["toxicity"] >= 0.7)),
    "civil_comments_score": Spec(scored("text", toxicity_band),
                                 ("How toxic is this comment, from 1 (not at all) to 5 (very toxic)?",
                                  "Rate the toxicity of the comment from 1 (none) to 5 (severe)."), levels=5, balance=True),
    "hh_rlhf": Spec(preference, ("Which final assistant reply is better?",), True),
    "go_emotions": Spec(go_emotions, ("Which emotion does the writer express?",
                                      "Pick the emotion that fits the text best."), needs_names="labels"),
    "banking77": Spec(from_names("text", "label"), ("Which intent does this customer message express?",),
                      needs_names="label"),
    "helpsteer2": Spec(helpsteer, ("How helpful is the response to the user's request, from 1 (not helpful) to 5 (very helpful)?",
                                   "Rate the helpfulness of the response from 1 (not at all) to 5 (fully helpful)."),
                       levels=5, balance=True),
    "ledgar": Spec(ledgar, ("Which type of contract provision is this?", "What kind of clause is this contract provision?"),
                   needs_names="label"),
    # dev
    "sst2": Spec(fixed(SENTIMENT, "sentence", "label", ["negative", "positive"]),
                 ("What is the sentiment of this review?", "Is this review positive or negative?")),
    "imdb": Spec(fixed(SENTIMENT, "text", "label", ["negative", "positive"]),
                 ("What is the sentiment of this review?", "Is this review positive or negative?")),
    "rotten_tomatoes": Spec(fixed(SENTIMENT, "text", "label", ["negative", "positive"]),
                            ("What is the sentiment of this review?", "Decide the sentiment of the text.")),
    "tweet_sentiment": Spec(fixed({"negative": "Negative", "neutral": "Neutral", "positive": "Positive"}, "text",
                                  "label", ["negative", "neutral", "positive"]), ("What is the sentiment of this tweet?",)),
    "tweet_irony": Spec(boolean(lambda r: r["text"], lambda r: "This tweet is ironic.", lambda r: r["label"] == 1)),
    "rte": Spec(pair("sentence1", "sentence2", [("entailment", "The text supports the claim"),
                                                 ("not_entailment", "The text does not support the claim")],
                     lambda r: r["label"] if r["label"] >= 0 else None), ("Does the text support this claim? {q}",)),
    "cb": Spec(pair("premise", "hypothesis", NLI3, nli_label), ("Does the text support this claim? {q}",)),
    "anli_general": Spec(pair("premise", "hypothesis", NLI3, nli_label), ("Does the text support this claim? {q}",)),
    "stsb": Spec(scored("sentence1", lambda r: 1 + round(float(r["score"]) * 4), "sentence2"),
                 ("How similar in meaning is the text to this sentence, from 1 (unrelated) to 5 (equivalent)? {q}",),
                 levels=5),
    "mrpc": Spec(boolean(lambda r: r["text1"], lambda r: f'This sentence says the same thing: "{r["text2"]}"',
                         lambda r: r["label"] == 1)),
    "qqp": Spec(boolean(lambda r: r["question1"], lambda r: f'This question asks the same thing: "{r["question2"]}"',
                        lambda r: None if r["label"] < 0 else r["label"] == 1)),
    "wic": Spec(boolean(lambda r: f'1. {r["sentence1"]}\n2. {r["sentence2"]}',
                        lambda r: f'The word "{r["word"]}" has the same meaning in both sentences.',
                        lambda r: None if r["label"] < 0 else r["label"] == 1)),
    "winogrande": Spec(lettered("sentence", "sentence", lambda r: [r["option1"], r["option2"]],
                                lambda r: int(r["answer"]) - 1 if r["answer"] in ("1", "2") else None),
                       ("Which option fills the blank (_) in the text?",), True),
    "piqa": Spec(lettered(None, "goal", lambda r: [r["sol1"], r["sol2"]], lambda r: r["label"] if r["label"] >= 0 else None),
                 ("Which solution achieves this goal? {q}",), True),
    "siqa": Spec(lettered("context", "question", lambda r: [r["answerA"], r["answerB"], r["answerC"]],
                          lambda r: int(r["label"]) - 1), QW, True),
    "copa": Spec(lettered("premise", "question", lambda r: [r["choice1"], r["choice2"]],
                          lambda r: r["label"] if r["label"] >= 0 else None), ("What was the {q} of this?",), True),
    "swag": Spec(lettered("startphrase", "sent2", lambda r: [r[f"ending{i}"] for i in range(4)], lambda r: r["label"]),
                 ("Which ending continues the text most plausibly?",), True),
    "hellaswag": Spec(lettered("ctx", "activity_label", lambda r: r["endings"],
                               lambda r: int(r["label"]) if r["label"] != "" else None),
                      ("Which ending continues the text most plausibly?", "Choose the most likely continuation."), True),
    # final (fresh)
    "bbq": Spec(bbq, ("{q}",), balance=True, one_per_group=True),
    "blimp": Spec(blimp, ("Which of the two sentences is grammatically acceptable English?",), balance=True),
    "fig_qa": Spec(fig_qa, ("What does the figurative sentence mean?", "Which interpretation of the text is right?")),
    "circa": Spec(circa, ("How does X understand Y's answer?",), balance=True),
    "ethics_deontology": Spec(boolean(lambda r: f'Request: "{r["scenario"]}"\nReply: "{r["excuse"]}"',
                                      lambda r: "The reply gives a reasonable excuse.",
                                      lambda r: {"1": True, "0": False}.get(r["label"]))),
    "ethics_justice": Spec(boolean(lambda r: r["scenario"], lambda r: "This is reasonable and just.",
                                   lambda r: {"1": True, "0": False}.get(r["label"]))),
    "tabfact": Spec(tabfact, ("According to the table, this statement is true: {q}",)),
    "esci": Spec(esci, ("How relevant is the product to the search query, from 1 to 4? 1: irrelevant; 2: a complement "
                        "(useful together with what was searched for); 3: a substitute (could be used instead); "
                        "4: an exact match.",), levels=4, balance=True, one_per_group=True),
    "cefr_sp": Spec(scored("text", lambda r: CEFR.get(r["cefr_level"])),
                    ("How difficult is this sentence for a learner of English, from 1 (A1, beginner) to 6 (C2, proficient)?",),
                    levels=6, balance=True),
    "politeness": Spec(scored("prompt", lambda r: POLITENESS.get(r["completion"])),
                       ("How polite is this request, from 1 (impolite) to 3 (polite)?",), levels=3, balance=True),
    # final-seen (an earlier held-out tier)
    "ag_news": Spec(fixed(NEWS, "text", "label", list(NEWS)), ("Which section of a newspaper does this article belong to?",)),
    "trec": Spec(fixed(TREC, "text", "coarse_label", list(TREC)), ("What kind of answer does this question ask for?",)),
    "newsgroups": Spec(from_names("text", "label"), ("Which newsgroup was this message posted to?",), needs_names="label"),
    "dbpedia": Spec(from_names(lambda r: f'{r["title"]}: {r["content"]}', "label"),
                    ("What kind of thing does this article describe?",), needs_names="label"),
    "yahoo": Spec(from_names(lambda r: f'{r["question_title"]} {r["question_content"]}', "topic"),
                  ("Which topic does this question belong to?",), needs_names="topic"),
    "pubmedqa": Spec(pair(lambda r: " ".join(r["context"]["contexts"]), "question",
                          [("yes", "Yes"), ("no", "No"), ("maybe", "Maybe")],
                          lambda r: {"yes": 0, "no": 1, "maybe": 2}.get(r["final_decision"])),
                     ("According to the abstract, what is the answer? {q}",)),
    "medmcqa": Spec(lettered(None, "question", lambda r: [r["opa"], r["opb"], r["opc"], r["opd"]], lambda r: r["cop"]),
                    QW, True),
    "med_qa": Spec(lettered(None, "question", lambda r: r["choices"],
                            lambda r: r["choices"].index(r["answer"][0]) if r["answer"] and r["answer"][0] in r["choices"]
                            else None), QW, True),
    "casehold": Spec(lettered(None, "citing_prompt", lambda r: [r[f"holding_{i}"] for i in range(5)],
                              lambda r: int(r["label"]) if str(r["label"]).isdigit() else None),
                     ("Which holding does the citation in this passage refer to?",), True),
    "aqua_rat": Spec(lettered(None, "question", lambda r: [o.split(")", 1)[-1] for o in r["options"]],
                              lambda r: "ABCDE".find(r["correct"])), QW, True),
    "fin_tweets": Spec(fixed({"bearish": "Bearish", "bullish": "Bullish", "neutral": "Neutral"}, "text", "label",
                             ["bearish", "bullish", "neutral"]), ("What is the sentiment of this financial news tweet?",)),
    "truthful_qa": Spec(lettered(None, "question", lambda r: r["mc1_targets"]["choices"],
                                 lambda r: r["mc1_targets"]["labels"].index(1)), ("{q}",), True),
    "mmlu_pro": Spec(lettered(None, "question", lambda r: r["options"], lambda r: r["answer_index"]), ("{q}",), True),
    "yelp": Spec(scored("text", lambda r: r["label"] + 1),
                 ("How many stars did the reviewer give, from 1 (worst) to 5 (best)?",
                  "Rate how positive this review is, from 1 (very negative) to 5 (very positive)."), levels=5),
    "boolq": Spec(boolean(lambda r: r["passage"], lambda r: r["question"].rstrip("?") + "?", lambda r: r["answer"]),
                  ('The answer to the question "{q}" is yes.', 'According to the text, the answer to "{q}" is yes.')),
    # confirm (untouched families, 2026-09-29)
    "clutrr": Spec(clutrr, ("{q}",), needs_names="label", balance=True, one_per_group=True),
    "spartqa": Spec(spartqa, ('The answer to the question "{q}" is yes.',), balance=True, one_per_group=True),
    "argument_quality": Spec(argument_quality, ("Disregarding your own opinion on the topic, how strongly would you "
                                                "recommend using this argument, as is, in a speech on the topic, from 1 "
                                                "(not recommended) to 3 (recommended)?",), levels=3, balance=True),
    "humicroedit": Spec(humicroedit, ("How funny is the edited headline, from 1 (not funny) to 3 (funny)?",),
                        levels=3, balance=True),
    "code_outcome": Spec(code_outcome, ("This program was submitted to an online judge and run on the problem's "
                                        "hidden test inputs. What was the verdict?",), balance=True, distinct=True),
}


# ---- generated tasks (five choice, two bool and two score generators) ----------------------

def _choice(task, item_id, state, question, values, right):
    options = [(f"o{j + 1}", str(v)) for j, v in enumerate(values)]
    return Item(item_id, state, question, options, options[right][0], task=task, type="choice", wordings=("{q}",))


def generated(n: int) -> Iterator[Item]:
    import datetime as dt
    rng = random.Random(f"{SEED}:generated")
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fruit = ["apple", "banana", "cherry", "avocado", "blueberry", "apricot", "kiwi", "mango", "melon", "orange",
             "olive", "peach", "pear", "plum", "quince", "raspberry", "fig", "grape", "lime", "lemon"]
    conversions = [("minutes", "hours", 60), ("centimetres", "metres", 100), ("grams", "kilograms", 1000),
                   ("millilitres", "litres", 1000), ("seconds", "minutes", 60), ("metres", "kilometres", 1000)]
    people = ["Ash", "Bea", "Cai", "Dev", "Eli", "Fay", "Gus", "Ida"]
    for i in range(n):
        kind = i % 9
        iid = f"generated-{i:06d}"
        if kind == 0:
            boxes, per, sold = rng.randint(3, 40), rng.randint(4, 60), rng.randint(1, 100)
            right = boxes * per - sold
            wrong = {boxes * per + sold, boxes * per, right + rng.choice([-10, -1, 1, 10]), (boxes - 1) * per - sold} - {right}
            yield _choice("gen_arithmetic", iid, f"A warehouse receives {boxes} boxes with {per} units in each box. "
                          f"During the week {sold} units are shipped to customers.",
                          "How many units remain in the warehouse?", [right] + sorted(wrong)[:3], 0)
        elif kind == 1:
            start = dt.date(2031, 1, 1) + dt.timedelta(days=rng.randint(0, 3000))
            ahead = rng.randint(2, 60)
            yield _choice("gen_dates", iid, f"The contract was signed on {start:%A %-d %B %Y}. Delivery is due {ahead} days later.",
                          "On which day of the week is delivery due?", days, (start + dt.timedelta(days=ahead)).weekday())
        elif kind == 2:
            small, big, factor = rng.choice(conversions)
            whole = rng.randint(2, 45)
            wrong = {whole * 10, whole // 2 if whole > 3 else whole + 5, whole + 1, whole * factor,
                     whole * factor // 10} - {whole}
            yield _choice("gen_units", iid, f"The sensor logged a reading of {whole * factor} {small}.",
                          f"How many {big} is that?", [whole] + sorted(wrong)[:3], 0)
        elif kind == 3:
            chosen = rng.sample(fruit, rng.randint(5, 9))
            letter = rng.choice(sorted({w[0] for w in chosen}))
            right = sum(w.startswith(letter) for w in chosen)
            wrong = sorted({right + 1, right + 2, max(0, right - 1), right + 3} - {right})[:3]
            yield _choice("gen_counting", iid, "Shopping list: " + ", ".join(chosen) + ".",
                          f'How many items on the list start with the letter "{letter}"?', [right] + wrong, 0)
        elif kind == 4:
            names = rng.sample(people, rng.randint(3, 5))
            values = rng.sample(range(10, 400), len(names))
            yield _choice("gen_order", iid, ". ".join(f"{a} scored {v} points" for a, v in zip(names, values)) + ".",
                          "Who scored the most points?", names, values.index(max(values)))
        elif kind == 5:
            a, b = rng.sample(range(100, 9999), 2)
            claim = rng.choice([f"{a} is larger than {b}.", f"{a} is smaller than {b}."])
            truth = (a > b) == ("larger" in claim)
            yield Item(iid, f"Two invoices were issued: invoice A for {a} dollars and invoice B for {b} dollars.",
                       claim.replace(str(a), "Invoice A", 1).replace(str(b), "invoice B", 1),
                       [], "true" if truth else "false", task="gen_compare", type="bool", wordings=("{q}",))
        elif kind == 6:
            d1 = dt.date(2030, 1, 1) + dt.timedelta(days=rng.randint(0, 2000))
            d2 = d1 + dt.timedelta(days=rng.choice([-1, 1]) * rng.randint(1, 400))
            yield Item(iid, f"The inspection took place on {d1:%-d %B %Y} and the repair on {d2:%-d %B %Y}.",
                       "The repair happened before the inspection.", [], "true" if d2 < d1 else "false",
                       task="gen_date_order", type="bool", wordings=("{q}",))
        elif kind == 7:
            checks = ["backup", "firewall", "updates", "antivirus", "password policy", "audit log", "encryption"]
            picked = rng.sample(checks, 5)
            passed = rng.randint(1, 5)
            ok = set(rng.sample(picked, passed))
            state = "Security review: " + "; ".join(f"{c} {'passed' if c in ok else 'failed'}" for c in picked) + "."
            yield Item(iid, state, None, [], str(passed), task="gen_checklist", type="score", levels=5,
                       wordings=("How many of the five checks passed, from 1 to 5?",))
        else:
            t = rng.randint(-15, 45)
            band = 1 if t < 0 else 2 if t < 10 else 3 if t < 20 else 4 if t < 30 else 5
            yield Item(iid, f"The weather station recorded {t} degrees Celsius at noon.", None, [], str(band),
                       task="gen_temperature", type="score", levels=5,
                       wordings=("On a scale of 1 (below 0 C) to 5 (30 C or more), in bands of 10 degrees, how warm was it?",))


# ---- stance ---------------------------------------------------------------------

STANCE_KEYS = ("supports", "contradicts", "no_bearing")
STANCE_OPTIONS = [
    ("The evidence supports the claim", "The evidence contradicts the claim", "The evidence has no bearing on the claim"),
    ("The passage is consistent with the claim", "The passage contradicts the claim", "The passage is independent of the claim"),
    ("The evidence backs the claim", "The evidence argues against the claim",
     "The evidence neither backs nor argues against the claim"),
]
STANCE_QUESTIONS = ("How does the evidence bear on the claim?", "What is the relation between the evidence and the claim?",
                    "Does the evidence support or contradict the claim, or neither?")
NLI_STANCE = {0: "supports", 1: "no_bearing", 2: "contradicts"}
FEVER_STANCE = {"SUPPORTS": "supports", "REFUTES": "contradicts", "NOT ENOUGH INFO": "no_bearing"}


def stance_item(item_id, claim, evidence, label):
    if not label or not claim or not evidence:
        return None
    return Item(item_id, "", None, [], label, claim=claim.strip(), evidence=evidence.strip())


WIKI_TOKENS = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}", "-COLON-": ":"}


def unwiki(text: str) -> str:
    """FEVER's tokenised Wikipedia text and page titles, with the bracket tokens restored."""
    for token, char in WIKI_TOKENS.items():
        text = text.replace(token, char)
    return text.replace("_", " ").strip()


def page_key(title: str) -> str:
    return norm(unwiki(title))


def climate_fever_pages(reg: dict) -> set[str]:
    """The Wikipedia articles Climate-FEVER (stance dev) draws its evidence from."""
    from datasets import load_dataset
    info = reg["sources"]["climate_fever"]
    pages = set()
    for split in info["splits"]:
        for row in load_dataset(info["hub"], info.get("config"), split=split, revision=info["revision"]):
            pages |= {page_key(e["article"]) for e in row["evidences"]}
    return pages


def fever_setup(reg: dict):
    """New in judgly: FEVER's SUPPORTS and REFUTES claims with their gold evidence sentences
    grouped by page, each group prefixed by its page title. NOT ENOUGH INFO claims are left out:
    their evidence was retrieved by a system (not annotated by FEVER), in a release whose card
    also names GPL-3.0 (docs/licences.md). A claim whose evidence page is also a Climate-FEVER
    evidence article is left out; items sharing an evidence page share a fit split."""
    excluded = climate_fever_pages(reg)

    def convert(row, item_id):
        if row["label"] == "NOT ENOUGH INFO":
            return "not enough info (retrieved, not gold, evidence)"
        label = FEVER_STANCE.get(row["label"])
        if not label or not row["evidence"]:
            return None
        by_page: dict[str, list[str]] = {}
        for page, _, sentence in row["evidence"]:
            by_page.setdefault(page, []).append(unwiki(sentence))
        keys = {page_key(p) for p in by_page}
        if keys & excluded:
            return "evidence page also a Climate-FEVER (dev) article"
        evidence = " ".join(f"{unwiki(p)}: {' '.join(ss)}" for p, ss in by_page.items())
        it = stance_item(item_id, row["claim"], evidence, label)
        if it:
            it.group = "page:" + min(keys)
        return it
    return convert


def vitaminc(row, item_id):
    it = stance_item(item_id, row["claim"], row["evidence"], FEVER_STANCE.get(row["label"]))
    if it and row.get("page"):
        it.group = "page:" + page_key(row["page"])
    return it


WANLI = {"entailment": "supports", "neutral": "no_bearing", "contradiction": "contradicts"}
SCINLI = {"entailment": "supports", "neutral": "no_bearing"}


def scinli(row, item_id):
    if row["label"] not in SCINLI:
        return "class not used (contrasting or reasoning)"
    return stance_item(item_id, row["sentence2"], row["sentence1"], SCINLI[row["label"]])


def climatecheck(row, item_id):
    """Confirm tier: each labelled claim x abstract pair; items sharing a claim form a group."""
    it = stance_item(item_id, row["claim"], row["abstract"], {"Supports": "supports", "Refutes": "contradicts",
                                                               "Not Enough Information": "no_bearing"}.get(row["annotation"]))
    if it:
        it.group = short("claim:", row["claim"])
    return it


STANCE: dict[str, Spec] = {
    "mnli": Spec(lambda r, i: stance_item(i, r["hypothesis"], r["premise"], NLI_STANCE.get(r["label"]))),
    "vitaminc": Spec(vitaminc),
    "fever": Spec(None, balance=True, setup=fever_setup),
    "snli": Spec(lambda r, i: stance_item(i, r["hypothesis"], r["premise"], NLI_STANCE.get(r["label"])), balance=True),
    "wanli": Spec(lambda r, i: stance_item(i, r["hypothesis"], r["premise"], WANLI.get(r["gold"])), balance=True),
    "scinli": Spec(scinli, balance=True),
    "anli": Spec(lambda r, i: stance_item(i, r["hypothesis"], r["premise"], NLI_STANCE.get(r["label"]))),
    "climate_fever": Spec(lambda r, i: stance_item(i, r["claim"], " ".join(e["evidence"] for e in r["evidences"]),
                                                   {0: "supports", 1: "contradicts", 2: "no_bearing"}.get(int(r["claim_label"])))),
    "climatecheck": Spec(climatecheck),
}


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def check_covid(info: dict) -> Iterator[Item]:
    """Check-COVID: each claim with the sentences of its CORD-19 abstract that the annotators
    marked as its evidence, in abstract order. Not the whole abstract: a not-enough-info item
    repeats a supported or refuted claim with another sentence of the same abstract, so the
    whole abstract would carry the evidence. The whole abstract is the item's context, so that
    an item is dropped when its abstract overlaps an earlier tier. Grouped by abstract."""
    d = registry.RAW / "check_covid"
    corpus = {x["cord_id"]: x for x in read_jsonl(d / "corpus.json")}
    labels = {"SUPPORT": "supports", "REFUTE": "contradicts", "NOTENOUGHINFO": "no_bearing"}
    for c in read_jsonl(d / "Check-COVID_all.json"):
        doc = corpus.get(c["cord_id"])
        index = sorted({e["sent_index"] for e in c.get("evidence_set") or []})
        if doc is None or not index or index[-1] >= len(doc["abstract"]):
            continue
        it = stance_item(f"check_covid-{c['id']}", c["claim"], " ".join(doc["abstract"][i] for i in index),
                         labels.get(c["label"]))
        if it:
            it.group = c["cord_id"]
            it.context = (doc.get("title") or "").strip() + ". " + " ".join(doc["abstract"])
            yield it


def healthfc(info: dict) -> Iterator[Item]:
    """HealthFC: the English question as the claim (verbatim: a yes/no question, which the
    evidence supports when the answer is yes), the English evidence sentences as evidence (never
    the explanation, which states the verdict)."""
    labels = {"0": "supports", "1": "no_bearing", "2": "contradicts"}
    with open(registry.RAW / "healthfc" / "Datensatz.csv", encoding="utf-8-sig", newline="") as f:
        for n, row in enumerate(csv.DictReader(f)):
            it = stance_item(f"healthfc-{n:04d}", row["en_claim"], row["en_top_sentences"], labels.get(row["label"].strip()))
            if it:
                yield it


RAW_LOADERS = {"check_covid": check_covid, "healthfc": healthfc}


def multivers_dir() -> Path:
    return registry.RAW / "multivers" / "data"


def multivers(name: str, splits: list[str]) -> Iterator[Item]:
    """Claim x abstract pairs: supports / contradicts where the abstract holds evidence for the
    claim, no bearing where the abstract is among the claim's documents without evidence."""
    corpus = {}
    for line in open(multivers_dir() / name / "corpus.jsonl"):
        d = json.loads(line)
        corpus[d["doc_id"]] = (d.get("title") or "") + ". " + " ".join(d["abstract"])
    for split in splits:
        path = multivers_dir() / name / f"claims_{split}.jsonl"
        if not path.exists():
            continue
        for line in open(path):
            c = json.loads(line)
            evidence = c.get("evidence") or {}
            for doc in c.get("doc_ids") or c.get("cited_doc_ids") or []:
                ev = evidence.get(str(doc))
                label = {"SUPPORT": "supports", "CONTRADICT": "contradicts"}[ev[0]["label"]] if ev else "no_bearing"
                if int(doc) in corpus:
                    it = stance_item(f"{name}-{split}-{c['id']}-{doc}", c["claim"], corpus[int(doc)], label)
                    if it:
                        yield it


def reserved_texts(reg: dict, fmt: str, held: bool = True) -> list[str]:
    """The claims and passages of every reserved source of a format, as the tiers are checked
    against them: for a MultiVerS source every claim and every abstract (with and without its
    title), for any other source the texts of every item it holds (every file of it). With
    `held`, also the sources reserved when the 0.1.0 tiers were built (`was_reserved`), which
    those tiers are checked against; without, the reserved sources of now (the confirm tier)."""
    return source_texts(reg, fmt, lambda name, src: registry.tier_of(reg, name) == "reserved"
                        or (held and src.get("was_reserved", False)))


def confirm_texts(reg: dict, fmt: str) -> list[str]:
    """The texts the confirm tier is checked against besides the other tiers: the reserved
    sources and, for stance, every claim and passage of every evaluation source as a whole (the
    tiers hold only a sample of them)."""
    tiers = ("reserved",) + (("dev", "final", "final-flagged", "final-seen") if fmt == "stance" else ())
    return source_texts(reg, fmt, lambda name, src: registry.tier_of(reg, name) in tiers and registry.used(reg, name))


def source_texts(reg: dict, fmt: str, chosen: Callable[[str, dict], bool]) -> list[str]:
    out: list[str] = []
    for name, src in reg["sources"].items():
        if src["format"] != fmt or not chosen(name, src):
            continue
        if src.get("raw") == "multivers":
            for line in open(multivers_dir() / name / "corpus.jsonl"):
                d = json.loads(line)
                out += [" ".join(d["abstract"]), (d.get("title") or "") + ". " + " ".join(d["abstract"])]
            for path in sorted((multivers_dir() / name).glob("claims_*.jsonl")):
                for line in open(path):
                    out.append(json.loads(line)["claim"])
            continue
        s = Source(name, fmt, "reserved", src["family"], src.get("type", "choice"),
                   (GENERAL if fmt == "general" else STANCE).get(name, Spec(None)))
        load_source(reg, s, keep_long=True, all_files=True)
        for it in s.items:
            out += list(item_texts(fmt, it))
    return out


def reserved_digests(reg: dict, fmt: str) -> set[str]:
    return {d for d in (digest(t) for t in reserved_texts(reg, fmt)) if d}


# ---- loading, cleaning, sampling -----------------------------------------------------------

@dataclass
class Source:
    name: str
    fmt: str
    tier: str
    family: str
    type: str
    spec: Spec
    items: list[Item] = field(default_factory=list)
    drops: dict[str, int] = field(default_factory=dict)

    def drop(self, why: str, n: int = 1) -> None:
        self.drops[why] = self.drops.get(why, 0) + n


def too_long(fmt: str, it: Item) -> bool:
    size = len(it.state) + len(it.question or "") + sum(len(t) for _, t in it.options) + len(it.claim) + len(it.evidence)
    return size + (200 if fmt == "stance" else 0) > MAX_CHARS[fmt]


def read_rows(path: Path, columns: list[str] | None = None) -> list[dict]:
    """The rows of one data file, by its extension: Parquet, JSON Lines (optionally gzipped),
    JSON (a list of rows) or CSV."""
    name = path.name
    if name.endswith(".parquet"):
        import pyarrow.parquet as pq
        return pq.read_table(path, columns=columns).to_pylist()
    if name.endswith(".csv"):
        with open(path, encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    opener = gzip.open if name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        if name.endswith(".json"):
            data = json.load(f)
            return data if isinstance(data, list) else data["data"]
        return [json.loads(line) for line in f if line.strip()]


def class_names(path: Path, column: str) -> list[str]:
    """The class names of a ClassLabel column, from a Parquet file's Hugging Face metadata."""
    import pyarrow.parquet as pq
    meta = json.loads(pq.read_schema(path).metadata[b"huggingface"])
    return meta["info"]["features"][column]["names"]


def hub_files(info: dict) -> Iterator[tuple[str, Path]]:
    """The data files of a source pinned by `files` (or `file_pattern` and `configs`), each
    downloaded at the pinned revision (and checked against its SHA-256 where one is recorded)."""
    from huggingface_hub import hf_hub_download
    files = info.get("files") or {c: info["file_pattern"].format(config=c) for c in info["configs"]}
    for tag, f in files.items():
        name, sha = (f["path"], f.get("sha256")) if isinstance(f, dict) else (f, None)
        path = Path(hf_hub_download(info["hub"], name, repo_type="dataset", revision=info["revision"]))
        if sha and registry.sha256(path) != sha:
            raise SystemExit(f"prep_tiers: {info['hub']}/{name} does not have the registered SHA-256")
        yield tag, path


def add(src: Source, it) -> None:
    if it is None:
        src.drop("no usable label or fields")
    elif isinstance(it, str):
        src.drop(it)
    else:
        src.items.append(it)


def load_source(reg: dict, src: Source, pool_hint: int = 0, keep_long: bool = False, all_files: bool = False) -> None:
    """The items of a source; of its `files`, only those named in `item_files` (if given)
    unless `all_files`."""
    info = reg["sources"][src.name]
    spec = src.spec
    if src.name == "generated":
        src.items = list(generated(pool_hint * 3))
        return
    if info.get("raw") == "multivers":
        src.items = list(multivers(src.name, info["splits"]))
    elif info.get("raw"):
        src.items = list(RAW_LOADERS[info["raw"]](info))
    elif info.get("files") or info.get("file_pattern"):
        convert = spec.setup(reg) if spec.setup else spec.convert
        only = None if all_files else info.get("item_files")
        for tag, path in hub_files(info | ({"files": {t: info["files"][t] for t in only}} if only else {})):
            if spec.needs_names:
                convert = spec.convert(class_names(path, spec.needs_names))
            for n, row in enumerate(read_rows(path, info.get("columns"))):
                add(src, convert(row | {"_tag": tag}, f"{src.name}-{tag}-{n:07d}"))
    else:
        from datasets import load_dataset
        for split in info["splits"]:
            data = load_dataset(info["hub"], info.get("config"), split=split, revision=info["revision"])
            convert = spec.setup(reg) if spec.setup else spec.convert
            if spec.needs_names:
                feature = data.features[spec.needs_names]
                feature = getattr(feature, "feature", feature)     # a sequence of class labels
                if hasattr(feature, "names"):
                    names = feature.names
                else:
                    col = data[spec.needs_names]
                    names = [""] * (max(col) + 1)
                    for value, text in zip(col, data[spec.needs_names + "_text"]):
                        names[value] = text
                convert = convert(names)
            for n, row in enumerate(data):
                add(src, convert(row, f"{src.name}-{split}-{n:07d}"))
    kept = []
    for it in src.items:
        if too_long(src.fmt, it) and not keep_long:
            src.drop("too long")
        elif src.fmt == "general" and it.state is None:
            src.drop("no usable label or fields")
        else:
            kept.append(it)
    src.items = kept


def item_texts(fmt: str, it: Item) -> tuple[str, ...]:
    if fmt == "stance":
        return (it.claim, it.evidence, it.context)
    return (it.state, it.question or "", passage_of(it.question or ""))


def item_digests(fmt: str, it: Item) -> set[str]:
    return {d for d in (digest(t) for t in item_texts(fmt, it)) if d}


def item_segments(fmt: str, it: Item) -> set[str]:
    out: set[str] = set()
    for t in item_texts(fmt, it):
        out |= segment_digests(t)
    return out


def dedup_key(fmt: str, it: Item) -> str:
    if fmt == "stance":
        return norm(it.claim) + "\x00" + norm(it.evidence)
    return norm(it.state) + "\x00" + norm(it.question or "") + "\x00" + "\x00".join(norm(t) for _, t in it.options)


def group_key(fmt: str, it: Item) -> str:
    """Items with the same passage (evidence, or state), or the same recorded group, go to the
    same fit split."""
    if it.group:
        return "group:" + it.group
    if fmt == "stance":
        return digest(it.evidence) or it.id
    return digest(it.state) or digest(passage_of(it.question or "")) or digest(it.question or "") or it.id


def clean(src: Source, against: set[str], seen: set[str]) -> None:
    kept = []
    for it in sorted(src.items, key=lambda x: unit(x.id)):
        if item_digests(src.fmt, it) & against:
            src.drop("overlaps an earlier tier")
            continue
        key = dedup_key(src.fmt, it)
        if key in seen:
            src.drop("duplicate in tier")
            continue
        seen.add(key)
        kept.append(it)
    src.items = kept


class Strict:
    """The sentence-level, passage-level and near-duplicate check of one tier against earlier
    ones."""

    def __init__(self, fmt: str, segments: set[str], near: list[NearIndex], shingles: ShingleIndex):
        self.fmt, self.segments, self.near, self.shingles = fmt, segments, near, shingles

    def __call__(self, it: Item) -> str | None:
        texts = [t for t in item_texts(self.fmt, it) if t]
        if any(segment_digests(t) & self.segments for t in texts):
            return "shares a sentence with an earlier tier"
        if any(self.shingles.shared(t) for t in texts):
            return "shares a passage with an earlier tier"
        if any(index.near(t) for index in self.near for t in texts):
            return "near duplicate of an earlier tier"
        return None


def strict_for(fmt: str, tiers: tuple[str, ...], segments: dict, near: dict, shingles: dict) -> Strict:
    joined = ShingleIndex()
    for t in tiers:
        if t in shingles:
            joined.update(shingles[t])
    return Strict(fmt, set().union(*(segments.get(t, set()) for t in tiers)), [near[t] for t in tiers if t in near],
                  joined)


def sample(src: Source, pool: int | None, bad: Callable[[Item], str | None] | None = None) -> None:
    """The first `pool` items in hash order (already sorted by clean), or evenly across labels
    (or strata) in turns; with `bad`, an item it names a reason for is dropped and the next one
    taken; with Spec.one_per_group, at most one item per group. None: every item."""
    taken_groups: set[str] = set()

    def ok(it: Item) -> bool:
        if src.spec.one_per_group and it.group:
            if it.group in taken_groups:
                src.drop("another item of its group already taken")
                return False
        why = bad(it) if bad else None
        if why:
            src.drop(why)
            return False
        if src.spec.one_per_group and it.group:
            taken_groups.add(it.group)
        return True

    pool = len(src.items) if pool is None else pool
    if not src.spec.balance:
        out = []
        for it in src.items:
            if len(out) >= pool:
                break
            if ok(it):
                out.append(it)
        src.items = out
        return
    by_label: dict[str, list[Item]] = {}
    for it in src.items:
        by_label.setdefault(it.stratum or it.label, []).append(it)
    queues = [v for _, v in sorted(by_label.items())]
    pos = [0] * len(queues)
    out = []
    while len(out) < pool and any(p < len(q) for p, q in zip(pos, queues)):
        for k, q in enumerate(queues):
            if len(out) >= pool:
                break
            while pos[k] < len(q):
                it = q[pos[k]]
                pos[k] += 1
                if ok(it):
                    out.append(it)
                    break
    src.items = sorted(out, key=lambda x: unit(x.id))


def confirm_check(src: Source, bad: Callable[[Item], str | None], earlier: Strict) -> Callable[[Item], str | None]:
    """The confirm tier's check of one item: the strict check against every other tier, the
    same check against the confirm sources built before this one, and, for a `distinct` source,
    no near duplicate of an item of it already taken (taken items are indexed as they pass)."""
    taken = NearIndex()

    def check(it: Item) -> str | None:
        why = bad(it)
        if why:
            return why
        if earlier(it):
            return "matches an item of another confirm family"
        texts = [t for t in item_texts(src.fmt, it) if t]
        if src.spec.distinct:
            if any(taken.near(t) for t in texts):
                return "near duplicate of an item of its source already taken"
            for k, t in enumerate(texts):
                taken.add(f"{it.id}#{k}", t)
        return None
    return check


def fit_split(fmt: str, it: Item) -> str:
    u = unit("split:" + group_key(fmt, it))
    return "train" if u < 0.70 else "validation" if u < 0.85 else "test"


def quota(sources: list[Source], split: str, cap: int) -> dict[str, int]:
    """Items of `split` each source keeps: the same number from every source in turn, as far as
    each source's supply allows, up to `cap` in total."""
    supply = {s.name: sum(fit_split(s.fmt, it) == split for it in s.items) for s in sources}
    kept = {n: 0 for n in supply}
    left = cap
    while left > 0 and any(kept[n] < supply[n] for n in supply):
        for n in sorted(supply):
            if left > 0 and kept[n] < supply[n]:
                kept[n] += 1
                left -= 1
    return kept


# ---- bench: external benchmarks, scored as published -------------------------------------

def typed_question(q: dict, expected: str, dist: dict | None) -> dict | str:
    """One typed question (the request format shared by typed-decisions and JevBench) as a
    judgly question with its gold distribution over judgly's options, in their order. The
    per-option descriptions stay in: for choice as the options, for bool and score written into
    the instructions (judgly's bool and score have no description field)."""
    kind, crit, text = q["type"], q.get("criteria"), q["instructions"].strip()
    if kind == "choice":
        keys = list(crit or {})
        if expected not in keys or not 2 <= len(keys) <= 26:
            return "the label names no option, or too many options"
        gold = [float(dist.get(k, 0.0)) for k in keys] if dist else [float(k == expected) for k in keys]
        row = {"type": "choice", "instructions": text, "options": {k: crit[k] for k in keys}, "label": expected}
    elif kind == "noul":
        truth = {"true": True, "yes": True, "false": False, "no": False}.get(str(expected))
        if truth is None:
            return "the label names no option"
        p_true = float(dist.get("true", dist.get("yes"))) if dist else float(truth)
        if crit:          # a bare statement has no criteria
            text = f"{text} Answer true if: {crit['true'].strip()} Answer false if: {crit['false'].strip()}"
        row = {"type": "bool", "label": "true" if truth else "false", "instructions": text}
        gold = [p_true, 1.0 - p_true]
    elif kind == "score":
        k = len(crit or [])
        if not 2 <= k <= 9 or not str(expected).isdigit() or not 0 <= int(expected) < k:
            return "the label names no level, or too many levels"
        levels = "; ".join(f"{i + 1} = {c.strip().rstrip('.')}" for i, c in enumerate(crit))
        row = {"type": "score", "instructions": f"{text} Levels: {levels}.", "levels": k, "label": int(expected) + 1}
        gold = [float(dist.get(str(i), 0.0)) for i in range(k)] if dist else [float(i == int(expected)) for i in range(k)]
    else:
        return f"question type {kind} not supported"
    total = sum(gold)
    return row | {"gold": [round(g / total, 6) for g in gold]}


def bench_row(src: Source, item_id: str, task: str, state, q: dict, expected: str, dist: dict | None,
              group: str) -> Item | str:
    rendered = typed_question(q, expected, dist)
    if isinstance(rendered, str):
        return rendered
    state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    size = len(state) + len(rendered["instructions"]) + sum(len(k) + len(v) for k, v in rendered.get("options", {}).items())
    if size > MAX_CHARS["bench"]:
        return "too long"
    row = {"id": item_id, "source": src.name, "task": task, "family": src.family, "format": "general",
           "type": rendered.pop("type"), "split": "heldout", "state": state,
           "instructions": rendered.pop("instructions"), "source_question": None, **rendered, "group": group}
    return Item(item_id, state, None, [], str(row["label"]), task=task, type=row["type"], group=group, row=row)


def typed_decisions(reg: dict, src: Source) -> None:
    """LocalLLaMA/typed-decisions: five questions per case, each against the teacher's soft
    gold (its `label` is the argmax)."""
    info = reg["sources"][src.name]
    for _, path in hub_files(info):
        for case in read_rows(path):
            questions, gold = json.loads(case["questions"]), json.loads(case["gold"])
            for qname, q in questions.items():
                g = gold[qname]
                add(src, bench_row(src, f"{src.name}-{case['id']}-{qname}", f"td_{case['workflow']}", case["state"], q,
                                   g["label"], g.get("probabilities"), case["id"]))


def jevbench(reg: dict, src: Source) -> None:
    """JevBench public decisions: one question per state; the gold is the expected label
    (or, for the probability family, the published gold probabilities)."""
    for part in ("original", "easy", "hard"):
        for d in read_jsonl(registry.RAW / "jevbench" / f"{part}.jsonl"):
            if d.get("expected") is None:
                src.drop("no expected answer (not scorable)")
                continue
            dist = d.get("provenance", {}).get("gold_probs")
            add(src, bench_row(src, f"{src.name}-{d['id']}", f"jb_{d['family']}", d["state"], d["question"],
                               d["expected"], dist, d.get("group") or d["id"]))


BENCH_LOADERS = {"typed_decisions": typed_decisions, "jevbench": jevbench}


# ---- rendering ---------------------------------------------------------------------------

def render_general(src: Source, it: Item, split: str, pool: list[Item]) -> dict:
    rng = random.Random(f"{SEED}:render:{it.id}")
    typ = it.type or src.type
    wording = rng.choice(it.wordings or src.spec.wordings)
    ex = {"id": it.id, "source": src.name, "task": it.task or src.name, "family": src.family, "format": "general",
          "type": typ, "split": split, "state": it.state,
          "instructions": wording.replace("{q}", it.question or "").strip(), "source_question": it.question}
    if typ == "bool":
        return ex | {"label": it.label}
    if typ == "score":
        return ex | {"levels": it.levels or src.spec.levels, "label": int(it.label)}
    options, label, augmented = list(it.options), it.label, []
    if split in ("train", "validation"):
        if src.spec.open_options and rng.random() < P_DISTRACTOR:
            other = rng.choice(pool)
            if other.id != it.id and other.options:
                text = rng.choice(other.options)[1]
                if text not in {t for _, t in options}:
                    options.append(("distractor", text))
                    augmented.append("distractor")
        if rng.random() < P_NONE and len(options) < 26:
            if rng.random() < 0.5 and len(options) > 2:
                options = [o for o in options if o[0] != label]
                label = NONE_KEY
            options.append((NONE_KEY, NONE_TEXT))
            augmented.append("none")
    rng.shuffle(options)
    return ex | {"options": dict(options), "label": label, "augmented": augmented}


def render_stance(src: Source, it: Item, split: str, pool: list[Item]) -> dict:
    rng = random.Random(f"{SEED}:{it.id}")
    options = list(zip(STANCE_KEYS, rng.choice(STANCE_OPTIONS)))
    rng.shuffle(options)
    return {"id": it.id, "source": src.name, "task": src.name, "family": src.family, "format": "stance",
            "type": "choice", "split": split, "state": f"Claim: {it.claim}\n\nEvidence: {it.evidence}",
            "instructions": rng.choice(STANCE_QUESTIONS), "options": dict(options), "label": it.label, "augmented": []}


def eval_group(src: Source, it: Item) -> str:
    """The resampling group of an evaluation item: items that share a claim, an abstract, a
    table, a question or a template are not independent (Item.cluster, else Item.group)."""
    return f"{src.name}:{it.cluster or it.group or it.id}"


# ---- one format --------------------------------------------------------------------------

def pool_for(reg: dict, src: Source, cfg: dict) -> int | None:
    """The source's `pool` from the registry, else the tier's; in a --quick build never more
    than the tier's quick size."""
    default = cfg[POOL_KEY[src.tier]]
    own = reg["sources"][src.name].get("pool")
    if own is None:
        return default
    return min(own, default) if cfg.get("quick") else own


def build_format(reg: dict, fmt: str, cfg: dict, out: Path) -> dict:
    specs = GENERAL if fmt == "general" else STANCE
    names = [n for n, s in reg["sources"].items() if s["format"] == fmt and registry.used(reg, n)
             and registry.tier_of(reg, n) != "reserved"]
    for n in names:
        tier = registry.tier_of(reg, n)
        if tier == "fit":
            ok, why = registry.fit_allowed(reg, n)
            if not ok:
                raise SystemExit(f"prep_tiers: {n} is in the fit tier but {why}; the policy forbids fitting on it")
        info = reg["sources"][n]
        if (n not in specs and n != "generated" and info.get("raw") not in ("multivers", *RAW_LOADERS)
                and n not in BENCH_LOADERS):
            raise SystemExit(f"prep_tiers: no converter for {n}")
        if tier == "bench" and n not in BENCH_LOADERS:
            raise SystemExit(f"prep_tiers: {n} is in the bench tier but has no bench loader")
    sources = {n: Source(n, fmt, registry.tier_of(reg, n), reg["sources"][n]["family"],
                         reg["sources"][n].get("type", "choice"), specs.get(n, Spec(None)))
               for n in names}

    texts = reserved_texts(reg, fmt)
    against = {d for d in (digest(t) for t in texts) if d}
    n_reserved = len(against)
    segments: dict[str, set[str]] = {"reserved": set()}
    near: dict[str, NearIndex] = {"reserved": NearIndex()}
    shingles: dict[str, ShingleIndex] = {"reserved": ShingleIndex()}
    for i, t in enumerate(texts):
        segments["reserved"] |= segment_digests(t)
        near["reserved"].add(f"reserved-{i}", t)
        shingles["reserved"].add(f"reserved-{i}", t)
    tier_digests: set[str] = set()          # the digests of every tier's items (not the reserved texts)
    for tier in BUILD_ORDER:
        seen: set[str] = set()
        tier_sources = [s for s in sources.values() if s.tier == tier]
        exact = against
        if tier == "confirm" and tier_sources:
            extra = confirm_texts(reg, fmt)
            exact = tier_digests | {d for d in (digest(t) for t in extra) if d}
            segments["sources"], near["sources"], shingles["sources"] = set(), NearIndex(), ShingleIndex()
            for i, t in enumerate(extra):
                segments["sources"] |= segment_digests(t)
                near["sources"].add(f"sources-{i}", t)
                shingles["sources"].add(f"sources-{i}", t)
            print(f"{fmt} confirm: checked against {len(extra)} texts of reserved and evaluation sources", file=sys.stderr)
        strict = STRICT.get(tier)
        bad = strict_for(fmt, strict, segments, near, shingles) if strict else None
        earlier = Strict(fmt, set(), [NearIndex()], ShingleIndex())     # confirm: the confirm sources built so far
        for s in tier_sources:
            if tier == "bench":
                BENCH_LOADERS[s.name](reg, s)
                s.items.sort(key=lambda x: unit(x.id))
                if cfg[POOL_KEY[tier]] is not None:
                    s.items = s.items[:cfg[POOL_KEY[tier]]]
            else:
                load_source(reg, s, pool_for(reg, s, cfg) or 0)
                clean(s, exact, seen)
                sample(s, pool_for(reg, s, cfg), confirm_check(s, bad, earlier) if tier == "confirm" else bad)
            print(f"{fmt} {tier} {s.name}: {len(s.items)} items; dropped {s.drops}", file=sys.stderr)
            if tier == "confirm":
                for it in s.items:
                    earlier.segments |= item_segments(fmt, it)
                    for k, t in enumerate(item_texts(fmt, it)):
                        if t:
                            earlier.near[0].add(f"{it.id}#{k}", t)
                            earlier.shingles.add(f"{it.id}#{k}", t)
        segments[tier], near[tier], shingles[tier] = set(), NearIndex(), ShingleIndex()
        for s in tier_sources:
            for it in s.items:
                tier_digests |= item_digests(fmt, it)
                against |= item_digests(fmt, it)
                segments[tier] |= item_segments(fmt, it)
                for k, t in enumerate(item_texts(fmt, it)):
                    if t:
                        near[tier].add(f"{it.id}#{k}", t)
                        shingles[tier].add(f"{it.id}#{k}", t)

    render = render_general if fmt == "general" else render_stance
    fit = [s for s in sources.values() if s.tier == "fit"]
    keep = {split: quota(fit, split, cfg["caps"][split]) for split in ("train", "validation", "test")}
    rows: dict[str, list[dict]] = {}
    counts: dict[str, dict[str, int]] = {}
    labels: dict[str, dict[str, int]] = {}
    for s in sources.values():
        for it in s.items:
            if s.tier == "fit":
                split = fit_split(fmt, it)
                if keep[split][s.name] == 0:
                    continue
                keep[split][s.name] -= 1
            else:
                split = "heldout"
            ex = it.row if it.row is not None else render(s, it, split, s.items)
            if s.tier in GROUPED:
                ex["group"] = eval_group(s, it)
            rows.setdefault(OUT_FILE[s.tier], []).append(ex)
            tier_split = split if s.tier == "fit" else s.tier
            counts.setdefault(ex["task"], {}).setdefault(tier_split, 0)
            counts[ex["task"]][tier_split] += 1
            labels.setdefault(ex["task"], {}).setdefault(str(ex["label"]), 0)
            labels[ex["task"]][str(ex["label"])] += 1
    (out / fmt).mkdir(parents=True, exist_ok=True)
    files = {}
    for name, exs in sorted(rows.items()):
        exs.sort(key=lambda e: (SPLIT_ORDER[e["split"]], e["source"], unit(e["id"])))
        path = out / fmt / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for e in exs:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        files[f"{fmt}/{name}.jsonl"] = {"sha256": registry.sha256(path), "items": len(exs)}
    return {"settings": cfg, "reserved_texts": n_reserved, "counts": counts, "labels": labels,
            "drops": {n: s.drops for n, s in sources.items()},
            "tiers": {t: sorted(n for n, s in sources.items() if s.tier == t) for t in registry.EVAL_TIERS + ("fit",)},
            "files": files}


def ensure_raw(reg: dict) -> None:
    """Every raw file present with its registered SHA-256 (make fetch), and the MultiVerS
    release unpacked."""
    for name, raw in reg["raw"].items():
        files = raw.get("files") or {Path(raw["url"]).name: {"sha256": raw["sha256"]}}
        for fname, f in files.items():
            path = registry.RAW / name / fname
            if not path.exists():
                raise SystemExit(f"prep_tiers: {path} missing; run `make fetch` first")
            if registry.sha256(path) != f["sha256"]:
                raise SystemExit(f"prep_tiers: {path} does not have the registered SHA-256")
    if not multivers_dir().exists():
        with tarfile.open(registry.RAW / "multivers" / "data.tar.gz") as t:
            t.extractall(registry.RAW / "multivers", filter="data")


def main(out: Path, quick: bool) -> None:
    import datasets
    reg = registry.load()
    problems = registry.check_policy(reg)
    if problems:
        raise SystemExit("prep_tiers: licence policy problems:\n  " + "\n  ".join(problems))
    ensure_raw(reg)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"built": datetime.date.today().isoformat(), "quick": quick, "seed": SEED,
                "registry_sha256": registry.registry_sha256(), "max_chars": MAX_CHARS,
                "python": platform.python_version(), "datasets": datasets.__version__,
                "raw_sha256": {f"{n}/{f}": v["sha256"] for n, r in reg["raw"].items()
                               for f, v in (r.get("files") or {Path(r["url"]).name: r}).items()},
                "formats": {}}
    for fmt, cfg in (QUICK if quick else FULL).items():
        manifest["formats"][fmt] = build_format(reg, fmt, cfg | {"quick": quick}, out)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    for fmt, m in manifest["formats"].items():
        print(f"{fmt}: " + ", ".join(f"{k} {v['items']} items" for k, v in m["files"].items()), file=sys.stderr)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) not in (1, 2) or args[1:] not in ([], ["--quick"]):
        sys.exit(__doc__)
    main(Path(args[0]), len(args) == 2)
