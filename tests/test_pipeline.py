"""The pipeline's guards: licence policy, contamination checker, split grouping and the bench
mapping. No model: the checker runs on small hand-made tier files with planted leaks (the
stance reserved sources are read from the pinned releases, so the stance checks need them)."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("yaml")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import calibration_record  # noqa: E402
import check_contamination  # noqa: E402
import prep_tiers  # noqa: E402
import registry  # noqa: E402
import texthash  # noqa: E402

LONG = "a passage long enough to be hashed by the contamination checker, number {}"


def text(key: str) -> str:
    """A text of its own for every key: no two of them are near duplicates."""
    return " ".join(hashlib.sha256(f"{key}:{k}".encode()).hexdigest()[:8] for k in range(9)) + "."


def ex(i, source, family, split, state=None, question=None):
    return {"id": f"{source}-{i}", "source": source, "task": source, "family": family, "format": "general",
            "type": "bool", "split": split, "state": state or text(f"{source}-{i}"),
            "instructions": "x", "source_question": question, "label": "true"}


def write(tmp: Path, fitdev, final, seen=(), bench=(), flagged=()):
    (tmp / "general").mkdir(parents=True, exist_ok=True)
    for name, rows in (("fitdev", fitdev), ("final", final), ("final-seen", seen), ("bench", bench),
                       ("final-flagged", flagged)):
        (tmp / "general" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return check_contamination.check_format(registry.load(), tmp, "general")


def clean_tiers():
    fitdev = [ex(i, "mmlu", "knowledge", s) for i, s in enumerate(["train", "validation", "test"] * 3)]
    fitdev += [ex(i, "sst2", "sentiment", "heldout") for i in range(3)]
    return fitdev, [ex(i, "fig_qa", "figurative", "heldout") for i in range(3)]


def seen_tier():
    return [ex(i, "ag_news", "topic", "heldout") for i in range(3)]


def test_clean_tiers_pass(tmp_path):
    assert write(tmp_path, *clean_tiers()) == 0
    assert write(tmp_path, *clean_tiers(), seen=seen_tier(), bench=[ex(0, "typed_decisions", "bench_typed_decisions",
                                                                         "heldout")]) == 0


def test_final_seen_item_in_fit_fails(tmp_path):
    fitdev, final = clean_tiers()
    seen = seen_tier()
    fitdev.append(ex(95, "mmlu", "knowledge", "train", state=seen[0]["state"]))
    assert write(tmp_path, fitdev, final, seen) > 0


def test_near_duplicate_of_final_fails(tmp_path):
    fitdev, final = clean_tiers()
    words = final[0]["state"].split()
    fitdev.append(ex(94, "mmlu", "knowledge", "train", state=" ".join(words[:-1] + ["changed", "word."])))
    assert write(tmp_path, fitdev, final) > 0


def test_final_sharing_a_sentence_with_dev_fails(tmp_path):
    fitdev, final = clean_tiers()
    final[0]["state"] = fitdev[-1]["state"] + " " + text("more") + " " + text("and more")
    assert write(tmp_path, fitdev, final) > 0


def test_bench_near_duplicate_of_fit_fails_but_equal_to_final_seen_passes(tmp_path):
    fitdev, final = clean_tiers()
    seen = seen_tier()
    bench = [ex(0, "typed_decisions", "bench_typed_decisions", "heldout", state=seen[0]["state"])]
    assert write(tmp_path, fitdev, final, seen, bench) == 0      # evaluation tiers may share: reported
    bench.append(ex(1, "typed_decisions", "bench_typed_decisions", "heldout", state=fitdev[0]["state"] + " x"))
    assert write(tmp_path, fitdev, final, seen, bench) > 0


def test_final_sharing_a_passage_with_dev_fails(tmp_path):
    """A reworded copy of a passage: no sentence in common and too few shared words overall
    for a near duplicate, but a long run of the same words."""
    fitdev, final = clean_tiers()
    run = "we collected blood from patients who have recently become virus free and were discharged home"
    fitdev[-1]["state"] = "In this study, " + run + ", and measured their antibody responses over the following months."
    final[0]["state"] = ("Here " + run + " early; then detected specific humoral and cellular immunity in them. " +
                         text("filler") + " " + text("more filler"))
    assert write(tmp_path, fitdev, final) > 0


def test_flagged_tier_is_checked_like_final(tmp_path):
    fitdev, final = clean_tiers()
    flagged = [ex(i, "politeness", "politeness", "heldout") for i in range(3)]
    assert write(tmp_path, fitdev, final, flagged=flagged) == 0
    flagged.append(ex(9, "politeness", "politeness", "heldout", state=final[0]["state"]))   # equal to final
    assert write(tmp_path, fitdev, final, flagged=flagged) > 0
    final.append(ex(8, "politeness", "politeness", "heldout"))      # a final-flagged source in final
    assert write(tmp_path, fitdev, final) > 0


def test_confirm_tier_is_checked_against_every_other_tier(tmp_path):
    fitdev, final = clean_tiers()
    confirm = [ex(i, "clutrr", "kinship", "heldout") for i in range(3)]
    (tmp_path / "general").mkdir(parents=True, exist_ok=True)
    path = tmp_path / "general" / "confirm.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in confirm))
    assert write(tmp_path, fitdev, final, seen=seen_tier()) == 0
    for leak in (final[0]["state"], seen_tier()[0]["state"], fitdev[0]["state"] + " x"):
        path.write_text("".join(json.dumps(r) + "\n" for r in confirm + [ex(9, "clutrr", "kinship", "heldout", state=leak)]))
        assert write(tmp_path, fitdev, final, seen=seen_tier()) > 0


def test_confirm_passage_check_counts_boilerplate():
    quote = "global average air and ocean temperatures widespread melting of snow and ice and rising sea level"
    idx = texthash.ShingleIndex()
    for i in range(5):                                  # a quotation held by more than SHINGLE_DF texts
        idx.add(f"t{i}", f"Text {i} about something else entirely. {quote}. And more words number {i}.")
    probe = f"A new abstract that quotes {quote} in its opening."
    assert not idx.shared(probe)                        # the other tiers' rule: boilerplate
    assert idx.shared(probe, None)                      # the confirm tier's rule: a shared passage


def test_linked_clusters_join_claims_that_share_an_abstract():
    def item(i, claim, abstract):
        it = prep_tiers.stance_item(f"c-{i}", claim, abstract, "supports")
        it.group = prep_tiers.short("claim:", claim)
        return it
    items = [item(0, "claim A", "abstract 1"), item(1, "claim B", "abstract 1"), item(2, "claim B", "abstract 2"),
             item(3, "claim C", "abstract 2"), item(4, "claim D", "abstract 3")]
    prep_tiers.link_clusters(items)
    assert len({it.cluster for it in items[:4]}) == 1 and items[4].cluster != items[0].cluster


def test_flagged_sources_record_a_caveat():
    reg = registry.load()
    flagged = [n for n in reg["sources"] if registry.tier_of(reg, n) == "final-flagged"]
    assert flagged and all(reg["sources"][n].get("caveat") for n in flagged)
    assert not any(reg["sources"][n].get("caveat") for n in reg["sources"] if n not in flagged)


def test_title_prefix_and_treebank_tokens_do_not_hide_a_sentence():
    sentence = "The Wire premiered on June 2, 2002, and ended on March 9, 2008, after five seasons on HBO."
    fever = "The Wire: " + sentence.replace(",", " ,").replace("HBO", "HBO -LRB- cable -RRB-")
    wiki = "Other text first. " + sentence.replace("HBO", "HBO (cable)") + " And more after."
    assert texthash.segment_digests(fever) & texthash.segment_digests(wiki)


def test_shingle_index_ignores_boilerplate():
    boiler = "caused by the novel severe acute respiratory syndrome coronavirus 2 sars cov 2"
    idx = texthash.ShingleIndex()
    for i in range(texthash.SHINGLE_DF + 1):
        idx.add(f"b{i}", f"Study {text(str(i))} of disease {boiler} in cohort {text(str(i) + 'x')}")
    assert idx.shared(f"A new report on {boiler} and nothing else") == []
    idx.add("own", "we collected blood from patients who have recently become virus free and were discharged")
    hits = idx.shared("Here we collected blood from patients who have recently become virus free and were sent home")
    assert [k for k, _ in hits] == ["own"] and all(n >= texthash.SHINGLE_MIN for _, n in hits)


def test_fitted_on_names_the_licence_the_source_is_used_under():
    reg = registry.load()
    assert calibration_record.licence_used(reg["sources"]["scinli"]) == ["cc-by-sa-4.0"]
    assert calibration_record.licence_used(reg["sources"]["fever"]) == ["cc-by-sa-3.0"]
    assert calibration_record.licence_used(reg["sources"]["qasc"]) == ["cc-by-4.0"]


def test_group_bootstrap_resamples_whole_groups():
    import numpy as np
    rows = [{"id_hash": i, "task_id": 0, "family_id": 0, "type": 0, "label": 0,
             "p": np.array([0.9, 0.1]) if i < 50 else np.array([0.1, 0.9])} for i in range(100)]
    x = calibration_record.per_item(rows)
    idx = np.arange(100)
    items = calibration_record.with_interval(x, idx, np.random.default_rng(0))
    two = calibration_record.with_interval(x, idx, np.random.default_rng(0), ["a" if i < 50 else "b" for i in idx])
    assert items["accuracy"]["value"] == two["accuracy"]["value"] == 0.5
    width = lambda m: m["accuracy"]["hi"] - m["accuracy"]["lo"]           # noqa: E731
    assert width(two) > 2 * width(items)                               # two groups: far wider


def test_near_index_meets_its_threshold():
    idx = texthash.NearIndex()
    idx.add("a", "vitamin C can prevent or treat COVID-19 (coronavirus)")
    hits = idx.near("Can vitamin C prevent infection with coronavirus or prevent deaths from covid-19?")
    assert [k for k, _ in hits] == ["a"] and all(j >= texthash.NEAR_JACCARD for _, j in hits)
    assert idx.near("Masks reduce the spread of respiratory infections in schools and offices") == []


def test_final_item_in_fit_fails(tmp_path):
    fitdev, final = clean_tiers()
    fitdev.append(ex(99, "mmlu", "knowledge", "train", state=final[0]["state"].upper() + "!!"))
    assert write(tmp_path, fitdev, final) > 0


def test_fit_question_in_dev_fails(tmp_path):
    fitdev, final = clean_tiers()
    q = "Which of these planets has the longest day of all planets in the solar system?"
    fitdev[0]["source_question"] = q
    fitdev.append(ex(98, "sst2", "sentiment", "heldout", question=q))
    assert write(tmp_path, fitdev, final) > 0


def test_shared_passage_with_other_question_fails(tmp_path):
    fitdev, final = clean_tiers()
    passage = ("This question refers to the following information.\n\"" + LONG.format("quoted") +
               ", and a little more text so that the passage is clearly long enough.\"")
    fitdev[0]["source_question"] = passage + "\nWhich of the following best describes the author?"
    final[0]["source_question"] = passage + "\nThe passage above was most likely written in response to"
    assert write(tmp_path, fitdev, final) > 0


def test_eval_only_source_in_fit_fails(tmp_path):
    fitdev, final = clean_tiers()
    fitdev.append(ex(97, "sst2", "sentiment", "train"))     # sst2: licence unknown, dev tier
    assert write(tmp_path, fitdev, final) > 0


def test_family_in_two_tiers_fails(tmp_path):
    fitdev, final = clean_tiers()
    final.append(ex(96, "ag_news", "sentiment", "heldout"))
    assert write(tmp_path, fitdev, final) > 0


def test_final_seen_source_in_final_fails(tmp_path):
    fitdev, final = clean_tiers()
    final.append(ex(93, "ag_news", "topic", "heldout"))      # ag_news: final-seen, not final
    assert write(tmp_path, fitdev, final) > 0


def test_short_texts_are_not_compared(tmp_path):
    fitdev, final = clean_tiers()
    fitdev[0]["source_question"] = final[0]["source_question"] = "Thank you!"
    assert write(tmp_path, fitdev, final) == 0


def test_registry_policy_is_clean_and_catches_violations():
    reg = registry.load()
    assert registry.check_policy(reg) == []
    bad = copy.deepcopy(reg)
    bad["sources"]["qasc"]["licence"] = ["cc-by-nc-4.0"]
    assert any("qasc" in p for p in registry.check_policy(bad))
    bad = copy.deepcopy(reg)
    bad["sources"]["vitaminc"]["format"] = "general"       # share-alike: stance only
    bad["sources"]["vitaminc"]["family"] = "knowledge"
    assert any("vitaminc" in p for p in registry.check_policy(bad))


def test_licence_decision_must_name_an_allowed_licence():
    reg = registry.load()
    assert registry.fit_allowed(reg, "mnli")[0]
    bad = copy.deepcopy(reg)                                # a decision that does not say what it allows
    bad["sources"]["mnli"]["licence_decision"] = "evaluation only"
    assert not registry.fit_allowed(bad, "mnli")[0]
    bad = copy.deepcopy(reg)                                # share-alike routed into the general head
    bad["sources"]["qasc"].update(licence=["other"], licence_text="x",
                                  licence_decision={"treat_as": "cc-by-sa-4.0", "reason": "y"})
    assert any("qasc" in p for p in registry.check_policy(bad))


def test_licence_decision_can_make_a_source_stricter_than_its_card():
    reg = registry.load()
    assert registry.fit_allowed(reg, "scinli")[1].startswith("allowed by recorded decision: treated as cc-by-sa-4.0")
    bad = copy.deepcopy(reg)                                # the card says Apache-2.0; the decision says share-alike
    bad["sources"]["scinli"].update(format="general", family="knowledge")
    assert not registry.fit_allowed(bad, "scinli")[0]
    bad = copy.deepcopy(reg)                                # a decision without its quoted evidence
    del bad["sources"]["fever"]["licence_decision"]["quote"]
    assert not registry.fit_allowed(bad, "fever")[0]


def test_typed_questions_keep_their_descriptions_and_gold():
    choice = prep_tiers.typed_question({"type": "choice", "instructions": "Route it.",
                                        "criteria": {"a": "Team A", "b": "Team B"}}, "b", {"a": 0.25, "b": 0.75})
    assert choice["options"] == {"a": "Team A", "b": "Team B"} and choice["gold"] == [0.25, 0.75]
    noul = prep_tiers.typed_question({"type": "noul", "instructions": "Refund it?",
                                      "criteria": {"true": "Allowed.", "false": "Not allowed."}}, "yes", None)
    assert noul["type"] == "bool" and noul["label"] == "true" and noul["gold"] == [1.0, 0.0]
    assert "Allowed." in noul["instructions"] and "Not allowed." in noul["instructions"]
    score = prep_tiers.typed_question({"type": "score", "instructions": "How urgent?",
                                       "criteria": ["Low.", "Mid.", "High."]}, "0", {"0": 0.5, "1": 0.3, "2": 0.2})
    assert score["levels"] == 3 and score["label"] == 1 and score["gold"] == [0.5, 0.3, 0.2]
    assert "1 = Low; 2 = Mid; 3 = High." in score["instructions"]


def test_items_sharing_a_passage_share_a_fit_split():
    items = [prep_tiers.Item(f"x-{i}", LONG.format("same"), f"question {i}", [], "true") for i in range(50)]
    assert len({prep_tiers.fit_split("general", it) for it in items}) == 1
    grouped = [prep_tiers.Item(f"y-{i}", LONG.format(i), None, [], "true", group="page:same") for i in range(50)]
    assert len({prep_tiers.fit_split("general", it) for it in grouped}) == 1
