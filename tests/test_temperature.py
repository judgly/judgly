"""The per-type temperature, the second calibration option.

- the head file: written by s1-train --head temperature, read by the library, refused when its
  temperature lies outside [0.05, 100] or its layout is wrong, checked by check_heads.py;
- the fit: the temperature that minimises the log loss of the rotation-averaged raw readout,
  rounded to three decimals;
- the application: after the mean over the option orders, never per order (s1-eval on a
  synthetic feature file, and the live engine against its own raw output);
- the choice: Pack.heads(calibration=...) and Engine.load(calibration=...), their defaults and
  their errors.

The trainer and evaluator tests run build/cli/s1-train and s1-eval and skip when the tools are not
built (make tools); the engine tests need the Qwen3-4B model (see conftest.py).
"""

import json
import math
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import FIXTURES, QWEN, REPO, need
from judgly import Engine, Pack
from judgly._native import _lib
from test_guardrails import check_heads, write_features

TOOLS = REPO / "build" / "cli"
K_MAX, EMBD = 26, 8
CHOICE, BOOL, SCORE = 0, 1, 2
FLOOR = 1e-12
PARAMS_AT = 8 + 4 + 4 + 4 + 65 + 65 + 4 * K_MAX   # csrc/s1_headfile.c: where the parameters start
# The shipped H2 heads of judgly 0.1.0: unchanged in this release.
H2_SHA = {("qwen3-4b-q8", "*"): "9da59ab5a64c7f3782853bee054a2be43173d519217e20e8b284bfe30a6c2828",
          ("qwen3-4b-q8", "stance"): "f85c4fe3e9e0bf78af143a55b460cb220d1293168d23c9501bdee8f471b99c0c",
          ("gemma4-12b-q8", "*"): "bdf84fac9bb91fefac915e168d95dc5f42fdadae48e5590d70223e0b8e78d9ff",
          ("gemma4-12b-q8", "stance"): "ac30505af5d62bad368df67cd12e14eff5684103f71e3897bd9c4387e4e5faa5"}
# The temperatures confirmed on untouched data (docs/calibration.md), per type: choice, bool, score.
CONFIRMED = {("qwen3-4b-q8", "*"): (6.846, 13.36, 24.22), ("qwen3-4b-q8", "stance"): (12.391, 1.0, 1.0),
             ("gemma4-12b-q8", "*"): (3.461, 7.491, 6.538), ("gemma4-12b-q8", "stance"): (7.151, 1.0, 1.0)}


def temperature_of(p: np.ndarray, t: float) -> np.ndarray:
    """csrc/s1_temperature.c: p_T proportional to exp(log(max(p, 1e-12)) / T)."""
    if t == 1.0:
        return p
    u = np.log(np.clip(p, FLOOR, 1.0)) / t
    e = np.exp(u - u.max())
    return e / e.sum()


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def choice_records(rng, n=400, K=3) -> list[dict]:
    """Choice items read in all K orders: the letter scores favour the right option, plus a
    preference for slot A and noise, so that the orders disagree. Items 1-200 are the train
    split, 201-300 validation, 301-400 held out."""
    out = []
    for i in range(n):
        label = int(rng.integers(0, K))
        split = 0 if i < 200 else 1 if i < 300 else 3
        for rot in range(K):
            z = [4.0 * (((s + rot) % K) == label) + 3.0 * (s == 0) + float(rng.normal(0, 2)) for s in range(K)]
            out.append({"id": i + 1, "type": CHOICE, "split": split, "K": K, "label": label, "rot": rot,
                        "z": z, "h": [0.0] * EMBD})
    return out


def run(tool: str, *args: str) -> subprocess.CompletedProcess:
    if not (TOOLS / tool).is_file():
        pytest.skip(f"build/cli/{tool} is not built (make tools)")
    return subprocess.run([str(TOOLS / tool), *map(str, args)], capture_output=True, text=True)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """A synthetic feature file and the temperature head s1-train fits on it."""
    tmp = tmp_path_factory.mktemp("temperature")
    records = choice_records(np.random.default_rng(7))
    feat, head = tmp / "f.feat", tmp / "temperature.bin"
    write_features(feat, records)
    done = run("s1-train", "--features", feat, "--head", "temperature", "--out", head, "--rotations")
    assert done.returncode == 0, done.stderr
    return {"records": records, "feat": feat, "head": head, "dir": tmp}


def items(records: list[dict], split: int) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Per item of a split: its label and its raw probabilities averaged over the orders."""
    by_id: dict[int, list[dict]] = {}
    for r in records:
        if r["split"] == split:
            by_id.setdefault(r["id"], []).append(r)
    ids, labels, probs = [], [], []
    for i, rs in sorted(by_id.items()):
        K = rs[0]["K"]
        mean = np.zeros(K)
        for r in rs:
            p = softmax(np.array(r["z"][:K], dtype=np.float32).astype(np.float64))
            for s in range(K):
                mean[(s + r["rot"]) % K] += p[s]
        ids.append(i)
        labels.append(rs[0]["label"])
        probs.append(mean / len(rs))
    return ids, np.array(labels), np.array(probs)


# ---- the head file ------------------------------------------------------------------------------

def test_head_file_layout_and_sidecar(fitted):
    head = fitted["head"]
    raw = head.read_bytes()
    assert raw[:8] == b"S1HEAD\0\0" and struct.unpack_from("<III", raw, 8) == (1, 3, 0)
    assert len(raw) == PARAMS_AT + 3 * 8
    assert _lib().judgly_test_head_load(str(head).encode()) == 0
    side = json.loads(Path(f"{head}.json").read_text())
    assert side["head"] == "temperature"
    assert side["engine"] == {"rotations": True, "max_rotations": 0, "content_free": False}
    assert side["types"]["choice"]["fallback"] is None
    assert side["types"]["bool"]["fallback"] == "identity" and side["types"]["score"]["fallback"] == "identity"
    params = check_heads.head_params(head)
    assert params["choice"] == [side["types"]["choice"]["temperature"]]
    assert params["bool"] == [1.0] and params["score"] == [1.0]
    assert check_heads.problems(head, side["engine"]) == []


@pytest.mark.parametrize("value, loads", [(0.05, True), (100.0, True), (1.0, True), (0.04, False),
                                          (100.5, False), (float("nan"), False), (float("inf"), False),
                                          (-3.0, False)])
def test_head_file_temperature_range(fitted, tmp_path, value, loads):
    raw = bytearray(fitted["head"].read_bytes())
    struct.pack_into("<d", raw, PARAMS_AT, value)
    path = tmp_path / "t.bin"
    path.write_bytes(bytes(raw))
    assert (_lib().judgly_test_head_load(str(path).encode()) == 0) is loads


def test_head_file_layout_refused(fitted, tmp_path):
    good = fitted["head"].read_bytes()
    cases = {"width": good[:16] + struct.pack("<I", 8) + good[20:],     # a temperature head has n_embd 0
             "short": good[:-1], "long": good + b"\0"}
    for name, content in cases.items():
        path = tmp_path / f"{name}.bin"
        path.write_bytes(content)
        assert _lib().judgly_test_head_load(str(path).encode()) == -1, name


def test_sidecar_must_match_the_file(fitted, tmp_path):
    head = tmp_path / "temperature.bin"
    head.write_bytes(fitted["head"].read_bytes())
    side = json.loads(Path(f"{fitted['head']}.json").read_text())
    side["types"]["choice"]["temperature"] += 0.001
    Path(f"{head}.json").write_text(json.dumps(side))
    assert any("is not the sidecar's" in p for p in check_heads.problems(head))


def test_trainer_refuses_options_that_do_not_apply(fitted, tmp_path):
    for extra in (["--limit-train", "10"], ["--probe"]):
        done = run("s1-train", "--features", fitted["feat"], "--head", "temperature", "--out",
                   tmp_path / "t.bin", "--rotations", *extra)
        assert done.returncode == 2 and "usage" in done.stderr


# ---- the fit ------------------------------------------------------------------------------------

def test_fit_minimises_the_training_log_loss(fitted):
    """The sidecar's unrounded temperature is the minimum of the mean training log loss of the
    rotation-averaged raw readout, found here independently by golden-section search in log T;
    the head holds it rounded to three decimals."""
    _, y, p = items(fitted["records"], 0)

    def loss(log_t: float) -> float:
        t = math.exp(log_t)
        return -float(np.mean([math.log(max(temperature_of(q, t)[k], FLOOR)) for q, k in zip(p, y)]))

    lo, hi = math.log(0.05), math.log(100.0)
    g = (math.sqrt(5) - 1) / 2
    for _ in range(80):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        lo, hi = (lo, b) if loss(a) < loss(b) else (a, hi)
    ours = math.exp((lo + hi) / 2)
    side = json.loads(Path(f"{fitted['head']}.json").read_text())["types"]["choice"]
    assert side["temperature_unrounded"] == pytest.approx(ours, rel=1e-6)
    assert side["temperature"] == round(side["temperature_unrounded"], 3)
    assert side["train_items"] == 200 and side["validation_items"] == 100
    assert side["train_loss"] == pytest.approx(loss(math.log(side["temperature"])), abs=1e-6)


# ---- the application: after the mean over orders -------------------------------------------------

def test_temperature_is_applied_after_the_mean_over_orders(fitted):
    """s1-eval with the temperature head equals the temperature applied to the averaged raw
    probabilities, and differs from the temperature applied to each order before averaging."""
    dump = fitted["dir"] / "items.tsv"
    done = run("s1-eval", "--features", fitted["feat"], "--split", "heldout", "--rotations",
               "--head", fitted["head"], "--dump-items", dump)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("temperature + rotations on heldout")
    t = check_heads.head_params(fitted["head"])["choice"][0]
    ids, _, p = items(fitted["records"], 3)
    rows = [line.split("\t") for line in dump.read_text().splitlines()[1:]]
    assert len(rows) == len(ids)
    engine = {r[0]: np.array([float(v) for v in r[6].split(",")]) for r in rows}
    worst, per_order_gap = 0.0, 0.0
    for i, q in zip(ids, p):
        ours = engine[f"{i:016x}"]
        worst = max(worst, float(np.abs(ours - temperature_of(q, t)).max()))
        rs = [r for r in fitted["records"] if r["id"] == i]
        before = np.zeros(len(q))
        for r in rs:
            pr = temperature_of(softmax(np.array(r["z"][:r["K"]], dtype=np.float32).astype(np.float64)), t)
            for s in range(r["K"]):
                before[(s + r["rot"]) % r["K"]] += pr[s] / len(rs)
        per_order_gap = max(per_order_gap, float(np.abs(ours - before).max()))
    assert worst < 1e-12
    assert per_order_gap > 1e-3   # the order of the two operations matters on these items


def test_temperature_one_is_the_identity(tmp_path):
    """A type whose temperature is 1 answers exactly as without a head."""
    records = [{**r, "type": BOOL, "K": 2, "z": r["z"][:2], "label": r["label"] % 2, "rot": r["rot"] % 2}
               for r in choice_records(np.random.default_rng(9), K=2)]
    feat, head = tmp_path / "f.feat", tmp_path / "t.bin"
    write_features(feat, records)
    done = run("s1-train", "--features", feat, "--head", "temperature", "--out", head, "--rotations")
    assert done.returncode == 0, done.stderr
    raw = bytearray(head.read_bytes())
    struct.pack_into("<d", raw, PARAMS_AT + 8, 1.0)          # bool
    head.write_bytes(bytes(raw))
    out = {}
    for name, extra in (("h0", []), ("t", ["--head", head])):
        done = run("s1-eval", "--features", feat, "--split", "heldout", "--rotations",
                   "--dump-items", tmp_path / f"{name}.tsv", *extra)
        assert done.returncode == 0, done.stderr
        out[name] = (tmp_path / f"{name}.tsv").read_text()
    assert out["h0"] == out["t"]


# ---- the choice ---------------------------------------------------------------------------------

@pytest.mark.parametrize("pack", ["qwen3-4b-q8", "gemma4-12b-q8"])
def test_packs_offer_both_options(pack):
    p = Pack.find(pack)
    assert p.spec["schema"] == 2
    h2, temp = p.heads(calibration="h2"), p.heads(calibration="temperature")
    assert set(h2) == set(temp) == {"*", "stance"}
    from judgly.packs import sha256_file
    for fmt in ("*", "stance"):
        assert sha256_file(h2[fmt]) == H2_SHA[(pack, fmt)]
        assert check_heads.head_params(temp[fmt]) == dict(zip(("choice", "bool", "score"),
                                                              ([v] for v in CONFIRMED[(pack, fmt)])))
    assert p.heads(calibration="raw") == {}


def test_pack_defaults_follow_the_confirmation():
    assert Pack.find("qwen3-4b-q8").defaults() == {"*": "temperature", "stance": "temperature"}
    assert Pack.find("gemma4-12b-q8").defaults() == {"*": "h2", "stance": "temperature"}
    gemma = Pack.find("gemma4-12b-q8")
    assert gemma.heads() == {"*": gemma.heads(calibration="h2")["*"],
                             "stance": gemma.heads(calibration="temperature")["stance"]}


def test_unknown_calibration_is_refused():
    with pytest.raises(ValueError, match="calibration must be one of"):
        Pack.find("qwen3-4b-q8").heads(calibration="platt")
    with pytest.raises(ValueError, match="calibration must be one of"):
        Engine.load("qwen3-4b-q8", calibration="Temperature")
    with pytest.raises(ValueError, match="heads=False"):
        Engine.load("qwen3-4b-q8", heads=False, calibration="h2")
    with pytest.raises(ValueError, match="heads given as files"):
        Engine.load("qwen3-4b-q8", heads={"*": "x.bin"}, calibration="temperature")


def test_schema_one_pack_has_one_option(tmp_path):
    (tmp_path / "template.tpl").write_text('{"open": "", "close": ""}')
    (tmp_path / "heads").mkdir()
    (tmp_path / "heads" / "h.bin").write_bytes(b"a head")
    import hashlib
    spec = {"schema": 1, "name": "old", "template": "template.tpl",
            "heads": {"*": {"file": "heads/h.bin", "sha256": hashlib.sha256(b"a head").hexdigest()}},
            "model": {"repo_id": "none/none", "revision": "0" * 40, "filename": "m.gguf",
                      "sha256": "0" * 64, "size": 1}}
    (tmp_path / "pack.json").write_text(json.dumps(spec))
    p = Pack.find(tmp_path)
    assert p.heads() == {"*": tmp_path / "heads" / "h.bin"} and p.defaults() == {"*": "head"}
    with pytest.raises(ValueError, match="offers no calibration 'temperature'"):
        p.heads(calibration="temperature")


# ---- the live engine (Qwen3-4B) ------------------------------------------------------------------

GOLDEN = json.loads((FIXTURES / "golden-qwen3-4b-q8.json").read_text())


@pytest.fixture(scope="module")
def qwen_temperature():
    need(QWEN)
    with Engine.load("qwen3-4b-q8", calibration="temperature", model_path=QWEN) as engine:
        yield engine


@pytest.mark.model
def test_default_and_explicit_choice_record_the_option():
    need(QWEN)
    with Engine.load("qwen3-4b-q8", model_path=QWEN, n_ctx=4096) as engine:
        assert engine.calibration == {"*": "temperature", "stance": "temperature"}
        assert engine.config.heads == Pack.find("qwen3-4b-q8").heads(calibration="temperature")


@pytest.mark.model
def test_engine_applies_the_temperature_to_its_raw_mean(qwen_temperature, qwen_h0):
    """The engine's temperature answer is its own raw (H0) answer, averaged over the orders, with
    the temperature of the question's type applied: nothing else differs."""
    request = json.dumps(GOLDEN["request"])
    raw = json.loads(qwen_h0.decide_json(request))["answers"]
    ours = json.loads(qwen_temperature.decide_json(request))["answers"]
    t = dict(zip(("choice", "bool", "score"), CONFIRMED[("qwen3-4b-q8", "*")]))
    for qid, spec in GOLDEN["request"]["questions"].items():
        a, r = ours[qid], raw[qid]
        if spec["type"] == "bool":
            # the response carries p_true only; p_false = 1 - p_true loses the relative precision
            # of a small p_false, which the temperature's root magnifies, hence the looser bound
            p, q, tol = np.array([r["p_true"], 1 - r["p_true"]]), np.array([a["p_true"], 1 - a["p_true"]]), 1e-7
        elif spec["type"] == "choice":
            p, q, tol = np.array(list(r["probs"].values())), np.array(list(a["probs"].values())), 1e-12
        else:
            p, q, tol = np.array(r["probs"]), np.array(a["probs"]), 1e-12
        assert float(np.abs(q - temperature_of(p, t[spec["type"]])).max()) < tol, qid
        assert a["head_format"] == "*" and a["head"] is not None
        assert a["rotation_spread"] == r["rotation_spread"] and a["slot_mass"] == r["slot_mass"]


@pytest.mark.model
def test_golden_temperature(qwen_temperature):
    from test_engine import assert_golden
    ours = json.loads(qwen_temperature.decide_json(json.dumps(GOLDEN["request"])))
    assert_golden(ours, GOLDEN["temperature"])
