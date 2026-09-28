"""Guardrails of the head fit and of serving; none needs a model.

- score questions are read in their natural order only (one rotation), everywhere;
- the fitted temperature is held within [0.05, 100], and H2 starts from a capped temperature;
- a question type whose fitted head is input-independent, or does not beat the raw readout on
  validation, falls back to the identity, recorded in the sidecar, and check_heads.py fails a
  head that lacks the fallback;
- a head is refused when served under other engine settings than it was fitted under.

The trainer tests run build/cli/s1-train on small synthetic feature files and skip when the
tools are not built (make tools).
"""

import ctypes
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from judgly._native import _lib
from judgly.engine import EngineConfig, JudglyError, check_head_settings

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import check_heads  # noqa: E402

TRAIN = REPO / "build" / "cli" / "s1-train"
K_MAX, EMBD = 26, 8
CHOICE, BOOL, SCORE = 0, 1, 2
f32, f64 = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_double)


# ---- the rules themselves, through the library ------------------------------------------------

def test_score_questions_keep_their_natural_order():
    n = _lib().judgly_test_n_rotations
    for K in range(2, 10):
        assert n(SCORE, K, 1, 0) == 1 and n(SCORE, K, 1, 4) == 1 and n(SCORE, K, 0, 4) == 1
    assert n(CHOICE, 5, 1, 4) == 4 and n(CHOICE, 3, 1, 4) == 3 and n(CHOICE, 5, 0, 4) == 1
    assert n(BOOL, 2, 0, 4) == 2 and n(BOOL, 2, 1, 1) == 1


def test_temperature_is_bounded():
    clamp = _lib().judgly_test_theta_clamp
    assert math.exp(-clamp(-math.log(7976649.6))) == pytest.approx(100.0)
    assert math.exp(-clamp(-math.log(0.001))) == pytest.approx(0.05)
    assert clamp(-math.log(4.69)) == -math.log(4.69)  # inside the bounds: untouched


def test_h2_starts_from_a_capped_temperature():
    start = _lib().judgly_test_h2_start_theta
    assert math.exp(-start(-math.log(100.0))) == pytest.approx(20.0)
    assert start(-math.log(8.47)) == -math.log(8.47)


def test_prob_sd_is_zero_for_an_input_independent_head():
    rng = np.random.default_rng(3)
    n = 50
    K = np.full(n, 5, dtype=np.uint8)
    h = rng.normal(size=(n, EMBD)).astype(np.float32)
    zc = np.zeros((n, K_MAX), dtype=np.float32)
    z = rng.normal(0, 3, (n, K_MAX)).astype(np.float32)

    def sd(x, z):
        return _lib().judgly_test_head_prob_sd(x.ctypes.data_as(f64), 0, EMBD, n, h.ctypes.data_as(f32),
                                               z.ctypes.data_as(f32), zc.ctypes.data_as(f32),
                                               K.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))

    x = np.zeros(2 + K_MAX)
    assert sd(x, z) > 0.1
    x[0] = -math.log(7976649.6)  # the collapsed score head: a = 1.25e-7
    assert sd(x, z) < 1e-5
    assert sd(np.zeros(2 + K_MAX), np.zeros_like(z)) == 0.0


# ---- the trainer, on synthetic feature files --------------------------------------------------

def write_features(path: Path, records: list[dict]) -> None:
    """A feature file in the layout of csrc/s1_feat.c."""
    header = b"S1FEAT\0\0" + struct.pack("<IIIIQ", 1, EMBD, K_MAX, 0, len(records))
    header += bytes(64) + struct.pack(f"<{K_MAX}I", *range(K_MAX))
    with open(path, "wb") as f:
        f.write(header)
        for r in records:
            K, rot, label = r["K"], r.get("rot", 0), r["label"]
            perm = [(s + rot) % K for s in range(K)] + [0] * (K_MAX - K)
            slot = perm.index(label)
            target = [0.0] * K_MAX
            target[slot] = 1.0
            z = list(r["z"]) + [0.0] * (K_MAX - K)
            f.write(struct.pack("<QIHBBBBB", r["id"], 1, 1, r["type"], r["split"], K, slot, rot))
            f.write(bytes(perm))
            f.write(struct.pack(f"<{K_MAX}f{K_MAX}f{K_MAX}ff{EMBD}f", *target, *z, *([0.0] * K_MAX), 1.0,
                                *r["h"]))


def score_records(rng, z_of, h_of, n_train=300, n_val=100) -> list[dict]:
    out = []
    for i in range(n_train + n_val):
        label = int(rng.integers(0, 5))
        out.append({"id": i + 1, "type": SCORE, "split": 0 if i < n_train else 1, "K": 5, "label": label,
                    "z": z_of(label), "h": h_of(label)})
    return out


def train(tmp_path: Path, records: list[dict], *engine: str) -> tuple[subprocess.CompletedProcess, Path]:
    if not TRAIN.is_file():
        pytest.skip("build/cli/s1-train is not built (make tools)")
    feat, head = tmp_path / "f.feat", tmp_path / "h2.bin"
    write_features(feat, records)
    done = subprocess.run([str(TRAIN), "--features", str(feat), "--head", "h2", "--out", str(head), *engine],
                          capture_output=True, text=True)
    return done, head


ENGINE = ("--rotations", "--max-rotations", "4")


def test_trainer_bounds_the_temperature_of_confidently_wrong_readings(tmp_path):
    """Readings that are confident and unrelated to the label drive the unbounded temperature
    to infinity (the head discards them); it is held at 100 instead."""
    rng = np.random.default_rng(1)
    wrong = lambda label: [30.0 * (k == (label + 1 + rng.integers(0, 4)) % 5) for k in range(5)]  # noqa: E731
    done, head = train(tmp_path, score_records(rng, wrong, lambda _: rng.normal(size=EMBD)), *ENGINE)
    assert done.returncode == 0, done.stderr
    score = json.loads(Path(f"{head}.json").read_text())["types"]["score"]
    assert score["temperature_bounded"] is True and score["temperature"] <= 100.0 + 1e-9
    assert check_heads.problems(head, {"rotations": True, "max_rotations": 4, "content_free": False}) == []


def test_trainer_falls_back_to_the_identity_for_an_input_independent_type(tmp_path):
    """No signal at all (z and h zero): the fitted head is a constant; the type falls back to
    the identity, the sidecar says so and why, and the head check accepts it."""
    rng = np.random.default_rng(2)
    done, head = train(tmp_path, score_records(rng, lambda _: [0.0] * 5, lambda _: [0.0] * EMBD), *ENGINE)
    assert done.returncode == 0, done.stderr
    side = json.loads(Path(f"{head}.json").read_text())
    assert side["engine"] == {"rotations": True, "max_rotations": 4, "content_free": False}
    assert side["types"]["score"]["fallback"] == "identity"
    assert side["types"]["score"]["reason"] == "input-independent on validation"
    assert check_heads.head_params(head)["score"] == [0.0] * (2 + K_MAX + K_MAX * EMBD)
    assert check_heads.problems(head) == []
    # the same head with the fallback removed from its sidecar fails the check
    side["types"]["score"].update(fallback=None, reason=None)
    Path(f"{head}.json").write_text(json.dumps(side))
    assert any("input-independent" in p for p in check_heads.problems(head))


def test_trainer_refuses_rotated_score_records(tmp_path):
    rng = np.random.default_rng(3)
    records = []
    for r in score_records(rng, lambda _: list(rng.normal(size=5)), lambda _: rng.normal(size=EMBD)):
        records += [{**r, "rot": rot} for rot in range(4)]
    done, _ = train(tmp_path, records, *ENGINE)
    assert done.returncode != 0 and "orders" in done.stderr


def test_head_check_requires_engine_settings(tmp_path):
    rng = np.random.default_rng(4)
    done, head = train(tmp_path, score_records(rng, lambda lv: [3.0 * (k == lv) for k in range(5)],
                                               lambda _: rng.normal(size=EMBD)), *ENGINE)
    assert done.returncode == 0, done.stderr
    assert check_heads.problems(head, {"rotations": True, "max_rotations": 4, "content_free": False}) == []
    assert check_heads.problems(head, {"rotations": True, "max_rotations": 0, "content_free": False})
    side = json.loads(Path(f"{head}.json").read_text())
    del side["engine"]
    Path(f"{head}.json").write_text(json.dumps(side))
    assert any("engine settings" in p for p in check_heads.problems(head))


# ---- serving: settings recorded with the head are enforced -------------------------------------

def config_with_head(tmp_path: Path, fitted: dict | None, **settings) -> EngineConfig:
    head = tmp_path / "h2.bin"
    head.write_bytes(b"")
    if fitted is not None:
        Path(f"{head}.json").write_text(json.dumps({"engine": fitted}))
    return EngineConfig(model=tmp_path / "m.gguf", template=tmp_path / "t.tpl", heads={"*": head}, **settings)


def test_head_fitted_under_other_engine_settings_is_refused(tmp_path):
    fitted = {"rotations": True, "max_rotations": 4, "content_free": False}
    check_head_settings(config_with_head(tmp_path, fitted))  # the defaults are the packs' settings
    with pytest.raises(JudglyError, match="engine settings"):
        check_head_settings(config_with_head(tmp_path, fitted, max_rotations=0))
    with pytest.raises(JudglyError, match="engine settings"):
        check_head_settings(config_with_head(tmp_path, fitted, content_free=True))
    with pytest.raises(JudglyError, match="does not record"):
        check_head_settings(_no_engine(tmp_path))


def _no_engine(tmp_path: Path) -> EngineConfig:
    head = tmp_path / "h2.bin"
    head.write_bytes(b"")
    Path(f"{head}.json").write_text(json.dumps({"head": "h2"}))
    return EngineConfig(model=tmp_path / "m.gguf", template=tmp_path / "t.tpl", heads={"*": head})


def test_builtin_pack_heads_record_their_engine_settings():
    from judgly.packs import Pack
    for name in ("qwen3-4b-q8", "gemma4-12b-q8"):
        pack = Pack.find(name)
        for head in pack.heads().values():
            side = json.loads(Path(f"{head}.json").read_text())
            assert side["engine"] == {k: pack.engine_options[k] for k in ("rotations", "max_rotations", "content_free")}
            assert check_heads.problems(head, side["engine"]) == [], (name, head)


def test_score_heads_learn_no_level_prior(tmp_path):
    """Score fit data where most answers are level 1: the level-balanced weights keep the fitted
    per-level biases from favouring level 1, and score heads carry no row corrections (H1)."""
    rng = np.random.default_rng(5)
    records = []
    for i in range(500):
        label = 0 if rng.random() < 0.8 else int(rng.integers(1, 5))
        z = [2.0 * (k == label) + float(rng.normal()) for k in range(5)]
        records.append({"id": i + 1, "type": SCORE, "split": 0 if i < 400 else 1, "K": 5, "label": label,
                        "z": z, "h": rng.normal(size=EMBD)})
    done, head = train(tmp_path, records, *ENGINE)
    assert done.returncode == 0, done.stderr
    x = check_heads.head_params(head)["score"]
    bias = x[2:7]
    assert max(bias) - min(bias) < 0.5, bias  # the level frequencies alone give log(0.8 / 0.05) = 2.8
    assert all(v == 0.0 for v in x[2 + K_MAX:])
