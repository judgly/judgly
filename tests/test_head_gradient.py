"""Self-test T7 in Python: the analytic gradient of the head loss (H1 and H2) against central
differences, on synthetic records with soft targets, content-free logits and a penalty, so that
every term of the loss is exercised. Runs the C code in libjudgly; needs no model."""

import ctypes

import numpy as np

from judgly._native import _lib

K_MAX = 26
RECORDS, EMBD, PROBES, STEP, TOL = 200, 24, 50, 1e-4, 1e-4


def n_param(h2: bool) -> int:
    return 2 + K_MAX + (K_MAX * EMBD if h2 else 0)


def records(rng):
    K = (2 + np.arange(RECORDS) % 9).astype(np.uint8)
    z = (4.0 * rng.uniform(-1, 1, (RECORDS, K_MAX))).astype(np.float32)
    zc = (2.0 * rng.uniform(-1, 1, (RECORDS, K_MAX))).astype(np.float32)
    t = rng.uniform(0, 2, (RECORDS, K_MAX))
    t[np.arange(K_MAX)[None, :] >= K[:, None]] = 0.0
    t = (t / t.sum(axis=1, keepdims=True)).astype(np.float32)
    h = rng.uniform(-1, 1, (RECORDS, EMBD)).astype(np.float32)
    return h, z, zc, t, K


def loss(x, h2, lam, data):
    h, z, zc, t, K = data
    g = np.zeros_like(x)
    f32 = ctypes.POINTER(ctypes.c_float)
    f64 = ctypes.POINTER(ctypes.c_double)
    value = _lib().judgly_test_head_loss(
        x.ctypes.data_as(f64), g.ctypes.data_as(f64), x.size, int(h2), EMBD, RECORDS,
        h.ctypes.data_as(f32), z.ctypes.data_as(f32), zc.ctypes.data_as(f32),
        t.ctypes.data_as(f32), K.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), lam)
    return value, g


def worst_error(h2, lam, data, rng):
    x = 0.3 * rng.uniform(-1, 1, n_param(h2))
    _, g = loss(x, h2, lam, data)
    worst = 0.0
    for i in rng.integers(0, x.size, PROBES):
        kept = x[i]
        x[i] = kept + STEP
        up, _ = loss(x, h2, lam, data)
        x[i] = kept - STEP
        down, _ = loss(x, h2, lam, data)
        x[i] = kept
        numeric = (up - down) / (2 * STEP)
        worst = max(worst, abs(g[i] - numeric) / max(1e-8, abs(g[i]) + abs(numeric)))
    return worst


def test_gradient_h1_h2():
    rng = np.random.default_rng(20260922)
    data = records(rng)
    assert worst_error(False, 0.0, data, rng) < TOL
    assert worst_error(True, 1e-2, data, rng) < TOL


def test_identity_head_is_h0():
    """All-zero parameters reproduce softmax(z - 0 * zc): the mean cross-entropy of H0."""
    rng = np.random.default_rng(1)
    h, z, zc, t, K = data = records(rng)
    value, _ = loss(np.zeros(n_param(True)), True, 0.0, data)
    expected = 0.0
    for r in range(RECORDS):
        u = z[r, :K[r]].astype(np.float64)
        logp = u - (u.max() + np.log(np.exp(u - u.max()).sum()))
        expected -= float((t[r, :K[r]] * logp).sum())
    assert abs(value - expected / RECORDS) < 1e-9
