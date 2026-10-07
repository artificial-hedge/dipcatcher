"""Adversarial probes for _ssl_synth (SYNTHETIC)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._ssl_synth import (
    _D,
    linear_probe_acc,
    make_views,
    synth_ssl,
    synth_tta_split,
)


def test_synth_ssl_deterministic():
    x1, y1 = synth_ssl(64, np.random.default_rng(0))
    x2, y2 = synth_ssl(64, np.random.default_rng(0))
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)


def test_synth_ssl_xor_not_linearly_separable():
    # y depends on sign(x0)*sign(x1) — XOR structure present by construction.
    x, y = synth_ssl(400, np.random.default_rng(1))
    s1 = np.sign(x[:, 0])
    s2 = np.sign(x[:, 1])
    assert np.mean((s1 * s2 > 0) == y) > 0.9


@pytest.mark.parametrize("n", [0, -1])
def test_synth_ssl_hostile_n(n):
    with pytest.raises(ValueError):
        synth_ssl(n, np.random.default_rng(0))


def test_make_views_rejects_bad_shape():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        make_views(np.zeros((5, _D - 1)), rng)
    with pytest.raises(ValueError):
        make_views(np.zeros((0, _D)), rng)
    with pytest.raises(ValueError):
        make_views(np.zeros(_D), rng)


def test_make_views_deterministic():
    x, _ = synth_ssl(8, np.random.default_rng(0))
    v1a, v2a = make_views(x, np.random.default_rng(3))
    v1b, v2b = make_views(x, np.random.default_rng(3))
    np.testing.assert_array_equal(v1a, v1b)
    np.testing.assert_array_equal(v2a, v2b)


def test_synth_tta_split_deterministic_rotation():
    out1 = synth_tta_split(32, 16, np.random.default_rng(5))
    out2 = synth_tta_split(32, 16, np.random.default_rng(5))
    for a, b in zip(out1, out2, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("n_tr,n_te", [(0, 8), (8, 0), (-2, 4)])
def test_synth_tta_split_hostile(n_tr, n_te):
    with pytest.raises(ValueError):
        synth_tta_split(n_tr, n_te, np.random.default_rng(0))


def test_synth_tta_split_test_is_rotated():
    xtr, _, xte, _ = synth_tta_split(64, 64, np.random.default_rng(0))
    # rotation+shift applied — columns 0/1 must NOT match a fresh ID draw.
    _, _, xid, _ = synth_tta_split(64, 64, np.random.default_rng(1))
    assert not np.allclose(xte.mean(0), xid.mean(0), atol=0.05)


def test_linear_probe_acc_perfect_labels():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 4))
    y = (x[:, 0] > 0).astype(np.int64)
    acc = linear_probe_acc(x, y, x, y)
    assert acc >= 0.9  # well-separated → near-perfect (K=4 head leaves margin)


def test_linear_probe_acc_hostile():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((10, 4))
    y = np.zeros(10, dtype=np.int64)
    with pytest.raises(ValueError):
        linear_probe_acc(x[:0], y[:0], x, y)
    with pytest.raises(ValueError):
        linear_probe_acc(x, y[:5], x, y)  # label/row mismatch
    with pytest.raises(ValueError):
        linear_probe_acc(x, y + 9, x, y)  # label out of range
    with pytest.raises(ValueError):
        linear_probe_acc(x[:, :3], y, x, y)  # feature-dim mismatch
