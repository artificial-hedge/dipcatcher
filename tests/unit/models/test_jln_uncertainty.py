"""Unit tests for quant_fund.models.jln_uncertainty."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.jln_uncertainty import bench_jln, jln_uncertainty


def _panel(n_s: int = 6, n_t: int = 200) -> np.ndarray:
    rng = np.random.default_rng(4)
    u = np.exp(np.cumsum(rng.standard_normal(n_t) * 0.05))
    p = np.empty((n_s, n_t))
    for i in range(n_s):
        e = np.empty(n_t)
        e[0] = 0.0
        for t in range(1, n_t):
            e[t] = 0.4 * e[t - 1] + np.sqrt(u[t]) * rng.standard_normal()
        p[i] = e
    return p


def test_output_shapes() -> None:
    res = jln_uncertainty(_panel(), window=30)
    assert res["index"].shape == (200,)
    assert res["series_u"].shape == (6, 200)
    assert 0.0 < float(res["share_pc1"][0]) <= 1.0


def test_index_tracks_common_vol() -> None:
    rng = np.random.default_rng(4)
    n_t = 200
    u = np.exp(np.cumsum(rng.standard_normal(n_t) * 0.05))
    u = u / u.mean()
    res = jln_uncertainty(_panel(), window=30)
    idx = res["index"]
    v = np.isfinite(idx)
    assert np.corrcoef(idx[v], np.log(u)[v])[0, 1] > 0.4


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        jln_uncertainty(np.ones((2, 200)))
    with pytest.raises(ValueError):
        jln_uncertainty(np.full((6, 200), np.nan))
    with pytest.raises(ValueError):
        jln_uncertainty(_panel(), window=500)


def test_bench_contract() -> None:
    out = bench_jln()
    assert out["score"] == 1.0
    assert out["synthetic_jln_corr"] > 0.55
