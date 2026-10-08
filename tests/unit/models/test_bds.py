"""Tests for bds — Brock-Dechert-Scheinkman independence test."""

import numpy as np
import pytest

from quant_fund.models.bds import bds_stat, bench_bds, synth_bds


def test_iid_null_passes() -> None:
    iid, _ = synth_bds(seed=1)
    r = bds_stat(iid, m=3)
    assert r["p2"] > 0.05
    assert r["p3"] > 0.05


def test_tent_map_rejected() -> None:
    _, chaos = synth_bds(seed=2)
    r = bds_stat(chaos, m=3)
    assert r["p2"] < 1e-6
    assert r["p3"] < 1e-6


def test_deterministic_structure_detected() -> None:
    _, chaos = synth_bds(seed=3)
    r = bds_stat(chaos, m=2)
    assert abs(r["w2"]) > 5


def test_fail_closed() -> None:
    iid, _ = synth_bds(seed=4)
    with pytest.raises(ValueError):
        bds_stat(iid[:100])
    with pytest.raises(ValueError):
        bds_stat(np.full(500, np.nan))
    with pytest.raises(ValueError):
        bds_stat(np.ones(500))
    with pytest.raises(ValueError):
        bds_stat(iid.reshape(450, 2))


def test_determinism() -> None:
    iid, chaos = synth_bds(seed=5)
    assert bds_stat(iid, m=3) == bds_stat(iid, m=3)
    assert bds_stat(chaos, m=3) == bds_stat(chaos, m=3)


def test_bench_schema_and_score() -> None:
    r = bench_bds()
    for k in (
        "synthetic_w2_null",
        "synthetic_p2_null",
        "synthetic_w2_chaos",
        "synthetic_p2_chaos",
        "synthetic_p3_chaos",
        "synthetic_score",
    ):
        assert np.isfinite(r[k])
    assert r["synthetic_score"] == 1.0
