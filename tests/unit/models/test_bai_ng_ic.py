"""Tests for bai_ng_ic (wave-57)."""

import numpy as np
import pytest

from quant_fund.models.bai_ng_ic import bai_ng_ic, bench_bai_ng, synth_factor


def test_recovers_true_r() -> None:
    panel, _ = synth_factor(seed=1)
    r = bai_ng_ic(panel)
    assert r["r_ic1"] == 3.0
    assert r["r_er"] == 3.0


def test_noise_panel_low_r() -> None:
    _, noise = synth_factor(seed=2)
    r = bai_ng_ic(noise)
    assert r["r_er"] <= 2.0


def test_fail_closed() -> None:
    panel, _ = synth_factor(seed=3)
    with pytest.raises(ValueError):
        bai_ng_ic(panel[:10])
    with pytest.raises(ValueError):
        bai_ng_ic(np.ones(30))
    with pytest.raises(ValueError):
        bai_ng_ic(np.full((50, 30), np.nan))


def test_determinism() -> None:
    panel, _ = synth_factor(seed=4)
    assert bai_ng_ic(panel) == bai_ng_ic(panel)


def test_bench_schema_and_score() -> None:
    r = bench_bai_ng()
    for k, v in r.items():
        assert np.isfinite(v), k
    assert r["synthetic_score"] == 1.0
