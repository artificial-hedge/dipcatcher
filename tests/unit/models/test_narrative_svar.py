"""Unit tests for quant_fund.models.narrative_svar."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.narrative_svar import (
    bench_narrative,
    narrative_irf,
    synth_narrative,
)


def test_admissible_share_pos() -> None:
    y, t_star, _ = synth_narrative(seed=1)
    r = narrative_irf(y, 0, t_star, 1.0, seed=5)
    assert float(r["share"]) > 0.05


def test_narrative_tightens_impact() -> None:
    y, t_star, b = synth_narrative(seed=2)
    r = narrative_irf(y, 0, t_star, 1.0, seed=6)
    med = np.asarray(r["median_irf"])
    unrestr = np.asarray(r["unrestr_med_irf"])
    assert float(med[0, 0]) > float(unrestr[0, 0]) + 0.2


def test_wrong_sign_narrative_kills_draws() -> None:
    y, t_star, _ = synth_narrative(seed=3)
    r = narrative_irf(y, 0, t_star, -1.0, seed=7, n_draws=200)
    # planted episode was positive; negative narrative ~empty
    assert float(r["share"]) < 0.35


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        narrative_irf(np.ones((10, 2)), 0, 5, 1.0)
    y, t_star, _ = synth_narrative(seed=4)
    with pytest.raises(ValueError):
        narrative_irf(y, 0, t_star, 0.5)


def test_bench_contract() -> None:
    out = bench_narrative()
    assert out["score"] == 1.0
    assert out["synthetic_ns_share"] > 0.0
