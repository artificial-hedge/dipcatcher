"""Limit aggregation and the drawdown stops that flatten a book."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.risk.gates import (
    GateSpec,
    apply_gate_stack,
    crash_leverage,
    crc_leverage,
    dd_remaining_leverage,
    es_leverage,
    kelly_leverage,
    vol_target,
    vol_target_leverage,
)
from quant_fund.risk.overlay import BookRiskOverlay


def test_gate_inputs_fail_closed_and_a_drawdown_stops_the_book() -> None:
    with pytest.raises(ValueError, match="vol target"):
        vol_target(np.ones(8), target=0.0)
    flat = vol_target(np.zeros(12), target=0.025, lookback=4)
    assert float(np.max(np.abs(flat))) == 0.0
    leverage = vol_target_leverage(np.zeros(12), target=0.025, lookback=4)
    assert float(leverage[0]) == pytest.approx(min(1.0, 0.025 / 0.20))

    path = np.concatenate([np.full(8, 0.001), np.array([float("nan"), -0.2])])
    remaining = dd_remaining_leverage(path, limit=0.05, cushion=0.005)
    assert float(remaining[-1]) == 0.0
    assert remaining.shape == path.shape

    with pytest.raises(ValueError, match="kelly"):
        kelly_leverage(np.ones(8), fraction=float("nan"))
    quiet = es_leverage(np.zeros(20), es_limit=0.01, lookback=8)
    assert float(np.max(quiet)) <= 1.0
    unmeasured = es_leverage(np.full(20, np.nan), es_limit=0.01, lookback=8)
    assert float(unmeasured[-1]) == 0.0
    tail = crc_leverage(np.full(20, np.nan), alpha=0.05, lookback=8, step=3)
    assert float(tail[-1]) == 0.0

    ruined = np.concatenate([np.full(4, 0.01), np.array([-1.0]), np.full(4, 0.01)])
    halted = crash_leverage(ruined, lookback=2, crash_return=-0.2)
    assert float(halted[-1]) == 0.0

    spec = GateSpec(
        vol_target=None,
        kelly_fraction=None,
        crc_alpha=None,
        es_limit=None,
        crash_lookback=0,
        dd_limit=0.05,
        dd_mode="remaining",
        stepm_enable=True,
        stepm_min_obs=8,
        stepm_step=4,
        stepm_n_boot=4,
    )
    series = np.concatenate([np.full(12, 0.001), np.full(6, -0.03)])
    stacked = apply_gate_stack(series, spec, stepm_excess=series)
    snap = stacked.snapshot()
    assert snap["live_pnl_claim"] is False
    assert snap["research_only"] is True
    assert "dd=remaining-budget" in snap["notes"]
    assert "stepm=Romano–Wolf 2005 allow/deny" in snap["notes"]
    assert snap["n_halt"] >= 1


def test_overlay_halts_on_drawdown_and_rejects_a_bad_budget() -> None:
    with pytest.raises(ValueError, match="vol_target"):
        BookRiskOverlay(vol_target=0.0)
    with pytest.raises(ValueError, match="dd_limit"):
        BookRiskOverlay(dd_limit=1.0)
    with pytest.raises(ValueError, match="es_limit"):
        BookRiskOverlay(es_limit=-1.0)
    with pytest.raises(ValueError, match="kelly_fraction"):
        BookRiskOverlay(kelly_fraction=-0.1)
    with pytest.raises(ValueError, match="crc_alpha"):
        BookRiskOverlay(crc_alpha=1.0)

    overlay = BookRiskOverlay(
        vol_target=0.10,
        dd_limit=0.05,
        lookback=8,
        kelly_lookback=8,
        kelly_fraction=0.25,
        crc_lookback=8,
        crc_alpha=0.05,
    )
    overlay.observe(float("nan"))
    assert overlay.ruined is True
    assert overlay.preview_scale() == 0.0
    snap = overlay.snapshot()
    assert snap["ruined"] is True
    assert snap["research_only"] is True

    book = BookRiskOverlay(
        vol_target=0.10,
        dd_limit=0.05,
        lookback=8,
        kelly_lookback=8,
        kelly_fraction=0.25,
        crc_lookback=8,
        crc_alpha=0.05,
    )
    for nav in (100.0, 100.0, 100.0):
        book.observe(nav)
    warm = book.preview_scale()
    assert 0.0 < warm <= 1.0
    for nav in np.linspace(100.0, 70.0, 12):
        book.observe(float(nav))
    assert book.preview_scale() == 0.0

    calm = BookRiskOverlay(
        vol_target=0.02, dd_limit=0.5, lookback=8, kelly_fraction=0.0, crc_alpha=0.0
    )
    for nav in [100.0] * 12:
        calm.observe(nav)
    scale = calm.preview_scale()
    assert 0.0 <= scale <= 1.0

    broken = BookRiskOverlay(lookback=8, kelly_lookback=8, crc_lookback=8)
    broken.navs = [0.0, 0.0]
    assert broken.preview_scale() == 0.0
    broken.navs = [100.0, float("nan"), 100.0, 101.0, 99.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    measured = broken.preview_scale()
    assert 0.0 <= measured <= 1.0
    losing = BookRiskOverlay(lookback=8, kelly_lookback=8, kelly_fraction=0.25, crc_alpha=0.0)
    losing.navs = [100.0 - index for index in range(12)]
    assert losing.preview_scale() == 0.0
