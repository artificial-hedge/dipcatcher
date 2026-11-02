"""Coverage: GateSpec/GateResult and the fail-closed branches of the gate stack."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.risk.gates import (
    GateResult,
    GateSpec,
    apply_gate_stack,
    crash_leverage,
    crc_leverage,
    dd_halt,
    dd_remaining_leverage,
    es_leverage,
    kelly_leverage,
    stepm_leverage,
    vol_target,
    vol_target_leverage,
)


def _spec(**overrides: object) -> GateSpec:
    """GateSpec with every sizing gate disabled except the one under test."""
    base: dict[str, object] = {
        "vol_target": None,
        "dd_limit": None,
        "es_limit": None,
        "kelly_fraction": None,
        "crc_alpha": None,
        "crash_lookback": 0,
        "stepm_enable": False,
    }
    base.update(overrides)
    return GateSpec(**base)  # type: ignore[arg-type]


def test_snapshot_defaults_for_empty_result() -> None:
    snap = GateResult(
        returns=np.zeros(0), scale=np.zeros(0), spec=GateSpec(), n_halt=2, n_scaled=3
    ).snapshot()
    assert snap["mean_scale"] == 1.0
    assert snap["last_scale"] == 1.0
    assert snap["n_halt"] == 2
    assert snap["n_scaled"] == 3
    assert snap["research_only"] is True
    assert snap["live_pnl_claim"] is False
    assert snap["sharpe_scale_invariant_for_constant_leverage"] is True


def test_snapshot_reports_notes_and_scale_stats() -> None:
    result = GateResult(
        returns=np.zeros(3), scale=np.array([1.0, 0.5, 0.0]), spec=GateSpec(), notes=["x"]
    )
    snap = result.snapshot()
    assert snap["mean_scale"] == pytest.approx(0.5)
    assert snap["last_scale"] == 0.0
    assert snap["notes"] == ["x"]


def test_vol_target_rejects_non_positive_or_non_finite_target() -> None:
    for target in (0.0, -0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="vol target must be finite and positive"):
            vol_target(np.zeros(10), target=target)


def test_vol_target_skips_degenerate_windows_and_floors_lookback() -> None:
    # Constant path -> zero volatility -> every bar stays flat (no size).
    out = vol_target(np.zeros(6), lookback=1)
    assert out.shape == (6,)
    assert np.all(out == 0.0)


def test_vol_target_clips_to_cap() -> None:
    r = np.full(40, 1e-4)
    r[10] = 0.5
    out = vol_target(r, target=0.025, lookback=5, cap=2.0)
    assert np.all(np.abs(out) <= 2.0 * np.abs(r) + 1e-15)


def test_vol_target_leverage_uses_prior_on_degenerate_or_nonfinite_windows() -> None:
    lev = vol_target_leverage(np.zeros(6), target=0.025, lookback=2)
    assert np.allclose(lev, 0.125)

    r = np.ones(10)
    r[4] = np.nan
    lev_nan = vol_target_leverage(r, target=0.025, lookback=3)
    assert np.isfinite(lev_nan).all()


def test_dd_halt_stays_cash_after_breach() -> None:
    out = dd_halt(np.array([-0.06, 0.10, 0.10]), limit=0.05)
    assert out[0] == pytest.approx(-0.06)
    assert np.all(out[1:] == 0.0)


def test_dd_halt_passes_a_monotone_winner() -> None:
    r = np.array([0.01, 0.02, -0.005])
    assert np.allclose(dd_halt(r, limit=0.05), r)


def test_dd_remaining_leverage_halts_on_breach_and_nonfinite_returns() -> None:
    lev = dd_remaining_leverage(np.full(10, -0.01), limit=0.05, cushion=0.005)
    assert lev[-1] == 0.0

    # A non-finite print halts the rest of the path. It does not stay fully levered.
    lev_nan = dd_remaining_leverage(np.array([np.nan, 0.0, 0.0]), limit=0.05)
    assert np.all(lev_nan == 0.0)


def test_es_leverage_fails_closed_when_tail_is_unmeasurable() -> None:
    lev = es_leverage(np.full(10, np.nan), lookback=8)
    assert np.all(lev[7:] == 0.0)
    assert np.all(lev[:7] == 1.0)


def test_es_leverage_keeps_full_size_when_es_is_non_positive() -> None:
    lev = es_leverage(np.full(10, 0.01), lookback=8)
    assert np.allclose(lev, 1.0)


def test_es_leverage_scales_down_on_a_fat_tail() -> None:
    r = np.full(20, 0.0005)
    r[-1] = -0.5
    lev = es_leverage(r, es_limit=0.006, lookback=10)
    assert lev[-1] < 1.0


def test_kelly_leverage_rejects_bad_fraction() -> None:
    for fraction in (float("nan"), -0.1):
        with pytest.raises(ValueError, match="kelly fraction must be finite and non-negative"):
            kelly_leverage(np.zeros(10), fraction=fraction)


def test_kelly_leverage_zero_variance_and_negative_mean_halt() -> None:
    assert np.all(kelly_leverage(np.full(10, 0.01), lookback=8)[8:] == 0.0)
    assert np.all(kelly_leverage(np.full(10, -0.01), lookback=8)[8:] == 0.0)


def test_kelly_leverage_caps_the_growth_optimum() -> None:
    # Low-but-nonzero variance keeps f* large enough to hit the wealth cap.
    r = np.array([0.0100, 0.0101] * 5)
    lev = kelly_leverage(r, fraction=0.25, lookback=8, cap=0.5)
    assert lev[-1] == pytest.approx(0.5)


def test_crc_leverage_unmeasurable_tail_drops_the_bound() -> None:
    lev = crc_leverage(np.full(12, np.nan), alpha=0.05, lookback=8, step=5)
    assert np.all(lev[7:] == 0.0)


def test_crc_leverage_keeps_full_size_when_es_is_non_positive() -> None:
    lev = crc_leverage(np.full(12, 0.01), alpha=0.05, lookback=8, step=1)
    assert np.all(lev[7:] == 1.0)


def test_crc_leverage_recomputes_only_every_step() -> None:
    r = np.full(20, 0.001)
    r[5] = -0.4
    lev = crc_leverage(r, alpha=0.05, lookback=8, step=5)
    # Bars 7..11 hold the value computed at bar 7, then it is refreshed.
    assert np.allclose(lev[7:12], lev[7])
    assert lev[-1] <= 1.0


def test_crash_leverage_ruined_anchor_halts() -> None:
    lev = crash_leverage(np.full(15, -1.0), lookback=10)
    assert np.all(lev[10:] == 0.0)


def test_crash_leverage_numerically_degenerate_anchor_halts() -> None:
    lev = crash_leverage(np.full(25, -0.5), lookback=10)
    assert np.all(lev[10:] == 0.0)


def test_crash_leverage_ignores_nonfinite_returns() -> None:
    r = np.full(15, 0.01)
    r[0] = np.nan
    lev = crash_leverage(r, lookback=10)
    assert np.all(np.isfinite(lev))


def test_crash_leverage_floors_lookback_to_one() -> None:
    lev = crash_leverage(np.array([-0.25, -0.25]), lookback=0)
    assert lev[1] == 0.0


def test_stepm_leverage_empty_input() -> None:
    lev = stepm_leverage(np.zeros(0))
    assert lev.shape == (0,)


def test_stepm_leverage_no_rejection_sizes_zero_and_shifts() -> None:
    lev = stepm_leverage(np.zeros(20), min_obs=10, step=5, n_boot=8)
    assert lev.shape == (20,)
    assert np.all(lev == 0.0)


def test_stepm_leverage_single_row_stays_zero() -> None:
    lev = stepm_leverage(np.zeros((1, 1)), min_obs=10, step=5, n_boot=8)
    assert np.all(lev == 0.0)


def test_apply_gate_stack_empty_returns() -> None:
    result = apply_gate_stack(np.zeros(0))
    assert result.n_halt == 0
    assert result.n_scaled == 0
    assert result.returns.size == 0
    assert result.scale.size == 0


def test_apply_gate_stack_drawdown_remaining_mode() -> None:
    r = np.full(30, -0.002)
    spec = _spec(dd_limit=0.05, dd_mode="remaining")
    result = apply_gate_stack(r, spec)
    assert "dd=remaining-budget" in result.notes
    assert result.n_halt >= 1
    assert result.scale[-1] == 0.0


def test_apply_gate_stack_drawdown_halt_mode_handles_zero_returns() -> None:
    r = np.array([0.0, 0.01, 0.0, -0.10, 0.01])
    result = apply_gate_stack(r, _spec(dd_limit=0.05, dd_mode="halt"))
    assert "dd=halt-and-stay-cash" in result.notes
    # Deliberate cash bars (zero return) count as halted scale, not sizing.
    assert result.n_halt >= 1
    assert result.scale[-1] == 0.0
    assert result.returns[-1] == 0.0


def test_apply_gate_stack_stepm_uses_excess_argument() -> None:
    r = np.zeros(20)
    spec = _spec(stepm_enable=True, stepm_min_obs=10, stepm_step=5, stepm_n_boot=8)
    result = apply_gate_stack(r, spec, stepm_excess=np.zeros(20))
    assert "stepm=Romano–Wolf 2005 allow/deny" in result.notes
    assert np.all(result.scale == 0.0)


def test_apply_gate_stack_vol_and_kelly_notes_and_counts() -> None:
    r = np.full(30, 0.001)
    spec = GateSpec(
        vol_target=0.001,
        vol_lookback=5,
        kelly_fraction=0.25,
        kelly_lookback=8,
        crc_alpha=0.05,
        crc_lookback=8,
        crash_lookback=0,
        dd_limit=None,
        es_limit=None,
    )
    result = apply_gate_stack(r, spec)
    assert "vol_target=Moreira–Muir 2017" in result.notes
    assert "kelly=Thorp fractional" in result.notes
    assert "crc=Angelopoulos et al. 2022" in result.notes
    assert result.n_scaled > 0
    assert np.all(result.scale <= 1.0)


def test_apply_gate_stack_es_and_crash_notes() -> None:
    r = np.full(30, 0.0005)
    r[-1] = -0.3
    spec = _spec(es_limit=0.006, es_lookback=10, crash_lookback=5, crash_return=-0.2)
    result = apply_gate_stack(r, spec)
    assert "es_halt=Acerbi–Tasche ES" in result.notes
    assert "crash=nautica −20%/10d analog" in result.notes
    assert result.n_halt >= 1
