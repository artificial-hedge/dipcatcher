"""Fail-closed branches in execution impact, costs and shortfall decomposition.

Every input here is SYNTHETIC. These tests pin *rejections*: non-finite
scalars, degenerate magnitudes, mismatched vectors and frames that would
otherwise surface a NaN cost or shortfall downstream.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from quant_fund.config.models import CostConfig
from quant_fund.execution.costs import (
    commission_cost,
    half_spread_cost,
    sqrt_impact,
    total_cost,
)
from quant_fund.execution.impact import (
    arrival_price_slippage,
    permanent_impact,
    perold_shortfall,
    pov_schedule,
    pow_law_total_impact,
    propagator_kernel,
    propagator_price_path,
    required_participation,
    sqrt_impact_bps,
    vwap_slippage,
)
from quant_fund.execution.implementation_shortfall import (
    aggregate_shortfall,
    fill_shortfall,
    shortfall_frame,
)

INF = float("inf")
NAN = float("nan")


# ---------------------------------------------------------------------------
# impact.py — scalar guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fraction_of_volume": NAN},
        {"fraction_of_volume": INF},
        {"fraction_of_volume": 0.1, "sigma": 0.0},
        {"fraction_of_volume": 0.1, "sigma": INF},
        {"fraction_of_volume": 0.1, "upsilon": 0.0},
        {"fraction_of_volume": -0.1},
    ],
)
def test_sqrt_impact_bps_rejects_nonfinite_and_degenerate_inputs(kwargs):
    base = {"fraction_of_volume": 0.01, "sigma": 0.02, "upsilon": 0.6}
    base.update(kwargs)
    with pytest.raises(ValueError):
        sqrt_impact_bps(**base)


@pytest.mark.parametrize(
    "quantity,daily_volume,sigma,gamma",
    [
        (NAN, 1e6, 0.02, 0.5),
        (-1.0, 1e6, 0.02, 0.5),
        (1e5, 0.0, 0.02, 0.5),
        (1e5, 1e6, 0.0, 0.5),
        (1e5, 1e6, 0.02, 0.0),
        (1e5, 1e6, INF, 0.5),
    ],
)
def test_permanent_impact_rejects_bad_inputs(quantity, daily_volume, sigma, gamma):
    with pytest.raises(ValueError):
        permanent_impact(quantity, daily_volume, sigma, gamma)


def test_propagator_kernel_rejects_nonfinite_tau_and_zero_beta():
    with pytest.raises(ValueError):
        propagator_kernel(np.array([0.0, NAN]))
    with pytest.raises(ValueError):
        propagator_kernel(np.array([0.0, 1.0]), beta=0.0)


def test_propagator_price_path_rejects_nonfinite_trades_and_scale():
    with pytest.raises(ValueError, match="signed_trades"):
        propagator_price_path(np.array([1.0, INF]))
    with pytest.raises(ValueError, match="impact_scale"):
        propagator_price_path(np.array([1.0]), impact_scale=NAN)
    with pytest.raises(ValueError, match="beta"):
        propagator_price_path(np.array([1.0]), beta=-0.5)


@pytest.mark.parametrize(
    "args",
    [
        (NAN, 1e6, 0.02, 0.5),
        (-1.0, 1e6, 0.02, 0.5),
        (1e4, 0.0, 0.02, 0.5),
        (1e4, 1e6, -0.02, 0.5),
        (1e4, 1e6, 0.02, INF),
    ],
)
def test_pow_law_total_impact_rejects_bad_inputs(args):
    with pytest.raises(ValueError):
        pow_law_total_impact(*args)


# ---------------------------------------------------------------------------
# impact.py — vector/schedule guards
# ---------------------------------------------------------------------------


def test_pov_schedule_rejects_zero_quantity_and_nonpositive_volume():
    with pytest.raises(ValueError, match="quantity"):
        pov_schedule(0.0, np.array([100.0]), 0.1)
    with pytest.raises(ValueError, match="volume_forecast"):
        pov_schedule(10.0, np.array([100.0, 0.0]), 0.1)
    with pytest.raises(ValueError, match="volume_forecast"):
        pov_schedule(10.0, np.array([]), 0.1)


@pytest.mark.parametrize("participation", [0.0, -0.1, 1.5, NAN])
def test_pov_schedule_rejects_out_of_range_participation(participation):
    with pytest.raises(ValueError, match="participation"):
        pov_schedule(10.0, np.array([100.0, 100.0]), participation)


def test_pov_schedule_truncates_the_last_touched_bar_and_zeroes_the_rest():
    schedule = pov_schedule(120.0, np.array([100.0, 100.0, 100.0]), 0.5)
    assert schedule.tolist() == [50.0, 50.0, 20.0]
    assert schedule.sum() == pytest.approx(120.0)


@pytest.mark.parametrize("func", [vwap_slippage, arrival_price_slippage])
def test_slippage_rejects_length_mismatch(func):
    with pytest.raises(ValueError, match="match length"):
        func(np.array([100.0, 101.0]), np.array([1.0]), 100.0)


@pytest.mark.parametrize("func", [vwap_slippage, arrival_price_slippage])
def test_slippage_rejects_nonpositive_trade_prices(func):
    with pytest.raises(ValueError, match="trade_prices"):
        func(np.array([100.0, 0.0]), np.array([1.0, 1.0]), 100.0)


@pytest.mark.parametrize("func", [vwap_slippage, arrival_price_slippage])
def test_slippage_rejects_zero_net_quantity(func):
    with pytest.raises(ValueError, match="non-zero"):
        func(np.array([100.0]), np.array([0.0]), 100.0)


def test_slippage_rejects_nonfinite_reference_price():
    with pytest.raises(ValueError, match="market_vwap"):
        vwap_slippage(np.array([100.0]), np.array([1.0]), NAN)
    with pytest.raises(ValueError, match="arrival_price"):
        arrival_price_slippage(np.array([100.0]), np.array([1.0]), 0.0)


# ---------------------------------------------------------------------------
# impact.py — Perold decomposition guards
# ---------------------------------------------------------------------------


def test_perold_rejects_negative_unfilled_quantity():
    with pytest.raises(ValueError, match="unfilled_qty"):
        perold_shortfall(100.0, np.array([101.0]), np.array([1.0]), -1.0, 101.0)


def test_perold_rejects_mismatched_vectors():
    with pytest.raises(ValueError, match="match length"):
        perold_shortfall(100.0, np.array([101.0, 102.0]), np.array([1.0]), 0.0, 101.0)


def test_perold_rejects_negative_fill_quantity():
    with pytest.raises(ValueError, match="exec_qty"):
        perold_shortfall(100.0, np.array([101.0]), np.array([-1.0]), 0.0, 101.0)


def test_perold_rejects_order_with_no_quantity_at_all():
    with pytest.raises(ValueError, match="positive total quantity"):
        perold_shortfall(100.0, np.array([101.0]), np.array([0.0]), 0.0, 101.0)


def test_perold_rejects_nonfinite_cancel_price_and_fees():
    with pytest.raises(ValueError, match="cancel_price"):
        perold_shortfall(100.0, np.array([101.0]), np.array([1.0]), 0.0, NAN)
    with pytest.raises(ValueError, match="fees"):
        perold_shortfall(100.0, np.array([101.0]), np.array([1.0]), 0.0, 101.0, fees=NAN)


def test_perold_unfilled_only_order_uses_decision_price_for_execution_leg():
    out = perold_shortfall(100.0, np.array([101.0]), np.array([0.0]), 10.0, 110.0)
    assert out["execution"] == 0.0
    assert out["opportunity"] == pytest.approx(0.10)
    assert out["fill_rate"] == 0.0


# ---------------------------------------------------------------------------
# impact.py — required participation guards
# ---------------------------------------------------------------------------


def test_required_participation_rejects_bad_horizon():
    v = np.full(5, 100.0)
    for max_bars in (0, -1, 6):
        with pytest.raises(ValueError, match="max_bars"):
            required_participation(100.0, v, max_bars=max_bars)


def test_required_participation_rejects_degenerate_inputs():
    with pytest.raises(ValueError, match="quantity"):
        required_participation(0.0, np.array([100.0]))
    with pytest.raises(ValueError, match="volume_forecast"):
        required_participation(1.0, np.array([100.0, -1.0]))


# ---------------------------------------------------------------------------
# costs.py
# ---------------------------------------------------------------------------


def test_half_spread_and_commission_reject_nonfinite_or_negative_rates():
    with pytest.raises(ValueError):
        half_spread_cost(NAN, 5.0)
    with pytest.raises(ValueError):
        half_spread_cost(1.0, -5.0)
    with pytest.raises(ValueError):
        commission_cost(INF, 1.0)
    with pytest.raises(ValueError):
        commission_cost(1.0, -1.0)


@pytest.mark.parametrize(
    "quantity,price,adv,sigma,upsilon",
    [
        (NAN, 100.0, 1e6, 0.02, 0.1),
        (1.0, 0.0, 1e6, 0.02, 0.1),
        (1.0, 100.0, 0.0, 0.02, 0.1),
        (1.0, 100.0, 1e6, 0.02, -0.1),
        (1.0, 100.0, 1e6, -0.02, 0.1),
    ],
)
def test_sqrt_impact_rejects_bad_inputs(quantity, price, adv, sigma, upsilon):
    with pytest.raises(ValueError):
        sqrt_impact(quantity, price, adv, sigma, upsilon)


def test_sqrt_impact_zero_quantity_is_exactly_free():
    assert sqrt_impact(0.0, 100.0, 1e6, 0.02, 0.1) == 0.0


def test_total_cost_rejects_bad_market_inputs():
    cfg = CostConfig()
    with pytest.raises(ValueError, match="quantity"):
        total_cost(1.0, 0.0, 1e6, 0.02, cfg)
    with pytest.raises(ValueError, match="adv_dollars"):
        total_cost(1.0, 100.0, -1.0, 0.02, cfg)
    with pytest.raises(ValueError, match="sigma"):
        total_cost(1.0, 100.0, 1e6, NAN, cfg)


def test_total_cost_rejects_negative_configured_rates():
    # The config schema forbids negatives, so model_copy bypasses validation to
    # prove total_cost re-checks its own inputs (defence in depth).
    bad = CostConfig().model_copy(update={"commission_bps": -1.0})
    with pytest.raises(ValueError, match="commission_bps"):
        total_cost(1.0, 100.0, 1e6, 0.02, bad)


def test_total_cost_rejects_negative_spread_override():
    with pytest.raises(ValueError, match="half_spread_bps"):
        total_cost(1.0, 100.0, 1e6, 0.02, CostConfig(), half_spread_bps=-1.0)


def test_total_cost_frictionless_is_labelled_and_zero():
    cfg = CostConfig(frictionless=True)
    out = total_cost(10.0, 100.0, 1e6, 0.02, cfg)
    assert out["label"] == "FRICTIONLESS RESEARCH ONLY"
    assert out["total"] == 0.0


def test_total_cost_components_match_building_blocks():
    cfg = CostConfig(commission_bps=1.0, half_spread_bps=5.0, impact_y=0.1)
    out = total_cost(10.0, 100.0, 1e6, 0.02, cfg)
    notional = 1000.0
    assert out["label"] == "costed"
    assert out["commission"] == pytest.approx(commission_cost(notional, 1.0))
    assert out["spread"] == pytest.approx(half_spread_cost(notional, 5.0))
    assert out["total"] == pytest.approx(
        out["commission"] + out["spread"] + out["impact"] + out["turnover_bps"]
    )


# ---------------------------------------------------------------------------
# implementation_shortfall.py — single fill guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quantity", [0.0, -1.0, NAN])
def test_fill_shortfall_rejects_bad_quantity(quantity):
    with pytest.raises(ValueError, match="quantity"):
        fill_shortfall(side_sign=1.0, quantity=quantity, decision_price=100.0, exec_price=101.0)


@pytest.mark.parametrize("price", [0.0, NAN])
def test_fill_shortfall_rejects_bad_prices(price):
    with pytest.raises(ValueError, match="prices"):
        fill_shortfall(side_sign=1.0, quantity=1.0, decision_price=price, exec_price=101.0)
    with pytest.raises(ValueError, match="prices"):
        fill_shortfall(side_sign=1.0, quantity=1.0, decision_price=100.0, exec_price=price)


@pytest.mark.parametrize("sign", [0.0, 2.0, NAN])
def test_fill_shortfall_rejects_ambiguous_side(sign):
    with pytest.raises(ValueError, match="side_sign"):
        fill_shortfall(side_sign=sign, quantity=1.0, decision_price=100.0, exec_price=101.0)


@pytest.mark.parametrize("field", ["fee", "spread_cost", "impact_cost"])
def test_fill_shortfall_rejects_negative_or_nonfinite_costs(field):
    with pytest.raises(ValueError, match=field):
        fill_shortfall(
            side_sign=1.0,
            quantity=1.0,
            decision_price=100.0,
            exec_price=101.0,
            **{field: -0.5},
        )


# ---------------------------------------------------------------------------
# implementation_shortfall.py — frame guards
# ---------------------------------------------------------------------------


def _frame(**overrides) -> pl.DataFrame:
    data = {
        "security_id": ["A", "A"],
        "quantity": [10.0, 10.0],
        "decision_price": [100.0, 100.0],
        "price": [101.0, 99.0],
    }
    data.update(overrides)
    return pl.DataFrame(data)


def test_shortfall_frame_requires_the_contract_columns():
    with pytest.raises(ValueError, match="missing columns"):
        shortfall_frame(pl.DataFrame({"security_id": ["A"]}))


@pytest.mark.parametrize("sign", [0.0, 2.0, NAN])
def test_shortfall_frame_rejects_invalid_side_sign(sign):
    frame = _frame(side_sign=[sign, 1.0])
    with pytest.raises(ValueError, match="side_sign must be"):
        shortfall_frame(frame)


def test_shortfall_frame_rejects_signed_quantity_with_side_sign():
    frame = _frame(side_sign=[1.0, -1.0], quantity=[10.0, -10.0])
    with pytest.raises(ValueError, match="unsigned"):
        shortfall_frame(frame)


@pytest.mark.parametrize(
    "overrides",
    [
        {"price": [101.0, 0.0]},
        {"price": [101.0, NAN]},
        {"decision_price": [100.0, 0.0]},
        {"quantity": [10.0, 0.0]},
        {"quantity": [10.0, NAN]},
    ],
)
def test_shortfall_frame_rejects_unscorable_rows(overrides):
    with pytest.raises(ValueError, match="finite"):
        shortfall_frame(_frame(**overrides))


def test_shortfall_frame_rejects_unparsable_cells():
    frame = _frame().with_columns(pl.Series("price", ["101.0", "junk"]))
    with pytest.raises(ValueError, match="finite"):
        shortfall_frame(frame)


@pytest.mark.parametrize("column", ["fee", "spread_cost", "impact_cost"])
def test_shortfall_frame_rejects_negative_explicit_cost_columns(column):
    with pytest.raises(ValueError, match=column):
        shortfall_frame(_frame(**{column: [0.0, -1.0]}))


def test_shortfall_frame_empty_input_yields_null_metrics():
    empty = _frame().head(0)
    out = shortfall_frame(empty)
    assert out.height == 0
    for column in ("drift", "explicit", "total_is", "notional", "is_bps", "drift_bps"):
        assert column in out.columns


def test_shortfall_frame_derives_side_sign_from_signed_quantity():
    frame = pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "quantity": [10.0, -10.0],
            "decision_price": [100.0, 100.0],
            "price": [101.0, 99.0],
        }
    )
    out = shortfall_frame(frame)
    # Both a buy that paid up and a sell that sold cheaper are adverse.
    assert out["drift"].to_list() == [10.0, 10.0]
    assert out["side_sign"].to_list() == [1.0, -1.0]


# ---------------------------------------------------------------------------
# implementation_shortfall.py — aggregate guards
# ---------------------------------------------------------------------------


def test_aggregate_shortfall_empty_frame_is_flat_and_labelled():
    out = aggregate_shortfall(_frame().head(0))
    assert out["n_fills"] == 0
    assert out["is_bps"] is None
    assert out["live_pnl_claim"] is False
    assert out["research_only"] is True


def test_aggregate_shortfall_rejects_unprepared_frames():
    with pytest.raises(ValueError, match="missing columns"):
        aggregate_shortfall(pl.DataFrame({"security_id": ["A"]}))


def test_aggregate_shortfall_is_bps_none_without_notional():
    frame = pl.DataFrame(
        {
            "drift": [1.0],
            "explicit": [0.0],
            "total_is": [1.0],
            "notional": [0.0],
            "side_sign": [1.0],
            "security_id": ["A"],
        }
    )
    out = aggregate_shortfall(frame)
    assert out["is_bps"] is None
    assert out["drift_bps"] is None
