"""Tests for the ``scripts/arb_*.py`` experiment helpers.

The arb lane is a family of research scripts that all share the same
selection protocol: slice a carry book into dev/holdout windows, score a
candidate configuration, then screen the funding cross-section down to a
name set. None of those helpers had any test coverage.

What is pinned here:

- ``_slice`` — the dev/holdout window filter (half-open ``[lo, hi)``), in all
  three spellings used across the family (ISO-string coercing, plain, and the
  lambda form).
- ``_score`` — the selection score, including the ``n < 30`` rejection floor
  and the liquidation / margin-reject penalties.
- ``_topk_mass`` / ``_topk_sharpe`` / ``_topk_cnt`` — universe screens.
- ``_pctile_funding`` — within-day cross-sectional percentile ranking.
- ``_z_funding`` / ``_z_funding_hybrid`` — per-name trailing z-scores.
- ``_snipe_funding`` — lead/hold funding event synthesis.
- ``_basis_z_funding`` — negative basis z-score.
- ``_features`` — the ML spread feature builder.
- ``_screen_names`` — the dev-computed eligibility screens.
- ``_pair_filter`` — the venue-direction regex filter.

Every frame is synthetic and in-memory. No artifact is written and no market
data is read.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _load(name: str):
    import importlib

    return importlib.import_module(name)


arb_1h = _load("arb_1h_grid")
arb_asym = _load("arb_asym_enter")
arb_basis = _load("arb_basis_fade")
arb_elig = _load("arb_elig_screen")
arb_lev = _load("arb_lev_profile")
arb_mass = _load("arb_mass_tilt")
arb_ml = _load("arb_ml_spread")
arb_pair = _load("arb_pair_decomp")
arb_pctile = _load("arb_pctile_entry")
arb_screen = _load("arb_screen_sizing")
arb_sharpe5 = _load("arb_sharpe5_grid")
arb_snipe = _load("arb_snipe_1h")
arb_zscore = _load("arb_zscore_spread")

# Every script in the family that exposes ``_score``.
_SCORE_MODULES = [
    arb_1h,
    arb_asym,
    arb_basis,
    arb_elig,
    arb_mass,
    arb_ml,
    arb_pair,
    arb_pctile,
    arb_screen,
    arb_sharpe5,
    arb_snipe,
    arb_zscore,
]

# ``_slice`` implementations, grouped by the two shapes they take.
_SLICE_ISO_MODULES = [arb_1h, arb_ml, arb_sharpe5, arb_snipe, arb_zscore, arb_basis, arb_lev]
_SLICE_PLAIN_MODULES = [arb_screen, arb_asym, arb_elig, arb_pair]
_SLICE_LAMBDA_MODULES = [arb_pctile, arb_mass]

DEV = datetime(2024, 1, 1, tzinfo=UTC)
SPLIT = datetime(2025, 1, 1, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _bars(sid: str, times: list[datetime], close: float = 100.0) -> list[dict]:
    return [
        {
            "event_time": t,
            "security_id": sid,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
            "source": "test",
        }
        for t in times
    ]


@pytest.fixture
def book() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """A tiny perp/spot/funding book straddling the dev/holdout split."""
    times = [
        datetime(2023, 12, 31, tzinfo=UTC),
        DEV,
        datetime(2024, 6, 1, tzinfo=UTC),
        SPLIT,
        datetime(2025, 6, 1, tzinfo=UTC),
    ]
    perp = pl.DataFrame(_bars("BTC", times))
    spot = pl.DataFrame(_bars("BTC", times, close=99.0))
    fund = pl.DataFrame([{"event_time": t, "security_id": "BTC", "value": 0.001} for t in times])
    return perp, spot, fund


# --------------------------------------------------------------------------- #
# _slice
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mod", _SLICE_ISO_MODULES, ids=lambda m: m.__name__)
def test_slice_accepts_iso_strings(mod, book) -> None:
    perp, spot, fund = book
    p, s, f = mod._slice(perp, spot, fund, "2024-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00")
    assert p.height == 2
    assert s.height == 2
    assert f.height == 2


@pytest.mark.parametrize(
    "mod", _SLICE_PLAIN_MODULES + _SLICE_LAMBDA_MODULES, ids=lambda m: m.__name__
)
def test_slice_accepts_datetimes(mod, book) -> None:
    perp, spot, fund = book
    p, s, f = mod._slice(perp, spot, fund, DEV, SPLIT)
    assert p.height == 2
    assert s.height == 2
    assert f.height == 2


@pytest.mark.parametrize("mod", _SLICE_ISO_MODULES, ids=lambda m: m.__name__)
def test_slice_is_half_open_on_upper_bound(mod, book) -> None:
    perp, spot, fund = book
    p, _, _ = mod._slice(perp, spot, fund, DEV, SPLIT)
    times = p["event_time"].to_list()
    assert SPLIT not in times
    assert all(t >= DEV for t in times)


@pytest.mark.parametrize("mod", _SLICE_ISO_MODULES, ids=lambda m: m.__name__)
def test_slice_includes_lower_bound(mod, book) -> None:
    perp, spot, fund = book
    p, _, _ = mod._slice(perp, spot, fund, DEV, SPLIT)
    assert DEV in p["event_time"].to_list()


@pytest.mark.parametrize("mod", _SLICE_ISO_MODULES, ids=lambda m: m.__name__)
def test_slice_empty_window_returns_empty_frames(mod, book) -> None:
    perp, spot, fund = book
    p, s, f = mod._slice(
        perp, spot, fund, datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 6, 1, tzinfo=UTC)
    )
    assert p.height == 0
    assert s.height == 0
    assert f.height == 0


@pytest.mark.parametrize("mod", _SLICE_ISO_MODULES, ids=lambda m: m.__name__)
def test_slice_preserves_schema_on_empty_result(mod, book) -> None:
    perp, spot, fund = book
    p, _, _ = mod._slice(
        perp, spot, fund, datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 6, 1, tzinfo=UTC)
    )
    assert p.schema["event_time"] == pl.Datetime("us", UTC)
    assert p.schema["security_id"] == pl.String


# --------------------------------------------------------------------------- #
# _score
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mod", _SCORE_MODULES, ids=lambda m: m.__name__)
def test_score_rejects_small_samples(mod) -> None:
    assert mod._score({"n": 29, "sharpe": 5.0}) == -999.0


@pytest.mark.parametrize("mod", _SCORE_MODULES, ids=lambda m: m.__name__)
def test_score_accepts_minimum_sample(mod) -> None:
    assert mod._score({"n": 30, "sharpe": 1.0}) == 1.0


@pytest.mark.parametrize("mod", _SCORE_MODULES, ids=lambda m: m.__name__)
def test_score_rejects_empty_metrics(mod) -> None:
    assert mod._score({}) == -999.0


@pytest.mark.parametrize("mod", _SCORE_MODULES, ids=lambda m: m.__name__)
def test_score_penalizes_liquidations(mod) -> None:
    assert mod._score({"n": 100, "sharpe": 2.0, "liquidation_count": 4}) == 0.0


@pytest.mark.parametrize("mod", _SCORE_MODULES, ids=lambda m: m.__name__)
def test_score_penalizes_margin_rejects(mod) -> None:
    assert mod._score({"n": 100, "sharpe": 2.0, "margin_rejects": 100}) == pytest.approx(1.9)


@pytest.mark.parametrize("mod", _SCORE_MODULES, ids=lambda m: m.__name__)
def test_score_penalty_is_additive(mod) -> None:
    got = mod._score({"n": 100, "sharpe": 2.0, "liquidation_count": 2, "margin_rejects": 500})
    assert got == pytest.approx(2.0 - 1.0 - 0.5)


@pytest.mark.parametrize("mod", _SCORE_MODULES, ids=lambda m: m.__name__)
def test_score_treats_missing_sharpe_as_floor(mod) -> None:
    assert mod._score({"n": 100}) == -999.0


@pytest.mark.parametrize("mod", _SCORE_MODULES, ids=lambda m: m.__name__)
def test_score_treats_zero_sharpe_as_floor(mod) -> None:
    # ``or -999.0`` — a zero Sharpe is indistinguishable from a missing one.
    assert mod._score({"n": 100, "sharpe": 0.0}) == -999.0


@pytest.mark.parametrize("mod", _SCORE_MODULES, ids=lambda m: m.__name__)
def test_score_is_monotone_in_sharpe(mod) -> None:
    assert mod._score({"n": 100, "sharpe": 3.0}) > mod._score({"n": 100, "sharpe": 1.0})


# --------------------------------------------------------------------------- #
# _topk_mass / _topk_sharpe / _topk_cnt
# --------------------------------------------------------------------------- #


@pytest.fixture
def funding_cross_section() -> pl.DataFrame:
    """Three names with strictly separated masses and above-bar counts.

    The counts are deliberately distinct (3 / 1 / 0 events above the 1e-3
    bar) so that ``head(k)`` selections never have to break a tie — polars
    does not guarantee a stable order among equal keys, and a tie would make
    these assertions flaky.
    """
    rows = []
    for sid, values in (
        ("AX:h>b:BTC", [0.003, 0.002, 0.003]),
        ("AX:h>b:ETH", [0.002, 0.001, 0.001]),
        ("AX:b>h:SOL", [0.0005, 0.0005, 0.0005]),
    ):
        for value in values:
            rows.append(
                {
                    "event_time": datetime(2024, 6, 1, tzinfo=UTC),
                    "security_id": sid,
                    "value": value,
                }
            )
    return pl.DataFrame(rows)


def test_topk_mass_ranks_by_total(funding_cross_section) -> None:
    got = arb_screen._topk_mass(funding_cross_section, 2)
    assert got == {"AX:h>b:BTC", "AX:h>b:ETH"}


def test_topk_mass_k_larger_than_universe(funding_cross_section) -> None:
    assert len(arb_screen._topk_mass(funding_cross_section, 99)) == 3


def test_topk_mass_k_zero_returns_empty(funding_cross_section) -> None:
    assert arb_screen._topk_mass(funding_cross_section, 0) == set()


def test_topk_sharpe_ranks_by_mean_over_std() -> None:
    # Y has the lower mean but a far tighter spread, so it wins on Sharpe.
    frame = pl.DataFrame(
        [
            {"event_time": datetime(2024, 6, 1, tzinfo=UTC), "security_id": "X", "value": 0.10},
            {"event_time": datetime(2024, 6, 2, tzinfo=UTC), "security_id": "X", "value": 0.05},
            {"event_time": datetime(2024, 6, 1, tzinfo=UTC), "security_id": "Y", "value": 0.06},
            {"event_time": datetime(2024, 6, 2, tzinfo=UTC), "security_id": "Y", "value": 0.04},
        ]
    )
    assert arb_screen._topk_sharpe(frame, 1) == {"Y"}


def test_topk_sharpe_ignores_total_mass() -> None:
    # X has the largest total (0.20 vs 0.10) but the lower dispersion-adjusted
    # score, so the Sharpe screen is not a proxy for the mass screen.
    frame = pl.DataFrame(
        [
            {"event_time": datetime(2024, 6, 1, tzinfo=UTC), "security_id": "X", "value": 0.12},
            {"event_time": datetime(2024, 6, 2, tzinfo=UTC), "security_id": "X", "value": 0.08},
            {"event_time": datetime(2024, 6, 1, tzinfo=UTC), "security_id": "Y", "value": 0.055},
            {"event_time": datetime(2024, 6, 2, tzinfo=UTC), "security_id": "Y", "value": 0.045},
        ]
    )
    assert arb_screen._topk_sharpe(frame, 1) == {"Y"}
    assert arb_screen._topk_mass(frame, 1) == {"X"}


def test_topk_mass_filters_by_venue_direction(funding_cross_section) -> None:
    got = arb_asym._topk_mass(funding_cross_section, 10, ("h>b",))
    assert got == {"AX:h>b:BTC", "AX:h>b:ETH"}


def test_topk_mass_without_direction_keeps_everything(funding_cross_section) -> None:
    got = arb_asym._topk_mass(funding_cross_section, 10, ())
    assert got == {"AX:h>b:BTC", "AX:h>b:ETH", "AX:b>h:SOL"}


def test_topk_cnt_counts_events_above_bar(funding_cross_section) -> None:
    # BTC clears the bar 3 times, ETH once, SOL never.
    got = arb_pctile._topk_cnt(funding_cross_section, 2)
    assert got == {"AX:h>b:BTC", "AX:h>b:ETH"}


def test_topk_cnt_k_zero_returns_empty(funding_cross_section) -> None:
    assert arb_pctile._topk_cnt(funding_cross_section, 0) == set()


# --------------------------------------------------------------------------- #
# _pctile_funding
# --------------------------------------------------------------------------- #


def test_pctile_funding_ranks_within_day() -> None:
    frame = pl.DataFrame(
        [
            {"event_time": datetime(2024, 6, 1, tzinfo=UTC), "security_id": "A", "value": 1.0},
            {"event_time": datetime(2024, 6, 1, tzinfo=UTC), "security_id": "B", "value": 2.0},
            {"event_time": datetime(2024, 6, 1, tzinfo=UTC), "security_id": "C", "value": 3.0},
        ]
    )
    out = arb_pctile._pctile_funding(frame)
    values = dict(zip(out["security_id"].to_list(), out["value"].to_list(), strict=True))
    assert values["A"] == pytest.approx(1 / 3)
    assert values["B"] == pytest.approx(2 / 3)
    assert values["C"] == pytest.approx(1.0)


def test_pctile_funding_is_per_day_not_global() -> None:
    frame = pl.DataFrame(
        [
            {"event_time": datetime(2024, 6, 1, tzinfo=UTC), "security_id": "A", "value": 1.0},
            {"event_time": datetime(2024, 6, 1, tzinfo=UTC), "security_id": "B", "value": 2.0},
            {"event_time": datetime(2024, 6, 2, tzinfo=UTC), "security_id": "A", "value": 1.0},
        ]
    )
    out = arb_pctile._pctile_funding(frame)
    # A on day 2 is the only name that day -> percentile 1.0, not 1/3.
    day2 = out.filter(pl.col("event_time") == datetime(2024, 6, 2, tzinfo=UTC))
    assert day2["value"].to_list() == [1.0]


def test_pctile_funding_sorts_by_time_then_sid() -> None:
    frame = pl.DataFrame(
        [
            {"event_time": datetime(2024, 6, 2, tzinfo=UTC), "security_id": "B", "value": 1.0},
            {"event_time": datetime(2024, 6, 1, tzinfo=UTC), "security_id": "Z", "value": 1.0},
            {"event_time": datetime(2024, 6, 1, tzinfo=UTC), "security_id": "A", "value": 2.0},
        ]
    )
    out = arb_pctile._pctile_funding(frame)
    assert out["security_id"].to_list() == ["A", "Z", "B"]


def test_pctile_funding_keeps_columns() -> None:
    frame = pl.DataFrame(
        [
            {"event_time": datetime(2024, 6, 1, tzinfo=UTC), "security_id": "A", "value": 1.0},
        ]
    )
    assert arb_pctile._pctile_funding(frame).columns == ["event_time", "security_id", "value"]


# --------------------------------------------------------------------------- #
# _z_funding / _z_funding_hybrid
# --------------------------------------------------------------------------- #


def _fund_series(values: list[float]) -> pl.DataFrame:
    base = datetime(2024, 6, 1, tzinfo=UTC)
    return pl.DataFrame(
        [
            {
                "event_time": base + timedelta(days=i),
                "security_id": "BTC",
                "value": v,
            }
            for i, v in enumerate(values)
        ]
    )


def test_z_funding_sign_flips_the_z_score() -> None:
    frame = _fund_series([0.001] * 20 + [0.010])
    pos = arb_zscore._z_funding(frame, 10, sign=1.0)
    neg = arb_zscore._z_funding(frame, 10, sign=-1.0)
    last_pos = pos["value"].to_list()[-1]
    last_neg = neg["value"].to_list()[-1]
    assert last_pos > 0
    assert last_neg == pytest.approx(-last_pos)


def test_z_funding_drops_null_lookback_rows() -> None:
    frame = _fund_series([0.001] * 30)
    out = arb_zscore._z_funding(frame, 20)
    assert out.height < frame.height


def test_z_funding_of_flat_series_is_zero() -> None:
    frame = _fund_series([0.002] * 40)
    out = arb_zscore._z_funding(frame, 20)
    assert out.height > 0
    assert all(abs(v) < 1e-6 for v in out["value"].to_list())


def test_z_funding_keeps_only_three_columns() -> None:
    frame = _fund_series([0.001] * 30 + [0.01])
    assert arb_zscore._z_funding(frame, 10).columns == ["event_time", "security_id", "value"]


def test_z_funding_hybrid_zeroes_below_threshold() -> None:
    frame = _fund_series([0.001] * 30 + [0.010])
    out = arb_zscore._z_funding_hybrid(frame, 20, ze=5.0)
    assert out["value"].to_list()[-1] == 0.0


def test_z_funding_hybrid_keeps_spike_above_threshold() -> None:
    frame = _fund_series([0.001] * 30 + [0.010])
    out = arb_zscore._z_funding_hybrid(frame, 20, ze=1.0)
    assert out["value"].to_list()[-1] == pytest.approx(0.010)


def test_z_funding_hybrid_zeroes_the_calm_stretch() -> None:
    frame = _fund_series([0.001] * 40)
    out = arb_zscore._z_funding_hybrid(frame, 20, ze=5.0)
    assert set(out["value"].to_list()) == {0.0}


# --------------------------------------------------------------------------- #
# _snipe_funding
# --------------------------------------------------------------------------- #


def test_snipe_funding_emits_entry_and_exit_rows() -> None:
    frame = pl.DataFrame(
        [
            {
                "event_time": datetime(2024, 6, 1, 8, tzinfo=UTC),
                "security_id": "BTC",
                "value": 0.001,
            }
        ]
    )
    out = arb_snipe._snipe_funding(frame, lead_bars=1, hold_bars=2)
    assert out.height == 2
    times = out["event_time"].to_list()
    assert datetime(2024, 6, 1, 7, tzinfo=UTC) in times
    assert datetime(2024, 6, 1, 10, tzinfo=UTC) in times


def test_snipe_funding_exit_row_is_negative_one() -> None:
    frame = pl.DataFrame(
        [
            {
                "event_time": datetime(2024, 6, 1, 8, tzinfo=UTC),
                "security_id": "BTC",
                "value": 0.001,
            }
        ]
    )
    out = arb_snipe._snipe_funding(frame, lead_bars=1, hold_bars=2)
    values = out["value"].to_list()
    assert -1.0 in values
    assert 0.001 in values


def test_snipe_funding_ignores_non_positive_events() -> None:
    frame = pl.DataFrame(
        [
            {
                "event_time": datetime(2024, 6, 1, 8, tzinfo=UTC),
                "security_id": "BTC",
                "value": -0.001,
            }
        ]
    )
    assert arb_snipe._snipe_funding(frame, lead_bars=1, hold_bars=2).height == 0


def test_snipe_funding_sorts_by_sid_then_time() -> None:
    frame = pl.DataFrame(
        [
            {
                "event_time": datetime(2024, 6, 1, 8, tzinfo=UTC),
                "security_id": "BTC",
                "value": 0.001,
            },
            {
                "event_time": datetime(2024, 6, 1, 8, tzinfo=UTC),
                "security_id": "ETH",
                "value": 0.001,
            },
        ]
    )
    out = arb_snipe._snipe_funding(frame, lead_bars=1, hold_bars=2)
    assert out["security_id"].to_list() == ["BTC", "BTC", "ETH", "ETH"]


# --------------------------------------------------------------------------- #
# _basis_z_funding
# --------------------------------------------------------------------------- #


def test_basis_z_funding_is_negative_of_basis_z() -> None:
    perp = pl.DataFrame(
        [
            {
                "event_time": datetime(2024, 6, 1 + i, tzinfo=UTC),
                "security_id": "BTC",
                "close": 100.0 if i < 20 else 90.0,
            }
            for i in range(21)
        ]
    )
    spot = pl.DataFrame(
        [
            {
                "event_time": datetime(2024, 6, 1 + i, tzinfo=UTC),
                "security_id": "BTC",
                "close": 100.0,
            }
            for i in range(21)
        ]
    )
    out = arb_basis._basis_z_funding(perp, spot, lookback=10)
    # A depressed basis (perp below spot) maps to a positive weight signal.
    assert out["value"].to_list()[-1] > 0


def test_basis_z_funding_only_keeps_inner_join_rows() -> None:
    perp = pl.DataFrame(
        [
            {
                "event_time": datetime(2024, 6, 1, tzinfo=UTC),
                "security_id": "BTC",
                "close": 100.0,
            }
        ]
    )
    spot = pl.DataFrame(
        [
            {
                "event_time": datetime(2024, 6, 2, tzinfo=UTC),
                "security_id": "BTC",
                "close": 100.0,
            }
        ]
    )
    assert arb_basis._basis_z_funding(perp, spot, lookback=5).height == 0


def test_basis_z_funding_keeps_only_three_columns() -> None:
    perp = pl.DataFrame(
        [
            {
                "event_time": datetime(2024, 6, 1 + i, tzinfo=UTC),
                "security_id": "BTC",
                "close": 100.0,
            }
            for i in range(15)
        ]
    )
    spot = perp.with_columns(pl.lit(100.0).alias("close"))
    assert arb_basis._basis_z_funding(perp, spot, lookback=10).columns == [
        "event_time",
        "security_id",
        "value",
    ]


def test_basis_z_funding_requires_lookback_of_at_least_ten() -> None:
    """``min_samples=max(10, lookback // 3)`` must not exceed the window size.

    A lookback below 10 makes the rolling window's minimum sample count larger
    than the window itself, which polars rejects. Pinned so the constraint is
    visible rather than a runtime surprise on a mis-sized grid.
    """
    perp = pl.DataFrame(
        [
            {
                "event_time": datetime(2024, 6, 1 + i, tzinfo=UTC),
                "security_id": "BTC",
                "close": 100.0,
            }
            for i in range(15)
        ]
    )
    spot = perp.with_columns(pl.lit(100.0).alias("close"))
    with pytest.raises(pl.exceptions.InvalidOperationError):
        arb_basis._basis_z_funding(perp, spot, lookback=3)


# --------------------------------------------------------------------------- #
# _features
# --------------------------------------------------------------------------- #


def _ml_fund(n: int = 40) -> pl.DataFrame:
    base = datetime(2024, 6, 1, tzinfo=UTC)
    return pl.DataFrame(
        [
            {
                "event_time": base + timedelta(days=i),
                "security_id": "BTC",
                "value": 0.001 * (i % 5),
            }
            for i in range(n)
        ]
    )


def _ml_perp(n: int = 40) -> pl.DataFrame:
    base = datetime(2024, 6, 1, tzinfo=UTC)
    return pl.DataFrame(
        [
            {
                "event_time": base + timedelta(days=i),
                "security_id": "BTC",
                "close": 100.0 + i,
            }
            for i in range(n)
        ]
    )


def test_features_produce_expected_columns() -> None:
    out = arb_ml._features(_ml_perp(), _ml_fund())
    for col in ("s_lag1", "s_ma3", "s_ma9", "s_sd9", "s_hi9", "target", "ret24", "vol24"):
        assert col in out.columns


def test_features_drop_first_row_per_sid() -> None:
    out = arb_ml._features(_ml_perp(), _ml_fund(40))
    assert out.height == 39
    assert out["s_lag1"].null_count() == 0


def test_features_lag_is_the_previous_value() -> None:
    fund = _ml_fund(10)
    out = arb_ml._features(_ml_perp(10), fund)
    assert out["s_lag1"].to_list()[0] == pytest.approx(0.0)
    assert out["s_lag1"].to_list()[1] == pytest.approx(0.001)


def test_features_target_is_the_next_value() -> None:
    fund = _ml_fund(10)
    out = arb_ml._features(_ml_perp(10), fund)
    # Row 0 is dropped (no lag), so the first surviving row is source row 1,
    # whose target is source row 2 -> 0.002.
    assert out["target"].to_list()[0] == pytest.approx(0.002)
    assert out["target"].to_list()[-1] is None


def test_features_ret24_is_log_return() -> None:
    out = arb_ml._features(_ml_perp(10), _ml_fund(10))
    # close goes 100 -> 101 -> 102, so the first log return is log(101/100).
    assert out["ret24"].to_list()[0] == pytest.approx(0.00995, abs=1e-4)


def test_features_sorts_by_sid_and_time() -> None:
    out = arb_ml._features(_ml_perp(10), _ml_fund(10))
    assert out["security_id"].to_list() == ["BTC"] * 9
    times = out["event_time"].to_list()
    assert times == sorted(times)


# --------------------------------------------------------------------------- #
# _screen_names
# --------------------------------------------------------------------------- #


@pytest.fixture
def screen_frame() -> pl.DataFrame:
    base = datetime(2024, 6, 1, tzinfo=UTC)
    rows = []
    for i in range(10):
        rows.append(
            {
                "event_time": base + timedelta(days=i),
                "security_id": "HIGH",
                "value": 0.005,
            }
        )
    for i in range(10):
        rows.append(
            {
                "event_time": base + timedelta(days=i),
                "security_id": "LOW",
                "value": 0.0001,
            }
        )
    return pl.DataFrame(rows)


def test_screen_names_topk_mass(screen_frame) -> None:
    assert arb_elig._screen_names(screen_frame, "topk_mass", 1) == {"HIGH"}


def test_screen_names_topk_mean(screen_frame) -> None:
    assert arb_elig._screen_names(screen_frame, "topk_mean", 1) == {"HIGH"}


def test_screen_names_topk_cnt(screen_frame) -> None:
    assert arb_elig._screen_names(screen_frame, "topk_cnt", 1) == {"HIGH"}


def test_screen_names_frac_uses_fraction_above_bar(screen_frame) -> None:
    # HIGH is always above 1e-3, LOW never is.
    assert arb_elig._screen_names(screen_frame, "frac", 0.5) == {"HIGH"}
    assert arb_elig._screen_names(screen_frame, "frac", 1.1) == set()


def test_screen_names_median_q_selects_upper_half(screen_frame) -> None:
    got = arb_elig._screen_names(screen_frame, "median_q", 0.5)
    assert "HIGH" in got


def test_screen_names_unknown_kind_raises(screen_frame) -> None:
    with pytest.raises(ValueError, match="bogus"):
        arb_elig._screen_names(screen_frame, "bogus", 1.0)


# --------------------------------------------------------------------------- #
# _pair_filter
# --------------------------------------------------------------------------- #


def test_pair_filter_matches_direction_prefix() -> None:
    expr = arb_pair._pair_filter(("h>b",))
    frame = pl.DataFrame(
        [
            {"security_id": "AX:h>b:BTC"},
            {"security_id": "AX:b>h:ETH"},
            {"security_id": "BTC"},
        ]
    )
    assert frame.filter(expr)["security_id"].to_list() == ["AX:h>b:BTC"]


def test_pair_filter_matches_multiple_directions() -> None:
    expr = arb_pair._pair_filter(("h>b", "b>h"))
    frame = pl.DataFrame(
        [
            {"security_id": "AX:h>b:BTC"},
            {"security_id": "AX:b>h:ETH"},
            {"security_id": "AX:x>y:SOL"},
        ]
    )
    assert frame.filter(expr).height == 2


def test_pair_filter_escapes_regex_metacharacters() -> None:
    expr = arb_pair._pair_filter(("h>b.c",))
    frame = pl.DataFrame(
        [
            {"security_id": "AX:h>b.c:BTC"},
            {"security_id": "AX:h>bc:BTC"},
        ]
    )
    assert frame.filter(expr)["security_id"].to_list() == ["AX:h>b.c:BTC"]


def test_pair_filter_requires_the_prefix_anchor() -> None:
    expr = arb_pair._pair_filter(("h>b",))
    frame = pl.DataFrame([{"security_id": "XXAX:h>b:BTC"}])
    assert frame.filter(expr).height == 0


# --------------------------------------------------------------------------- #
# Module-level constants
# --------------------------------------------------------------------------- #


def test_dev_split_constants_are_utc() -> None:
    for mod in (arb_asym, arb_elig, arb_mass, arb_pair, arb_screen, arb_sharpe5):
        assert mod.DEV_END.tzinfo is not None
        assert mod.DEV_MID.tzinfo is not None
        assert mod.DEV_MID < mod.DEV_END


def test_ml_feature_tuple_matches_builder() -> None:
    out = arb_ml._features(_ml_perp(10), _ml_fund(10))
    assert set(arb_ml.FEATURES).issubset(set(out.columns))


def test_screens_catalog_is_well_formed() -> None:
    kinds = {kind for kind, _ in arb_elig.SCREENS}
    assert "none" in kinds
    assert {"median_q", "frac", "topk_mass", "topk_mean", "topk_cnt"}.issubset(kinds)
