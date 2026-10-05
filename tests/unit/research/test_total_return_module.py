"""Unit tests for research/total_return.py — the ``dipcatcher data total-return`` core.

Wired through a *lazy* import inside the CLI command body
(:func:`quant_fund.cli.data_cmds.total_return_cmd`), so a module-level import
scan reports it as unreachable; the CLI end-to-end path is covered by
``tests/unit/cli/test_reality_total_return_cmds.py``. These tests pin the
function-level contract directly: Yahoo chart ``events`` → corporate-action
rows, dividend reinvestment arithmetic on a hand-computed 4-bar panel, split
factors applied only for raw prints, and the fail-closed edges (an in-sample
ex-date that matches no bar, late ``available_time``, foreign ids, non-bool
flags).

The panels are synthetic fixtures built in-test — no vendor data, no network,
and no performance claim of any kind: this is price-basis plumbing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_fund.data.adapters.stooq import session_close
from quant_fund.data.adapters.yahoo_eod import REVISION, SOURCE
from quant_fund.research.total_return import (
    YAHOO_CHART_EVENTS,
    apply_research_total_return,
    parse_yahoo_corporate_actions,
)
from quant_fund.schemas.errors import PointInTimeError

_DAYS = (
    datetime(2026, 3, 2, tzinfo=UTC).date(),  # Monday
    datetime(2026, 3, 3, tzinfo=UTC).date(),
    datetime(2026, 3, 4, tzinfo=UTC).date(),
    datetime(2026, 3, 5, tzinfo=UTC).date(),
)


def _session(day_index: int, *, symbol: str = "AAA") -> datetime:
    suffix = ".uk" if symbol.endswith(".L") else ".us"
    return session_close(_DAYS[day_index], suffix)


def _bars(
    closes: list[float] | None = None,
    *,
    security_id: str = "AAA",
    days: tuple[object, ...] = _DAYS,
    volumes: list[float] | None = None,
) -> pl.DataFrame:
    """A quote-basis daily panel whose event times are real session closes."""
    times = [session_close(day, ".us") for day in days]  # type: ignore[arg-type]
    if closes is None:
        prices = [100.0 + 2.0 * i for i in range(len(times))]
    else:
        assert len(closes) == len(times), "explicit closes must match the day grid"
        prices = closes
    vols = volumes if volumes is not None else [1_000_000.0] * len(times)
    return pl.DataFrame(
        {
            "security_id": [security_id] * len(times),
            "event_time": times,
            "open": [p - 0.5 for p in prices],
            "high": [p + 1.0 for p in prices],
            "low": [p - 1.0 for p in prices],
            "close": prices,
            "volume": vols,
        }
    )


def _chart(
    *,
    dividends: dict[str, object] | None = None,
    splits: dict[str, object] | None = None,
) -> dict[str, object]:
    """A Yahoo chart payload whose ``events`` block carries the given rows."""
    events: dict[str, object] = {}
    if dividends is not None:
        events["dividends"] = dividends
    if splits is not None:
        events["splits"] = splits
    block: dict[str, object] = {}
    if events:
        block["events"] = events
    return {"chart": {"result": [block]}}


def _dividend(day_index: int, amount: float) -> dict[str, object]:
    stamp = int(_session(day_index).timestamp())
    return {str(stamp): {"amount": amount, "date": stamp}}


def _split(day_index: int, numerator: float, denominator: float) -> dict[str, object]:
    stamp = int(_session(day_index).timestamp())
    return {str(stamp): {"numerator": numerator, "denominator": denominator, "date": stamp}}


def _actions(payload: dict[str, object], *, security_id: str = "AAA") -> pl.DataFrame:
    return parse_yahoo_corporate_actions(payload, security_id=security_id, yahoo_symbol=security_id)


# ---------------------------------------------------------------------------
# parse_yahoo_corporate_actions
# ---------------------------------------------------------------------------


def test_chart_events_option_requests_dividends_and_splits() -> None:
    assert YAHOO_CHART_EVENTS == "div,split"


def test_dividend_event_becomes_one_cash_dividend_row() -> None:
    actions = _actions(_chart(dividends=_dividend(2, 1.50)))
    assert actions.height == 1
    row = actions.row(0, named=True)
    assert row["security_id"] == "AAA"
    assert row["action_type"] == "cash_dividend"
    assert row["amount"] == pytest.approx(1.50)
    assert row["factor"] is None
    # The ex-date resolves to the same session close the bars use, and the
    # action is knowable at that close (never later, never earlier).
    assert row["event_time"] == _session(2)
    assert row["available_time"] == row["event_time"]
    assert row["source"] == SOURCE
    assert row["revision_id"] == REVISION


def test_split_event_becomes_one_split_row_with_a_factor() -> None:
    actions = _actions(_chart(splits=_split(1, 2.0, 1.0)))
    assert actions.height == 1
    row = actions.row(0, named=True)
    assert row["action_type"] == "split"
    assert row["factor"] == pytest.approx(2.0)
    assert row["amount"] is None
    assert row["event_time"] == _session(1)


def test_split_factor_accepts_string_numerator_and_denominator() -> None:
    stamp = int(_session(0).timestamp())
    payload = _chart(splits={str(stamp): {"numerator": "3", "denominator": "2", "date": stamp}})
    actions = _actions(payload)
    assert actions["factor"][0] == pytest.approx(1.5)


def test_dividends_and_splits_are_sorted_by_ex_date_then_type() -> None:
    payload = _chart(dividends=_dividend(3, 0.40), splits=_split(1, 4.0, 1.0))
    actions = _actions(payload)
    assert actions.height == 2
    assert actions["action_type"].to_list() == ["split", "cash_dividend"]
    assert actions["event_time"].to_list() == [_session(1), _session(3)]


def test_two_dividends_on_one_session_are_summed_into_one_row() -> None:
    stamp = int(_session(2).timestamp())
    payload = _chart(
        dividends={
            str(stamp): {"amount": 1.00, "date": stamp},
            f"{stamp}-b": {"amount": 0.25, "date": stamp},
        }
    )
    actions = _actions(payload)
    assert actions.height == 1
    assert actions["amount"][0] == pytest.approx(1.25)


def test_events_may_arrive_as_a_list_with_an_inline_date() -> None:
    stamp = int(_session(1).timestamp())
    payload = _chart(dividends=[{"amount": 0.75, "date": stamp}])
    actions = _actions(payload)
    assert actions.height == 1
    assert actions["amount"][0] == pytest.approx(0.75)
    assert actions["event_time"][0] == _session(1)


def test_london_symbol_uses_the_uk_session_close() -> None:
    stamp = int(datetime(2026, 3, 4, 12, 0, tzinfo=UTC).timestamp())
    payload = _chart(dividends={str(stamp): {"amount": 0.10, "date": stamp}})
    actions = parse_yahoo_corporate_actions(payload, security_id="BBB", yahoo_symbol="BBB.L")
    assert actions["event_time"][0] == session_close(_DAYS[2], ".uk")
    assert actions["event_time"][0] != _session(2)


def test_payload_without_a_chart_block_yields_an_empty_action_frame() -> None:
    for payload in ({}, {"chart": {}}, {"chart": {"result": []}}, {"chart": {"result": [{}]}}):
        actions = _actions(payload)
        assert actions.is_empty()
        assert actions.columns == [
            "security_id",
            "event_time",
            "available_time",
            "action_type",
            "amount",
            "factor",
            "source",
            "revision_id",
        ]


def test_payload_without_an_events_key_yields_no_actions() -> None:
    assert _actions({"chart": {"result": [{"timestamp": [1, 2, 3]}]}}).is_empty()


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"chart": {"result": [{"events": "nope"}]}}, "events is not an object"),
        (_chart(dividends="nope"), "dividend is not a map or list"),
        (_chart(dividends=[{"amount": 1.0, "date": "x"}]), "not a number"),
        (_chart(dividends=[{"amount": "abc", "date": 1_700_000_000}]), "not a number"),
        (_chart(dividends=[{"amount": -1.0, "date": 1_700_000_000}]), "must be positive"),
        (_chart(dividends=[{"amount": 0.0, "date": 1_700_000_000}]), "must be positive"),
        (_chart(dividends=[{"date": 1_700_000_000}]), "not a number"),
        (_chart(dividends=[{"amount": 1.0}]), "missing a timestamp"),
        (_chart(dividends=[{"amount": 1.0, "date": 1_700_000_000.5}]), "integer second"),
        (_chart(dividends=[{"amount": 1.0, "date": 0}]), "outside the seconds range"),
        (_chart(dividends=[{"amount": 1.0, "date": 99_999_999_999}]), "seconds range"),
        (_chart(dividends=[{"amount": True, "date": 1_700_000_000}]), "not a number"),
        (_chart(splits=[{"numerator": 2.0, "date": 1_700_000_000}]), "not a number"),
        (
            _chart(splits=[{"numerator": 0.0, "denominator": 1.0, "date": 1_700_000_000}]),
            "positive",
        ),
        (_chart(dividends=[{"amount": 1.0, "date": float("inf")}]), "not finite"),
    ],
)
def test_parser_fails_closed_on_malformed_events(payload: dict[str, object], match: str) -> None:
    with pytest.raises(PointInTimeError, match=match):
        _actions(payload)


def test_two_splits_on_one_session_fail_closed() -> None:
    stamp = int(_session(1).timestamp())
    payload = _chart(
        splits={
            str(stamp): {"numerator": 2.0, "denominator": 1.0, "date": stamp},
            f"{stamp}-b": {"numerator": 3.0, "denominator": 1.0, "date": stamp},
        }
    )
    with pytest.raises(PointInTimeError, match="duplicate yahoo split"):
        _actions(payload)


@pytest.mark.parametrize("security_id", ["", "   "])
@pytest.mark.parametrize("yahoo_symbol", ["AAA", " "])
def test_blank_identifiers_fail_closed(security_id: str, yahoo_symbol: str) -> None:
    with pytest.raises(PointInTimeError, match="non-blank"):
        parse_yahoo_corporate_actions(
            _chart(dividends=_dividend(1, 1.0)),
            security_id=security_id,
            yahoo_symbol=yahoo_symbol,
        )


# ---------------------------------------------------------------------------
# apply_research_total_return — hand-computed dividend reinvestment
# ---------------------------------------------------------------------------


def test_dividend_reinvestment_matches_hand_computed_arithmetic() -> None:
    """Closes 100/102/104/106 with a 2.00 cash dividend ex bar 2.

    Total-return close at the ex-date = prior close × (1 + price return +
    dividend / prior close) = 102 × (1 + 2/102 + 2/102) = 104 + 2 = 106, i.e.
    the quote close plus the reinvested cash. The next bar compounds on that
    basis: 106 × 106/104. Bars before the ex-date are untouched.
    """
    frame = _bars()
    adjusted = apply_research_total_return(frame, _actions(_chart(dividends=_dividend(2, 2.00))))
    ordered = adjusted.sort("event_time")

    quote = [100.0, 102.0, 104.0, 106.0]
    close = ordered["close"].to_list()
    assert close[0] == pytest.approx(quote[0])
    assert close[1] == pytest.approx(quote[1])
    assert close[2] == pytest.approx(106.0)
    assert close[3] == pytest.approx(106.0 * 106.0 / 104.0)
    assert ordered["close_quote"].to_list() == pytest.approx(quote)
    assert ordered["return_basis"].to_list() == ["total_return"] * 4


def test_dividend_scale_applies_to_the_whole_ohlc_row_and_not_to_volume() -> None:
    """No split in the sample → the split-adjusted basis equals the quote
    basis, so ``_tr_scale`` is the only factor and volume is unchanged."""
    frame = _bars()
    adjusted = apply_research_total_return(frame, _actions(_chart(dividends=_dividend(2, 2.00))))
    ordered = adjusted.sort("event_time")
    scale = 106.0 / 104.0

    assert ordered["open_quote"].to_list() == pytest.approx([99.5, 101.5, 103.5, 105.5])
    assert ordered["open"][2] == pytest.approx(103.5 * scale)
    assert ordered["high"][2] == pytest.approx(105.0 * scale)
    assert ordered["low"][2] == pytest.approx(103.0 * scale)
    assert ordered["low_quote"][2] == pytest.approx(103.0)
    # Pre-ex-date rows carry scale 1.0.
    assert ordered["open"][0] == pytest.approx(ordered["open_quote"][0])
    assert ordered["high"][1] == pytest.approx(ordered["high_quote"][1])
    assert ordered["volume"].to_list() == pytest.approx([1_000_000.0] * 4)
    assert ordered["volume_quote"].to_list() == pytest.approx([1_000_000.0] * 4)


def test_no_actions_is_a_basis_relabel_not_a_price_change() -> None:
    frame = _bars()
    adjusted = apply_research_total_return(frame, _actions(_chart()))
    assert adjusted["close"].to_list() == pytest.approx(frame["close"].to_list())
    assert adjusted["close_quote"].to_list() == pytest.approx(frame["close"].to_list())
    assert set(adjusted["return_basis"].to_list()) == {"total_return"}


def test_dividend_outside_the_sample_is_dropped_not_applied() -> None:
    """On/before the first bar or after the last bar → outside the sample."""
    frame = _bars()
    before = apply_research_total_return(frame, _actions(_chart(dividends=_dividend(0, 5.0))))
    assert before["close"].to_list() == pytest.approx(frame["close"].to_list())

    after_day = _DAYS[-1] + timedelta(days=1)
    stamp = int(session_close(after_day, ".us").timestamp())
    payload = _chart(dividends={str(stamp): {"amount": 5.0, "date": stamp}})
    after = apply_research_total_return(frame, _actions(payload))
    assert after["close"].to_list() == pytest.approx(frame["close"].to_list())


def test_split_factor_is_applied_for_raw_prints_only() -> None:
    """Raw closes 100/102 then a 2-for-1 → 52/53. With ``raw_prices`` the
    pre-split bars halve onto the post-split basis (50/51/52/53) and share
    volume doubles so dollar volume is preserved."""
    frame = _bars([100.0, 102.0, 52.0, 53.0])
    actions = _actions(_chart(splits=_split(2, 2.0, 1.0)))
    adjusted = apply_research_total_return(frame, actions, prices_already_split_adjusted=False)
    ordered = adjusted.sort("event_time")

    assert ordered["close"].to_list() == pytest.approx([50.0, 51.0, 52.0, 53.0])
    assert ordered["close_quote"].to_list() == pytest.approx([100.0, 102.0, 52.0, 53.0])
    assert ordered["open"][0] == pytest.approx(49.75)
    assert ordered["open_quote"][0] == pytest.approx(99.5)
    assert ordered["volume"].to_list() == pytest.approx([2e6, 2e6, 1e6, 1e6])
    assert ordered["volume_quote"].to_list() == pytest.approx([1e6] * 4)


def test_split_rows_are_ignored_when_prices_are_already_adjusted() -> None:
    """The default must not apply the factor twice."""
    frame = _bars([100.0, 102.0, 52.0, 53.0])
    actions = _actions(_chart(splits=_split(2, 2.0, 1.0)))
    adjusted = apply_research_total_return(frame, actions)
    assert adjusted["close"].to_list() == pytest.approx([100.0, 102.0, 52.0, 53.0])
    assert adjusted["close_quote"].to_list() == pytest.approx([100.0, 102.0, 52.0, 53.0])
    assert adjusted["volume"].to_list() == pytest.approx([1e6] * 4)


def test_dividend_and_split_compose_on_raw_prints() -> None:
    """Raw 100/102 → 2-for-1 → 52/53 with a 1.00 cash dividend ex bar 2.

    Split-adjusted closes are 50/51/52/53, so the ex-date total-return close is
    51 × (1 + 1/51 + 1/51) = 52 + 1 = 53.
    """
    frame = _bars([100.0, 102.0, 52.0, 53.0])
    payload = _chart(dividends=_dividend(2, 1.00), splits=_split(2, 2.0, 1.0))
    adjusted = apply_research_total_return(
        frame, _actions(payload), prices_already_split_adjusted=False
    )
    ordered = adjusted.sort("event_time")
    close = ordered["close"].to_list()
    assert close[0] == pytest.approx(50.0)
    assert close[1] == pytest.approx(51.0)
    assert close[2] == pytest.approx(53.0)
    assert close[3] == pytest.approx(53.0 * 53.0 / 52.0)


def test_adjustment_is_per_security() -> None:
    """Only the id that paid the dividend moves; the other name is untouched."""
    aaa = _bars(security_id="AAA")
    bbb = _bars([50.0, 51.0, 52.0, 53.0], security_id="BBB")
    frame = pl.concat([aaa, bbb])
    actions = _actions(_chart(dividends=_dividend(2, 2.00)), security_id="AAA")
    adjusted = apply_research_total_return(frame, actions)

    got_aaa = adjusted.filter(pl.col("security_id") == "AAA").sort("event_time")
    got_bbb = adjusted.filter(pl.col("security_id") == "BBB").sort("event_time")
    assert got_aaa["close"][2] == pytest.approx(106.0)
    assert got_bbb["close"].to_list() == pytest.approx([50.0, 51.0, 52.0, 53.0])
    assert got_bbb["close"].to_list() == pytest.approx(got_bbb["close_quote"].to_list())


def test_unsorted_input_is_ordered_by_security_and_time() -> None:
    frame = _bars().sort("event_time", descending=True)
    adjusted = apply_research_total_return(frame, _actions(_chart(dividends=_dividend(2, 2.00))))
    assert adjusted["event_time"].to_list() == sorted(adjusted["event_time"].to_list())
    assert adjusted.sort("event_time")["close"][2] == pytest.approx(106.0)


# ---------------------------------------------------------------------------
# Fail-closed edges
# ---------------------------------------------------------------------------


def test_in_sample_ex_date_that_matches_no_bar_fails_closed() -> None:
    """Bar 2 is missing from the panel: the ex-date is strictly inside the
    sample, so the dividend cannot be placed and must not be dropped."""
    frame = _bars(days=(_DAYS[0], _DAYS[1], _DAYS[3]))
    with pytest.raises(PointInTimeError, match="does not match a research bar"):
        apply_research_total_return(frame, _actions(_chart(dividends=_dividend(2, 2.00))))


def test_action_for_a_foreign_security_id_fails_closed() -> None:
    frame = _bars(security_id="AAA")
    actions = _actions(_chart(dividends=_dividend(2, 2.00)), security_id="ZZZ")
    with pytest.raises(PointInTimeError, match="absent from the research bars"):
        apply_research_total_return(frame, actions)


def test_late_available_time_fails_closed() -> None:
    frame = _bars()
    actions = _actions(_chart(dividends=_dividend(2, 2.00))).with_columns(
        (pl.col("event_time") + timedelta(days=1)).alias("available_time")
    )
    with pytest.raises(PointInTimeError, match="available_time must be at or before"):
        apply_research_total_return(frame, actions)


def test_null_available_time_fails_closed() -> None:
    frame = _bars()
    actions = _actions(_chart(dividends=_dividend(2, 2.00))).with_columns(
        pl.lit(None, dtype=pl.Datetime(time_zone="UTC")).alias("available_time")
    )
    with pytest.raises(PointInTimeError, match="available_time"):
        apply_research_total_return(frame, actions)


def test_already_adjusted_bars_are_rejected() -> None:
    frame = _bars()
    actions = _actions(_chart(dividends=_dividend(2, 2.00)))
    adjusted = apply_research_total_return(frame, actions)
    with pytest.raises(PointInTimeError, match="already total-return adjusted"):
        apply_research_total_return(adjusted, actions)


def test_missing_required_bar_columns_fail_closed() -> None:
    frame = _bars().drop("volume")
    with pytest.raises(PointInTimeError, match="missing required columns"):
        apply_research_total_return(frame, pl.DataFrame(schema={"action_type": pl.String()}))


def test_duplicate_bars_fail_closed() -> None:
    frame = pl.concat([_bars(), _bars()])
    with pytest.raises(PointInTimeError, match="duplicate security_id/event_time"):
        apply_research_total_return(frame, _actions(_chart()))


def test_empty_bars_pass_through_only_without_actions() -> None:
    empty = _bars().head(0)
    assert apply_research_total_return(empty, _actions(_chart())).is_empty()
    with pytest.raises(PointInTimeError, match="empty bar frame"):
        apply_research_total_return(empty, _actions(_chart(dividends=_dividend(1, 1.0))))


@pytest.mark.parametrize("flag", [1, 0, "yes", None])
def test_non_bool_split_adjusted_flag_fails_closed(flag: object) -> None:
    frame = _bars()
    with pytest.raises(PointInTimeError, match="must be a bool"):
        apply_research_total_return(
            frame,
            _actions(_chart()),
            prices_already_split_adjusted=flag,  # type: ignore[arg-type]
        )


def test_null_and_missing_action_type_fail_closed() -> None:
    frame = _bars()
    no_type = pl.DataFrame(
        {
            "security_id": ["AAA"],
            "event_time": [_session(2)],
            "amount": [1.0],
        }
    )
    with pytest.raises(PointInTimeError, match="missing action_type"):
        apply_research_total_return(frame, no_type)

    null_type = pl.DataFrame(
        {
            "security_id": ["AAA"],
            "event_time": [_session(2)],
            "action_type": [None],
            "amount": [1.0],
        }
    )
    with pytest.raises(PointInTimeError, match="null action_type"):
        apply_research_total_return(frame, null_type)


def test_unknown_action_types_are_dropped_not_applied() -> None:
    """A delist row is not a price-basis action: it must be ignored, keeping
    the quote basis intact rather than being silently treated as a dividend."""
    frame = _bars()
    actions = pl.DataFrame(
        {
            "security_id": ["AAA"],
            "event_time": [_session(2)],
            "available_time": [_session(2)],
            "action_type": ["delist"],
            "amount": [None],
            "factor": [None],
            "source": [SOURCE],
            "revision_id": [REVISION],
        }
    )
    adjusted = apply_research_total_return(frame, actions)
    assert adjusted["close"].to_list() == pytest.approx(frame["close"].to_list())


def test_non_positive_close_fails_closed_inside_adjust_prices() -> None:
    frame = _bars([100.0, 0.0, 104.0, 106.0])
    with pytest.raises(PointInTimeError, match="non-positive close"):
        apply_research_total_return(frame, _actions(_chart(dividends=_dividend(2, 1.0))))
