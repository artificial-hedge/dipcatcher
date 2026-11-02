"""Bounded parser properties for the offline OHLCV ingestion adapters."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from quant_fund.data.adapters import dolthub_stocks as dolthub
from quant_fund.data.adapters import hf_ohlcv_1m as hf
from tests.property._profiles import adversarial_settings

_SYMBOL_PARSERS = pytest.mark.parametrize("parser", [hf.parse_symbols, dolthub.parse_symbols])
_BOUND_PARSERS = pytest.mark.parametrize(
    ("parser", "error"),
    [(hf.parse_bound, hf.OhlcvQualityError), (dolthub.parse_bound, ValueError)],
)
_TICKER = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-",
    min_size=1,
    max_size=10,
)
_OFFSETS = st.integers(min_value=-12 * 60, max_value=14 * 60).map(
    lambda minutes: timezone(timedelta(minutes=minutes))
)
_AWARE_DATETIMES = st.datetimes(
    min_value=datetime(2000, 1, 1),
    max_value=datetime(2099, 12, 31),
    timezones=_OFFSETS,
)
_DAYS = st.dates(min_value=date(2000, 1, 1), max_value=date(2099, 12, 31))


@_SYMBOL_PARSERS
@given(
    parts=st.lists(st.one_of(_TICKER, st.just("")), max_size=8),
    form=st.sampled_from(["string", "list", "tuple"]),
)
@adversarial_settings()
def test_parse_symbols_normalizes_accepted_forms_without_reordering(
    parser: object, parts: list[str], form: str
) -> None:
    decorated = [f" \t{part.swapcase()}  " for part in parts]
    value: str | list[str] | tuple[str, ...]
    if form == "string":
        value = ",".join(decorated)
    elif form == "tuple":
        value = tuple(decorated)
    else:
        value = decorated
    expected = list(dict.fromkeys(part.upper() for part in parts if part))
    assert parser(value) == expected
    assert parser(expected) == expected


@_SYMBOL_PARSERS
@given(
    symbol=_TICKER,
    unsafe=st.sampled_from(["/", "\\", " ", "\t", "\n"]),
    form=st.sampled_from(["string", "list", "tuple"]),
)
@adversarial_settings()
def test_parse_symbols_rejects_embedded_unsafe_characters(
    parser: object, symbol: str, unsafe: str, form: str
) -> None:
    part = f"{symbol}{unsafe}{symbol}"
    value: str | list[str] | tuple[str, ...]
    if form == "string":
        value = part
    elif form == "tuple":
        value = (part,)
    else:
        value = [part]
    with pytest.raises(ValueError, match="unsafe ticker"):
        parser(value)


@pytest.mark.parametrize("value", [1, 1.5, True, b"AAPL", {"AAPL"}, {"A": "B"}])
@_SYMBOL_PARSERS
def test_parse_symbols_rejects_invalid_container_types(parser: object, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        parser(value)


@pytest.mark.parametrize("value", [[None], [7], ("AAPL", None), (object(),)])
@_SYMBOL_PARSERS
def test_parse_symbols_rejects_non_string_sequence_items(parser: object, value: object) -> None:
    with pytest.raises(ValueError, match="strings"):
        parser(value)


@given(day=_DAYS, role=st.sampled_from(["start", "end"]))
@adversarial_settings()
def test_hf_calendar_bounds_use_inclusive_new_york_session_days(day: date, role: str) -> None:
    if role == "start":
        local = datetime(day.year, day.month, day.day, tzinfo=hf.ET)
    else:
        following = day + timedelta(days=1)
        local = datetime(following.year, following.month, following.day, tzinfo=hf.ET)
        local -= timedelta(microseconds=1)
    expected = local.astimezone(UTC)
    assert hf.parse_bound(day, role=role) == expected
    assert hf.parse_bound(day.isoformat(), role=role) == expected


@given(day=_DAYS)
@adversarial_settings()
def test_dolthub_calendar_bounds_accept_dates_and_iso_strings(day: date) -> None:
    assert dolthub.parse_bound(day, role="start") == day
    assert dolthub.parse_bound(day.isoformat(), role="start") == day


@given(
    value=_AWARE_DATETIMES,
    separator=st.sampled_from(["T", " "]),
    zulu=st.booleans(),
)
@adversarial_settings()
def test_hf_aware_bounds_and_parse_aware_normalize_to_utc(
    value: datetime, separator: str, zulu: bool
) -> None:
    text_value = value.astimezone(UTC) if zulu else value
    text = text_value.isoformat(sep=separator)
    if zulu:
        text = text[:-6] + "Z"
    expected = value.astimezone(UTC)
    assert hf._parse_aware(text, role="start") == value
    assert hf.parse_bound(value, role="start") == expected
    assert hf.parse_bound(text, role="start") == expected


@given(
    value=_AWARE_DATETIMES,
    separator=st.sampled_from(["T", " "]),
    zulu=st.booleans(),
)
@example(
    value=datetime(2024, 1, 2, 23, 30, tzinfo=timezone(timedelta(hours=-5))),
    separator=" ",
    zulu=False,
)
@adversarial_settings()
def test_dolthub_aware_bounds_use_the_utc_calendar_date(
    value: datetime, separator: str, zulu: bool
) -> None:
    text_value = value.astimezone(UTC) if zulu else value
    text = text_value.isoformat(sep=separator)
    if zulu:
        text = text[:-6] + "Z"
    expected = value.astimezone(UTC).date()
    assert dolthub.parse_bound(value, role="start") == expected
    assert dolthub.parse_bound(text, role="start") == expected


@given(
    value=st.datetimes(
        min_value=datetime(2000, 1, 1),
        max_value=datetime(2099, 12, 31),
    ),
    separator=st.sampled_from(["T", " "]),
)
@adversarial_settings()
def test_hf_naive_datetime_strings_fail_closed(value: datetime, separator: str) -> None:
    with pytest.raises(hf.OhlcvQualityError, match="timezone-aware"):
        hf.parse_bound(value.isoformat(sep=separator), role="start")


@_BOUND_PARSERS
@given(
    day=_DAYS,
    suffix=st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789_/.;",
        min_size=1,
        max_size=12,
    ),
)
@adversarial_settings()
def test_parse_bound_rejects_valid_date_prefixes_with_garbage_suffixes(
    parser: object, error: type[Exception], day: date, suffix: str
) -> None:
    with pytest.raises(error):
        parser(f"{day.isoformat()}{suffix}", role="start")


@pytest.mark.parametrize("value", [None, True, 20240102, 1.5, b"2024-01-02", object()])
@_BOUND_PARSERS
def test_parse_bound_rejects_invalid_types(
    parser: object, error: type[Exception], value: object
) -> None:
    with pytest.raises(error):
        parser(value, role="start")
