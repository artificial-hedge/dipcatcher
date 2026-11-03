"""NYSE regular-session clock and price bands for the live gateway.

``continuous`` calendars do not consult this module. Session bounds come
from ``UsEquitiesCalendar``, so holidays, 13:00 early closes, and DST are
the same table the research calendars use. A band breach is not a mark.
"""

from __future__ import annotations

import math
from datetime import datetime
from zoneinfo import ZoneInfo

from quant_fund.calendars.us_equities import ET, XNYS

_UTC = ZoneInfo("UTC")


class ClockError(ValueError):
    """A stored band or quote flag cannot be read."""


def nyse_rth_closed(ts_ns: int) -> bool:
    """True when ``ts_ns`` is outside the NYSE regular or early-close session.

    The open is inclusive and the close is exclusive, matching the pre-trade
    kernel. Years the holiday table refuses are treated as closed.
    """
    if ts_ns < 0:
        return True
    instant = datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=_UTC)
    local = instant.astimezone(ET)
    try:
        session = XNYS.session_on(local.date())
    except ValueError:
        return True
    if session is None:
        return True
    return instant < session.open_utc or instant >= session.close_utc


def outside_band(price: float, band: tuple[float, float] | None) -> bool:
    """True when a finite band is installed and ``price`` is strictly outside it.

    Equality with either edge is still inside the band. A missing band does
    not constrain the price.
    """
    if band is None:
        return False
    lower, upper = band
    return price < lower or price > upper


def move_entry[V](mapping: dict[str, V], old: str, new: str) -> None:
    """Rename one key. A missing key is left alone."""
    if old in mapping:
        mapping[new] = mapping.pop(old)


def flag_map(raw: object) -> dict[str, bool]:
    """Restore quote-accept flags. A missing payload is an empty map."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ClockError("quote flags are malformed")
    return {str(key): bool(value) for key, value in raw.items()}


def band_map(raw: object) -> dict[str, tuple[float, float]]:
    """Restore price bands. A missing payload is an empty map."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ClockError("price bands are malformed")
    bands: dict[str, tuple[float, float]] = {}
    for symbol, item in raw.items():
        if not isinstance(item, dict):
            raise ClockError("price band is malformed")
        lower = item.get("lower")
        upper = item.get("upper")
        if isinstance(lower, bool) or isinstance(upper, bool):
            raise ClockError("price band is malformed")
        if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
            raise ClockError("price band is malformed")
        lo = float(lower)
        hi = float(upper)
        if not math.isfinite(lo) or not math.isfinite(hi) or lo <= 0.0 or hi < lo:
            raise ClockError("price band is malformed")
        bands[str(symbol)] = (lo, hi)
    return bands
