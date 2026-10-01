"""Cross-venue funding/basis bench (ULTRAPLAN P5.5).

Measures whether the same asset's spot-vs-futures basis and realized funding
rate differ across venues — the cross-venue carry differential. Each venue
leg supplies a daily spot frame, a daily futures/perp mark frame, and
(optionally) a funding-rate history; the bench inner-joins every pair of
venues on calendar date and reports the basis differential
``basis_a - basis_b`` and the funding differential on daily-summed realized
rates.

Reachability evidence on this box (2026-09-29): Binance is geo-blocked
(HTTP 451) and Bybit is geo-blocked (HTTP 403 via CloudFront); OKX public
REST resolves — ``okx_spot`` / ``okx_mark`` / ``okx_funding`` /
``okx_swap_universe`` adapters — alongside Kraken (``kraken_spot`` /
``kraken_futures_mark`` / ``kraken_funding`` added for P5.5). Funding
cadences differ by venue (Kraken perpetuals accrue hourly, OKX settles
every 8h), so the funding differential is computed on per-calendar-day
sums of realized rates — the honest common denominator — and partial-
coverage days are counted rather than hidden.

Output is descriptive statistics only — never P&L.
``write_crossvenue_basis_receipt`` seals hash-stamped JSON evidence under
``receipts/`` (the ``basis_carry`` convention). ``data_label`` is the
provenance tag sealed into the receipt: a leg of SYNTHETIC fixtures forces
every leg to be SYNTHETIC; otherwise the label is the sorted ``+``-join of
the venues' labels (e.g. ``kraken+okx``), so a mixed real+synthetic run
can never be sealed as market evidence.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_fund.data.sources.base import parse_time
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.utils.atomicio import publish_text_once
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

CROSSVENUE_SCHEMA = "crossvenue_basis.v1"
CROSSVENUE_KIND = "crossvenue_basis"

#: Minimum overlapping dates required per leg (spot-vs-mark) and per pair.
MIN_OVERLAP_DATES = 5
#: Calendar days per year in the annualized funding equivalent.
ANNUALIZATION_DAYS = 365.0
#: Seconds in one calendar day; feeds the per-day settlement count.
_SECONDS_PER_DAY = 86_400.0

_VENUE_RE_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")


@dataclass(frozen=True)
class VenueLeg:
    """One venue's view of the same asset: daily spot, daily mark, funding.

    ``spot``/``mark`` need an ``event_time`` (or ``date``) column and a
    ``close`` column — collected source frames qualify as-is. ``funding``
    needs ``event_time`` and ``value`` (the realized per-settlement rate);
    pass ``None`` for a venue without funding history — the pair's
    ``funding_diff`` then records ``unavailable`` rather than fabricating a
    series. ``data_label`` is the provenance tag sealed into the receipt
    (e.g. ``"kraken"``, ``"okx"``, ``"SYNTHETIC"`` for fixtures).
    """

    venue: str
    spot: pl.DataFrame
    mark: pl.DataFrame
    funding: pl.DataFrame | None
    data_label: str


def _to_utc_datetime(value: Any) -> datetime:
    """Normalize an event_time cell to an aware UTC datetime."""
    if isinstance(value, datetime):
        aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return aware.astimezone(UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if isinstance(value, str):
        text = value.strip()
        if len(text) == 10:
            return datetime.combine(date.fromisoformat(text), datetime.min.time(), tzinfo=UTC)
        return parse_time(text)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return parse_time(value)
    raise ValueError(f"unparseable event_time value: {value!r}")


def _price_series(frame: pl.DataFrame, *, name: str) -> dict[date, float]:
    """Extract a strictly-increasing {date: close} map, failing closed."""
    if frame.is_empty():
        raise ValueError(f"{name}: frame is empty")
    columns = set(frame.columns)
    time_col = "event_time" if "event_time" in columns else "date" if "date" in columns else None
    if time_col is None:
        raise ValueError(f"{name}: no event_time or date column")
    if "close" not in columns:
        raise ValueError(f"{name}: no close column")
    series: dict[date, float] = {}
    last_day: date | None = None
    for stamp, close_value in frame.select([time_col, "close"]).iter_rows():
        day = _to_utc_datetime(stamp).date()
        try:
            close = float(close_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: close is not numeric ({close_value!r})") from exc
        if not math.isfinite(close) or close <= 0.0:
            raise ValueError(f"{name}: close must be finite and positive ({close_value!r})")
        if last_day is not None and day <= last_day:
            raise ValueError(f"{name}: event_time is not strictly increasing")
        last_day = day
        series[day] = close
    return series


def _funding_series(frame: pl.DataFrame, *, name: str) -> list[tuple[datetime, float]]:
    """Extract a strictly-increasing [(settlement_ts, rate)] list."""
    if frame.is_empty():
        raise ValueError(f"{name}: funding frame is empty")
    columns = set(frame.columns)
    time_col = "event_time" if "event_time" in columns else "date" if "date" in columns else None
    if time_col is None:
        raise ValueError(f"{name}: no event_time or date column")
    value_col = "value" if "value" in columns else "rate" if "rate" in columns else None
    if value_col is None:
        raise ValueError(f"{name}: no value or rate column")
    rows: list[tuple[datetime, float]] = []
    last_ts: datetime | None = None
    for stamp, raw in frame.select([time_col, value_col]).iter_rows():
        ts = _to_utc_datetime(stamp)
        try:
            rate = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: funding rate is not numeric ({raw!r})") from exc
        if not math.isfinite(rate):
            raise ValueError(f"{name}: funding rate must be finite ({raw!r})")
        if last_ts is not None and ts <= last_ts:
            raise ValueError(f"{name}: funding event_time is not strictly increasing")
        last_ts = ts
        rows.append((ts, rate))
    return rows


def _series_digest(series: Mapping[date, float] | Sequence[tuple[datetime, float]]) -> str:
    """Content digest of one sealed series, in order."""
    if isinstance(series, Mapping):
        body = [[day.isoformat(), value] for day, value in series.items()]
    else:
        body = [[ts.isoformat(), value] for ts, value in series]
    return hash_bytes(canonical_json_bytes(body))


def _describe(diffs: Sequence[float]) -> dict[str, Any]:
    """Descriptive stats of a differential series — never a P&L claim."""
    array = np.asarray(diffs, dtype=float)
    sign_flips = int(
        sum(
            1
            for earlier, later in zip(diffs, diffs[1:], strict=False)
            if (earlier > 0) != (later > 0) and earlier != 0 and later != 0
        )
    )
    return {
        "n": int(array.size),
        "mean": float(array.mean()) if array.size else None,
        "std": float(array.std(ddof=1)) if array.size >= 2 else None,
        "min": float(array.min()) if array.size else None,
        "max": float(array.max()) if array.size else None,
        "max_abs": float(np.abs(array).max()) if array.size else None,
        "mean_abs": float(np.abs(array).mean()) if array.size else None,
        "frac_positive": float((array > 0).mean()) if array.size else None,
        "sign_flips": sign_flips,
    }


def _join_diff(
    a: Mapping[date, float],
    b: Mapping[date, float],
) -> tuple[list[date], list[float]]:
    """Inner-join two date->value maps and take (a - b) per shared date."""
    shared = sorted(day for day in a if day in b)
    return shared, [a[day] - b[day] for day in shared]


def _leg_basis(
    series_spot: Mapping[date, float], series_mark: Mapping[date, float]
) -> dict[date, float]:
    """Per-date basis (mark - spot)/spot on the inner join."""
    shared = sorted(day for day in series_spot if day in series_mark)
    return {day: (series_mark[day] - series_spot[day]) / series_spot[day] for day in shared}


def _daily_funding(
    rows: Sequence[tuple[datetime, float]],
) -> dict[date, tuple[float, int]]:
    """Sum realized rates per UTC calendar day with the settlement count."""
    daily: dict[date, tuple[float, int]] = {}
    for ts, rate in rows:
        day = ts.date()
        total, n = daily.get(day, (0.0, 0))
        daily[day] = (total + rate, n + 1)
    return daily


def _median_cadence_seconds(rows: Sequence[tuple[datetime, float]]) -> float | None:
    """Median gap between consecutive settlements (e.g. 3600 Kraken, 28800 OKX)."""
    if len(rows) < 2:
        return None
    gaps = [
        (later[0] - earlier[0]).total_seconds()
        for earlier, later in zip(rows, rows[1:], strict=False)
    ]
    return float(np.median(gaps))


def _funding_block(
    funding: pl.DataFrame | None,
    *,
    name: str,
    annualization_days: float,
) -> tuple[dict[str, Any], dict[date, tuple[float, int]] | None]:
    """Per-leg funding stats; returns (block, daily_map_or_None)."""
    if funding is None:
        return {"status": "unavailable", "reason": "no funding input declared"}, None
    try:
        rows = _funding_series(funding, name=name)
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}, None
    daily = _daily_funding(rows)
    days = sorted(daily)
    sums = [daily[day][0] for day in days]
    counts = [daily[day][1] for day in days]
    cadence = _median_cadence_seconds(rows)
    expected_per_day = _SECONDS_PER_DAY / cadence if cadence is not None and cadence > 0 else None
    # A day is partial when it shows fewer settlements than the cadence
    # implies (series head/tail days are naturally partial — flagged, not
    # dropped, so the receipt shows coverage exactly as measured).
    n_partial = (
        sum(1 for count in counts if count < expected_per_day - 1e-9)
        if expected_per_day is not None
        else 0
    )
    array = np.asarray(sums, dtype=float)
    block: dict[str, Any] = {
        "status": "ok",
        "n_settlements": len(rows),
        "first_ts": rows[0][0].isoformat(),
        "last_ts": rows[-1][0].isoformat(),
        "cadence_seconds": cadence,
        "expected_settlements_per_day": expected_per_day,
        "n_days": len(days),
        "n_partial_days": n_partial,
        "daily": [[day.isoformat(), daily[day][0], daily[day][1]] for day in days],
        "daily_mean": float(array.mean()),
        "daily_std": float(array.std(ddof=1)) if array.size >= 2 else None,
        "daily_min": float(array.min()),
        "daily_max": float(array.max()),
        "daily_mean_annualized": float(array.mean()) * annualization_days,
    }
    return block, daily


def _basis_leg_stats(basis: Mapping[date, float]) -> dict[str, Any]:
    days = sorted(basis)
    values = np.asarray([basis[day] for day in days], dtype=float)
    return {
        "n_dates": len(days),
        "first_date": days[0].isoformat(),
        "last_date": days[-1].isoformat(),
        "basis_first": float(values[0]),
        "basis_last": float(values[-1]),
        "basis_mean": float(values.mean()),
        "basis_std": float(values.std(ddof=1)) if values.size >= 2 else None,
        "basis_min": float(values.min()),
        "basis_max": float(values.max()),
        "basis_series": [[day.isoformat(), basis[day]] for day in days],
    }


def _derive_data_label(labels: Sequence[str]) -> str:
    """Provenance gate: SYNTHETIC mixes with nothing; else the sorted join."""
    distinct = {label for label in labels}
    if not distinct or any(not label.strip() for label in distinct):
        raise ValueError("every venue leg must declare a non-empty data_label")
    if "SYNTHETIC" in distinct:
        if len(distinct) > 1:
            raise ValueError(
                "crossvenue inputs carry mixed data_label values "
                f"{sorted(distinct)} — a SYNTHETIC leg can never join live tape"
            )
        return "SYNTHETIC"
    return "+".join(sorted(distinct))


def _leg_result(
    leg: VenueLeg,
    *,
    min_overlap: int,
    annualization_days: float,
) -> tuple[dict[str, Any], dict[date, float] | None, dict[date, tuple[float, int]] | None]:
    """Measure one venue leg; returns (row, basis_map, funding_daily_map)."""
    venue = leg.venue
    try:
        spot = _price_series(leg.spot, name=f"{venue}.spot")
        mark = _price_series(leg.mark, name=f"{venue}.mark")
    except ValueError as exc:
        return {"venue": venue, "status": "error", "error": str(exc)}, None, None
    basis = _leg_basis(spot, mark)
    if len(basis) < min_overlap:
        return (
            {
                "venue": venue,
                "status": "error",
                "error": f"insufficient_overlap: {len(basis)} shared dates < {min_overlap}",
            },
            None,
            None,
        )
    funding_block, funding_daily = _funding_block(
        leg.funding, name=f"{venue}.funding", annualization_days=annualization_days
    )
    row: dict[str, Any] = {
        "venue": venue,
        "status": "ok",
        **_basis_leg_stats(basis),
        "funding": funding_block,
    }
    if funding_block.get("status") == "error":
        row["status"] = "error"
        row["error"] = f"funding: {funding_block['error']}"
        return row, basis, None
    return row, basis, funding_daily


def _pair_row(
    a: dict[str, Any],
    basis_a: dict[date, float],
    daily_a: dict[date, tuple[float, int]] | None,
    b: dict[str, Any],
    basis_b: dict[date, float],
    daily_b: dict[date, tuple[float, int]] | None,
    *,
    min_overlap: int,
) -> dict[str, Any]:
    """Differential stats for one ordered venue pair (a - b)."""
    venue_a, venue_b = str(a["venue"]), str(b["venue"])
    pair = f"{venue_a}~{venue_b}"
    shared, diffs = _join_diff(basis_a, basis_b)
    if len(shared) < min_overlap:
        return {
            "pair": pair,
            "venue_a": venue_a,
            "venue_b": venue_b,
            "status": "error",
            "error": (f"insufficient_pair_overlap: {len(shared)} shared dates < {min_overlap}"),
        }
    basis_diff = _describe(diffs)
    basis_diff.update(
        {
            "n_dates": len(shared),
            "first_date": shared[0].isoformat(),
            "last_date": shared[-1].isoformat(),
        }
    )
    if daily_a is None or daily_b is None:
        funding_diff: dict[str, Any] = {
            "status": "unavailable",
            "reason": "a venue leg lacks funding history",
        }
    else:
        shared_f = sorted(day for day in daily_a if day in daily_b)
        if len(shared_f) < min_overlap:
            funding_diff = {
                "status": "error",
                "error": (
                    f"insufficient_funding_overlap: {len(shared_f)} shared dates < {min_overlap}"
                ),
            }
        else:
            f_diffs = [daily_a[day][0] - daily_b[day][0] for day in shared_f]
            funding_diff = _describe(f_diffs)
            funding_diff.update(
                {
                    "status": "ok",
                    "n_dates": len(shared_f),
                    "first_date": shared_f[0].isoformat(),
                    "last_date": shared_f[-1].isoformat(),
                    # Per-day settlement counts under the cadence expectation
                    # — coverage is reported, not assumed.
                    "n_partial_days_a": sum(
                        1 for day in shared_f if _is_partial_day(daily_a[day][1], a)
                    ),
                    "n_partial_days_b": sum(
                        1 for day in shared_f if _is_partial_day(daily_b[day][1], b)
                    ),
                }
            )
    row: dict[str, Any] = {
        "pair": pair,
        "venue_a": venue_a,
        "venue_b": venue_b,
        "status": "ok",
        "basis_diff": basis_diff,
        "funding_diff": funding_diff,
    }
    if funding_diff.get("status") == "error":
        row["status"] = "error"
        row["error"] = funding_diff["error"]
    return row


def _is_partial_day(n_settlements: int, leg_row: Mapping[str, Any]) -> bool:
    """Whether a shared date had fewer settlements than the leg's cadence."""
    funding = leg_row.get("funding")
    if not isinstance(funding, Mapping):
        return False
    expected = funding.get("expected_settlements_per_day")
    if not isinstance(expected, (int, float)) or isinstance(expected, bool):
        return False
    return n_settlements < float(expected) - 1e-9


def run_crossvenue_basis(
    *,
    legs: Sequence[VenueLeg],
    asset: str,
    min_overlap: int = MIN_OVERLAP_DATES,
    annualization_days: float = ANNUALIZATION_DAYS,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Run the cross-venue funding/basis bench; returns (frame, receipt).

    Every pair of venue legs produces one result row of differential
    descriptive statistics (``a - b``). Leg problems record a
    ``status="error"`` row so the sealed receipt documents the failure
    rather than dropping it. Needs >= 2 distinct venue legs — with one
    venue there is no differential to measure, so it fails closed.
    """
    if len(legs) < 2:
        raise ValueError("crossvenue basis needs at least two venue legs")
    if min_overlap < 2:
        raise ValueError("min_overlap must be >= 2")
    if annualization_days <= 0.0:
        raise ValueError("annualization_days must be positive")
    if not asset.strip():
        raise ValueError("asset must be non-empty")
    venues = [leg.venue for leg in legs]
    for venue in venues:
        if venue != venue.strip() or not venue or any(ch not in _VENUE_RE_CHARS for ch in venue):
            raise ValueError(f"venue names must be lowercase identifiers, got {venue!r}")
    if len(set(venues)) != len(venues):
        raise ValueError(
            f"duplicate venue legs: {sorted(v for v in venues if venues.count(v) > 1)}"
        )

    sorted_legs = sorted(legs, key=lambda leg: leg.venue)
    data_label = _derive_data_label([leg.data_label for leg in sorted_legs])
    params: dict[str, Any] = {
        "asset": asset,
        "min_overlap": min_overlap,
        "annualization_days": annualization_days,
        "conventions": (
            "basis_t = (mark_t - spot_t)/spot_t per venue on shared dates; "
            "basis_diff = a - b per venue pair; funding aggregated to "
            "per-calendar-day sums of realized rates before diffing, so "
            "venue cadences (hourly vs 8h) are comparable; "
            "daily_mean_annualized = daily_mean * annualization_days is a "
            "descriptive scaling, not a yield claim"
        ),
    }

    leg_rows: list[dict[str, Any]] = []
    basis_by_venue: dict[str, dict[date, float]] = {}
    daily_by_venue: dict[str, dict[date, tuple[float, int]] | None] = {}
    for leg in sorted_legs:
        row, basis, daily = _leg_result(
            leg, min_overlap=min_overlap, annualization_days=annualization_days
        )
        leg_rows.append(row)
        if row["status"] == "ok":
            assert basis is not None
            basis_by_venue[leg.venue] = basis
            daily_by_venue[leg.venue] = daily

    ok_leg_rows = [row for row in leg_rows if row["status"] == "ok"]
    pair_rows: list[dict[str, Any]] = []
    for a, b in combinations(ok_leg_rows, 2):
        pair_rows.append(
            _pair_row(
                a,
                basis_by_venue[str(a["venue"])],
                daily_by_venue[str(a["venue"])],
                b,
                basis_by_venue[str(b["venue"])],
                daily_by_venue[str(b["venue"])],
                min_overlap=min_overlap,
            )
        )
    n_error_rows = sum(1 for row in leg_rows if row["status"] != "ok") + sum(
        1 for row in pair_rows if row["status"] != "ok"
    )

    inputs_legs: dict[str, Any] = {}
    for leg in sorted_legs:
        venue = leg.venue
        entry: dict[str, Any] = {"data_label": leg.data_label}
        for field in ("spot", "mark", "funding"):
            frame = getattr(leg, field)
            if frame is None:
                entry[field] = None
                continue
            try:
                if field == "funding":
                    funding_rows = _funding_series(frame, name=f"{venue}.funding")
                    entry[field] = {
                        "sha256": _series_digest(funding_rows),
                        "n_rows": len(funding_rows),
                    }
                else:
                    price_series = _price_series(frame, name=f"{venue}.{field}")
                    entry[field] = {
                        "sha256": _series_digest(price_series),
                        "n_rows": len(price_series),
                        "first_date": min(price_series).isoformat(),
                        "last_date": max(price_series).isoformat(),
                    }
            except ValueError:
                entry[field] = {"sha256": "INVALID", "n_rows": 0}
        inputs_legs[venue] = entry
    inputs_block = {"asset": asset, "legs": inputs_legs, "params": params}
    inputs_sha256 = hash_bytes(canonical_json_bytes(inputs_block))
    dataset_sha256 = hash_bytes(
        canonical_json_bytes(
            {
                venue: {
                    field: entry[field].get("sha256") if entry[field] is not None else None
                    for field in ("spot", "mark", "funding")
                }
                for venue, entry in inputs_legs.items()
            }
        )
    )

    n_pairs_ok = sum(1 for row in pair_rows if row["status"] == "ok")
    if pair_rows and n_pairs_ok:
        sample = next(row for row in pair_rows if row["status"] == "ok")
        bdiff = sample["basis_diff"]["mean"]
        fdiff = (
            sample["funding_diff"].get("mean") if isinstance(sample["funding_diff"], dict) else None
        )
        verdict = (
            f"{len(ok_leg_rows)}/{len(leg_rows)} legs measured; {n_pairs_ok}/{len(pair_rows)} "
            f"pairs ok; {sample['pair']} basis-diff mean {bdiff:+.6%}"
            + (f", funding-diff mean {fdiff:+.6%}/day" if fdiff is not None else "")
        )
    else:
        verdict = (
            f"{len(ok_leg_rows)}/{len(leg_rows)} legs measured; "
            "no venue pair produced a differential"
        )
    receipt: dict[str, Any] = {
        "schema": CROSSVENUE_SCHEMA,
        "kind": CROSSVENUE_KIND,
        "claim": "research_only",
        "research_only": True,
        "live_pnl_claim": False,
        "simulated_only": False,
        "data_label": data_label,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(),
        "asset": asset,
        "venues": [leg.venue for leg in sorted_legs],
        "params": params,
        "inputs": inputs_block,
        "inputs_sha256": inputs_sha256,
        "dataset_sha256": dataset_sha256,
        "n_rows": len(pair_rows),
        "n_error_rows": n_error_rows,
        "n_leg_rows": len(leg_rows),
        "legs": leg_rows,
        "results": pair_rows,
        "verdict": verdict,
    }
    frame = pl.DataFrame(
        {
            key: [_pair_frame_value(row, key) for row in pair_rows]
            for key in (
                "pair",
                "status",
                "n_dates",
                "basis_diff_mean",
                "basis_diff_std",
                "basis_diff_max_abs",
                "funding_diff_mean",
                "funding_diff_max_abs",
            )
        }
    )
    return frame, receipt


def _pair_frame_value(row: Mapping[str, Any], key: str) -> Any:
    """Project one pair row onto the flat output frame schema."""
    if key == "n_dates":
        diff = row.get("basis_diff")
        return diff.get("n_dates") if isinstance(diff, Mapping) else None
    for block in ("basis_diff", "funding_diff"):
        if key.startswith(block + "_"):
            value = row.get(block)
            stat = value.get(key[len(block) + 1 :]) if isinstance(value, Mapping) else None
            return stat
    return row.get(key)


def crossvenue_basis_contract_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Fail-closed contract for a ``crossvenue_basis.v1`` payload (writer + verifier)."""
    errors: list[str] = []
    research_blob = {key: value for key, value in receipt.items() if key != "live_pnl_claim"}
    if receipt.get("schema") != CROSSVENUE_SCHEMA:
        errors.append("schema_not_crossvenue_basis_v1")
    if receipt.get("kind") != CROSSVENUE_KIND:
        errors.append("kind_not_crossvenue_basis")
    if receipt.get("research_only") is not True:
        errors.append("research_only_not_true")
    if receipt.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    if receipt.get("simulated_only") is not False:
        errors.append("simulated_only_not_false")
    if not str(receipt.get("data_label") or "").strip():
        errors.append("data_label_missing")
    if not str(receipt.get("verdict") or "").strip():
        errors.append("verdict_missing")
    if not family_blob_forbidden_metrics_absent(research_blob):
        errors.append("forbidden_metric_keys")

    inputs = receipt.get("inputs")
    params = receipt.get("params")
    if not isinstance(inputs, Mapping) or not isinstance(params, Mapping):
        errors.append("inputs_or_params_missing")
        return errors
    if inputs.get("params") != dict(params):
        errors.append("inputs_params_mismatch")
    legs_meta = inputs.get("legs")
    if not isinstance(legs_meta, Mapping) or not legs_meta:
        errors.append("inputs_legs_missing")
        return errors
    labels = {
        str(meta.get("data_label")) for meta in legs_meta.values() if isinstance(meta, Mapping)
    }
    try:
        expected_label = _derive_data_label(sorted(labels))
    except ValueError:
        expected_label = None
        errors.append("data_label_not_deriveable")
    if expected_label is not None and expected_label != receipt.get("data_label"):
        errors.append("data_label_not_derived")
    try:
        if hash_bytes(canonical_json_bytes(dict(inputs))) != receipt.get("inputs_sha256"):
            errors.append("inputs_sha256_mismatch")
    except (TypeError, ValueError):
        errors.append("inputs_sha256_uncomputable")
    try:
        expected_dataset = hash_bytes(
            canonical_json_bytes(
                {
                    venue: {
                        field: (meta.get(field) or {}).get("sha256")
                        if meta.get(field) is not None
                        else None
                        for field in ("spot", "mark", "funding")
                    }
                    for venue, meta in legs_meta.items()
                    if isinstance(meta, Mapping)
                }
            )
        )
    except (TypeError, ValueError, AttributeError):
        expected_dataset = "UNCOMPUTABLE"
    if expected_dataset != receipt.get("dataset_sha256"):
        errors.append("dataset_sha256_mismatch")

    legs = receipt.get("legs")
    results = receipt.get("results")
    if not isinstance(legs, list) or not legs:
        errors.append("legs_missing_or_empty")
        return errors
    if not isinstance(results, list):
        errors.append("results_missing")
        return errors
    if receipt.get("n_rows") != len(results):
        errors.append("n_rows_mismatch")
    if receipt.get("n_leg_rows") != len(legs):
        errors.append("n_leg_rows_mismatch")
    n_error = sum(
        1 for row in legs if not isinstance(row, Mapping) or row.get("status") != "ok"
    ) + sum(1 for row in results if not isinstance(row, Mapping) or row.get("status") != "ok")
    if receipt.get("n_error_rows") != n_error:
        errors.append("n_error_rows_mismatch")
    min_overlap = params.get("min_overlap")
    if not isinstance(min_overlap, int):
        errors.append("params_malformed")
        return errors
    venues = receipt.get("venues")
    if sorted(str(row.get("venue")) for row in legs if isinstance(row, Mapping)) != sorted(
        [str(v) for v in venues] if isinstance(venues, list) else []
    ):
        errors.append("venues_mismatch")
    if not str(receipt.get("asset") or "").strip():
        errors.append("asset_missing")

    leg_by_venue: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(legs):
        if not isinstance(row, Mapping):
            errors.append(f"legs[{index}]_not_object")
            continue
        venue = str(row.get("venue"))
        leg_by_venue[venue] = row
        errors.extend(_leg_row_errors(row, index=index, min_overlap=min_overlap))

    for index, row in enumerate(results):
        if not isinstance(row, Mapping):
            errors.append(f"results[{index}]_not_object")
            continue
        errors.extend(
            _pair_row_errors(row, index=index, legs=leg_by_venue, min_overlap=min_overlap)
        )
    return errors


def _leg_row_errors(row: Mapping[str, Any], *, index: int, min_overlap: int) -> list[str]:
    """Re-derive every sealed leg statistic from the stored series."""
    errors: list[str] = []
    if row.get("status") != "ok":
        if not str(row.get("error", "")).strip():
            errors.append(f"legs[{index}].error_empty")
        return errors
    series = row.get("basis_series")
    if not isinstance(series, list) or not series:
        return [f"legs[{index}].basis_series_missing"]
    days: list[date] = []
    values: list[float] = []
    for entry in series:
        if not (isinstance(entry, list) and len(entry) == 2):
            errors.append(f"legs[{index}].basis_series_row")
            return errors
        try:
            days.append(date.fromisoformat(str(entry[0])))
            values.append(float(entry[1]))
        except (TypeError, ValueError):
            errors.append(f"legs[{index}].basis_series_unparseable")
            return errors
    if days != sorted(days) or len(set(days)) != len(days):
        errors.append(f"legs[{index}].basis_series_order")
        return errors
    n = len(values)
    if n < min_overlap or row.get("n_dates") != n:
        errors.append(f"legs[{index}].n_dates")
    if row.get("first_date") != days[0].isoformat() or row.get("last_date") != days[-1].isoformat():
        errors.append(f"legs[{index}].date_bounds")
    array = np.asarray(values, dtype=float)
    expected = {
        "basis_first": float(values[0]),
        "basis_last": float(values[-1]),
        "basis_mean": float(array.mean()),
        "basis_min": float(array.min()),
        "basis_max": float(array.max()),
        "basis_std": float(array.std(ddof=1)) if n >= 2 else None,
    }
    for field, wanted in expected.items():
        claimed = row.get(field)
        if wanted is None:
            if claimed is not None:
                errors.append(f"legs[{index}].{field}")
        elif not isinstance(claimed, (int, float)) or not math.isclose(
            float(claimed), wanted, rel_tol=1e-9
        ):
            errors.append(f"legs[{index}].{field}")
    funding = row.get("funding")
    if not isinstance(funding, Mapping):
        errors.append(f"legs[{index}].funding_missing")
        return errors
    status = funding.get("status")
    if status == "unavailable":
        if not str(funding.get("reason", "")).strip():
            errors.append(f"legs[{index}].funding_reason_empty")
        return errors
    if status != "ok":
        errors.append(f"legs[{index}].funding_status")
        return errors
    daily = funding.get("daily")
    if not isinstance(daily, list) or not daily:
        errors.append(f"legs[{index}].funding_daily_missing")
        return errors
    d_days: list[date] = []
    sums: list[float] = []
    counts: list[int] = []
    for entry in daily:
        if not (isinstance(entry, list) and len(entry) == 3):
            errors.append(f"legs[{index}].funding_daily_row")
            return errors
        try:
            d_days.append(date.fromisoformat(str(entry[0])))
            sums.append(float(entry[1]))
            counts.append(int(entry[2]))
        except (TypeError, ValueError):
            errors.append(f"legs[{index}].funding_daily_unparseable")
            return errors
    if d_days != sorted(d_days) or len(set(d_days)) != len(d_days):
        errors.append(f"legs[{index}].funding_daily_order")
        return errors
    if funding.get("n_days") != len(d_days) or funding.get("n_settlements") != sum(counts):
        errors.append(f"legs[{index}].funding_counts")
    array = np.asarray(sums, dtype=float)
    expected_f: dict[str, Any] = {
        "daily_mean": float(array.mean()),
        "daily_min": float(array.min()),
        "daily_max": float(array.max()),
        "daily_std": float(array.std(ddof=1)) if array.size >= 2 else None,
    }
    for field, wanted in expected_f.items():
        claimed = funding.get(field)
        if wanted is None:
            if claimed is not None:
                errors.append(f"legs[{index}].funding.{field}")
        elif not isinstance(claimed, (int, float)) or not math.isclose(
            float(claimed), wanted, rel_tol=1e-9
        ):
            errors.append(f"legs[{index}].funding.{field}")
    expected_per_day = funding.get("expected_settlements_per_day")
    n_partial_claimed = funding.get("n_partial_days")
    if isinstance(expected_per_day, (int, float)) and not isinstance(expected_per_day, bool):
        expected_partial = sum(1 for c in counts if c < float(expected_per_day) - 1e-9)
        if n_partial_claimed != expected_partial:
            errors.append(f"legs[{index}].funding.n_partial_days")
    return errors


def _stats_block_errors(
    claimed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    where: str,
) -> list[str]:
    """Compare a sealed diff-stats block against a recomputed one."""
    errors: list[str] = []
    for field, wanted in expected.items():
        got = claimed.get(field)
        if wanted is None:
            if got is not None:
                errors.append(f"{where}.{field}")
        elif isinstance(wanted, str):
            if got != wanted:
                errors.append(f"{where}.{field}")
        elif not isinstance(got, (int, float)) or not math.isclose(
            float(got), float(wanted), rel_tol=1e-9
        ):
            errors.append(f"{where}.{field}")
    return errors


def _pair_row_errors(
    row: Mapping[str, Any],
    *,
    index: int,
    legs: Mapping[str, Mapping[str, Any]],
    min_overlap: int,
) -> list[str]:
    """Re-derive a pair row's diff stats from the two legs' stored series."""
    errors: list[str] = []
    venue_a, venue_b = str(row.get("venue_a")), str(row.get("venue_b"))
    if venue_a >= venue_b or row.get("pair") != f"{venue_a}~{venue_b}":
        errors.append(f"results[{index}].pair")
    leg_a, leg_b = legs.get(venue_a), legs.get(venue_b)
    if row.get("status") != "ok":
        if not str(row.get("error", "")).strip():
            errors.append(f"results[{index}].error_empty")
        return errors
    if leg_a is None or leg_b is None:
        errors.append(f"results[{index}].unknown_leg")
        return errors

    def _series_map(leg: Mapping[str, Any]) -> dict[date, float] | None:
        series = leg.get("basis_series")
        if not isinstance(series, list):
            return None
        out: dict[date, float] = {}
        try:
            for d, v in series:
                out[date.fromisoformat(str(d))] = float(v)
        except (TypeError, ValueError):
            return None
        return out

    basis_a, basis_b = _series_map(leg_a), _series_map(leg_b)
    if basis_a is None or basis_b is None:
        errors.append(f"results[{index}].leg_series_missing")
        return errors
    shared, diffs = _join_diff(basis_a, basis_b)
    if len(shared) < min_overlap:
        errors.append(f"results[{index}].basis_overlap")
        return errors
    expected_basis = _describe(diffs)
    expected_basis.update(
        {
            "n_dates": len(shared),
            "first_date": shared[0].isoformat(),
            "last_date": shared[-1].isoformat(),
        }
    )
    claimed_basis = row.get("basis_diff")
    if not isinstance(claimed_basis, Mapping):
        errors.append(f"results[{index}].basis_diff_missing")
    else:
        errors.extend(
            _stats_block_errors(claimed_basis, expected_basis, where=f"results[{index}].basis_diff")
        )

    funding_diff = row.get("funding_diff")
    daily_a = _daily_map(leg_a)
    daily_b = _daily_map(leg_b)
    if daily_a is None or daily_b is None:
        if not isinstance(funding_diff, Mapping) or funding_diff.get("status") != "unavailable":
            errors.append(f"results[{index}].funding_diff_status")
        return errors
    if not isinstance(funding_diff, Mapping):
        errors.append(f"results[{index}].funding_diff_missing")
        return errors
    if funding_diff.get("status") == "error":
        if not str(funding_diff.get("error", "")).strip():
            errors.append(f"results[{index}].funding_diff_error_empty")
        return errors
    if funding_diff.get("status") != "ok":
        errors.append(f"results[{index}].funding_diff_status")
        return errors
    shared_f = sorted(day for day in daily_a if day in daily_b)
    if len(shared_f) < min_overlap:
        errors.append(f"results[{index}].funding_overlap")
        return errors
    expected_funding = _describe([daily_a[d][0] - daily_b[d][0] for d in shared_f])
    expected_funding.update(
        {
            "n_dates": len(shared_f),
            "first_date": shared_f[0].isoformat(),
            "last_date": shared_f[-1].isoformat(),
        }
    )
    errors.extend(
        _stats_block_errors(funding_diff, expected_funding, where=f"results[{index}].funding_diff")
    )
    for suffix, leg, daily in (("a", leg_a, daily_a), ("b", leg_b, daily_b)):
        expected_partial = sum(1 for day in shared_f if _is_partial_day(daily[day][1], leg))
        if funding_diff.get(f"n_partial_days_{suffix}") != expected_partial:
            errors.append(f"results[{index}].funding_diff.n_partial_days_{suffix}")
    return errors


def _daily_map(leg: Mapping[str, Any]) -> dict[date, tuple[float, int]] | None:
    """Rebuild a leg's {date: (rate_sum, n)} map from the sealed daily list."""
    funding = leg.get("funding")
    if not isinstance(funding, Mapping) or funding.get("status") != "ok":
        return None
    daily = funding.get("daily")
    if not isinstance(daily, list):
        return None
    out: dict[date, tuple[float, int]] = {}
    try:
        for d, s, n in daily:
            out[date.fromisoformat(str(d))] = (float(s), int(n))
    except (TypeError, ValueError):
        return None
    return out


def crossvenue_basis_dataset_identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Input content digests bound by a v2 ``dataset_hash``."""
    inputs = receipt.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("crossvenue receipt has no inputs block")
    legs = inputs.get("legs")
    if not isinstance(legs, Mapping):
        raise ValueError("crossvenue receipt inputs lack legs block")
    return {
        "asset": inputs.get("asset"),
        "legs": {
            str(venue): {
                field: (meta.get(field) or {}).get("sha256")
                if meta.get(field) is not None
                else None
                for field in ("spot", "mark", "funding")
            }
            for venue, meta in legs.items()
            if isinstance(meta, Mapping)
        },
    }


def crossvenue_basis_params(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """The run parameters bound by a v2 ``params_hash``."""
    params = receipt.get("params")
    return dict(params) if isinstance(params, Mapping) else {}


def crossvenue_basis_verdict(receipt: Mapping[str, Any]) -> str:
    """``pass`` iff every leg and pair produced a measured row.

    This is an artifact-integrity verdict — whether the bench produced a
    complete measurement, not a market claim about the differentials.
    """
    return "pass" if receipt.get("n_error_rows") == 0 else "fail"


def crossvenue_basis_receipt_v2(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap a ``crossvenue_basis.v1`` payload in the unified ``receipt.v2`` envelope.

    The v1 payload is embedded verbatim under ``payload``; the envelope binds
    the input content digests, run params, this module's source hash, and the
    loaded numeric stack. Validates the v1 contract first — a malformed v1
    receipt is never wrapped.
    """
    from quant_fund.research.receipt_v2 import build_receipt_v2

    if crossvenue_basis_contract_errors(receipt):
        raise ValueError("crossvenue receipt violates its research contract")
    return build_receipt_v2(
        kind=str(receipt["kind"]),
        data_label=str(receipt["data_label"]),
        dataset=crossvenue_basis_dataset_identity(receipt),
        params=crossvenue_basis_params(receipt),
        code_files=(Path(__file__),),
        verdict=crossvenue_basis_verdict(receipt),
        payload=dict(receipt),
        generated_at=str(receipt["generated_at"]),
        revision=str(receipt["git_revision"]),
    )


def crossvenue_basis_v2_consistency_errors(envelope: Mapping[str, Any]) -> list[str]:
    """Re-derive a crossvenue receipt.v2 envelope's bound digests from its payload."""
    errors: list[str] = []
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping):
        return ["payload_not_object"]
    contract_errors = crossvenue_basis_contract_errors(payload)
    errors.extend(f"payload_{name}" for name in contract_errors)
    if contract_errors:
        return errors
    try:
        dataset = crossvenue_basis_dataset_identity(payload)
    except ValueError as exc:
        return [*errors, f"payload_{exc}"]
    if hash_bytes(canonical_json_bytes(dataset)) != envelope.get("dataset_hash"):
        errors.append("dataset_hash_mismatch")
    if hash_bytes(canonical_json_bytes(crossvenue_basis_params(payload))) != envelope.get(
        "params_hash"
    ):
        errors.append("params_hash_mismatch")
    if crossvenue_basis_verdict(payload) != envelope.get("verdict"):
        errors.append("verdict_mismatch")
    return errors


def write_crossvenue_basis_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
    *,
    receipt_version: int = 1,
) -> Path:
    """Seal a crossvenue receipt and write ``receipts/crossvenue_basis_<hash>.json``.

    The filename hash is the sha256 of the canonical receipt payload; the
    same digest is embedded as ``receipt_sha256`` (the sealed-evidence
    convention). The write is atomic and fail-closed on tampering.
    ``receipt_version=2`` wraps the v1 payload in the unified ``receipt.v2``
    envelope before sealing.
    """
    from quant_fund.research.receipt_v2 import seal_receipt

    if receipt_version == 1:
        if crossvenue_basis_contract_errors(receipt):
            raise ValueError("crossvenue receipt violates its research contract")
        body: Mapping[str, Any] = receipt
    elif receipt_version == 2:
        body = crossvenue_basis_receipt_v2(receipt)
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    payload = seal_receipt(body)
    path = Path(receipts_dir) / f"crossvenue_basis_{payload['receipt_sha256'][:16]}.json"
    publish_text_once(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


__all__ = [
    "ANNUALIZATION_DAYS",
    "CROSSVENUE_KIND",
    "CROSSVENUE_SCHEMA",
    "MIN_OVERLAP_DATES",
    "VenueLeg",
    "crossvenue_basis_contract_errors",
    "crossvenue_basis_dataset_identity",
    "crossvenue_basis_params",
    "crossvenue_basis_receipt_v2",
    "crossvenue_basis_v2_consistency_errors",
    "crossvenue_basis_verdict",
    "run_crossvenue_basis",
    "write_crossvenue_basis_receipt",
]
