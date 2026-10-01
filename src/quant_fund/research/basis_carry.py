"""Settlement-anchored cash-and-carry bench (ULTRAPLAN P5.3).

Measures the futures-over-spot basis curve on *dated* delivery contracts —
the one structural edge with positive published out-of-sample evidence. A
spot daily frame and per-contract dated-future daily mark frames (each with
its delivery instant) are inner-joined on calendar date; the bench computes
the annualized basis as a function of days-to-delivery, the basis at fixed
dte buckets, and the residual basis at the last observed date — for a
delivered contract that residual is the settlement-anchor convergence
number; for a live contract it is ``convergence_residual`` measured
``residual_dte_days`` before delivery (the fields never pretend otherwise).

Output is descriptive statistics only — never P&L.
``write_basis_carry_receipt`` seals hash-stamped JSON evidence under
``receipts/`` (the ``vol_bench`` convention). ``data_label`` is derived
from the declared input labels per the provenance-gate convention: every
declared input must carry the same label or the run fails closed —
SYNTHETIC corpora label every input ``SYNTHETIC``; a live venue run labels
every input by that venue (``kraken``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_fund.data.sources.base import parse_time
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.utils.atomicio import publish_text_once
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

BASIS_CARRY_SCHEMA = "basis_carry.v1"
BASIS_CARRY_KIND = "basis_carry"

#: Days-to-delivery anchors at which the basis curve is sampled.
DTE_BUCKETS: tuple[float, ...] = (90.0, 60.0, 30.0, 14.0, 7.0, 1.0)
#: Calendar days per year in the annualization factor.
ANNUALIZATION_DAYS = 365.0
#: Minimum overlapping spot/future dates required per contract.
MIN_OVERLAP_DATES = 5
#: Rows with fewer days to delivery than this are dropped from the
#: annualized-basis aggregates — division by a sub-day dte explodes.
DTE_FLOOR_DAYS = 1.0
#: Default |residual| share-of-spot tolerance for the settlement anchor.
DEFAULT_TOLERANCE = 0.005
#: Guard against division by zero inside ``max(dte, eps)``.
_EPS_DTE = 1e-9

#: Kraken dated-contract families and their delivery hour-of-day (UTC).
#: ``FI_*_YYMMDD`` contracts settle 16:00 UTC on the dated day;
#: ``FF_*_YYMMDD`` flexible futures settle 08:00 UTC. Used to recover the
#: delivery instant for expired contracts the instruments endpoint no
#: longer lists.
_KRAKEN_DELIVERY_UTC_HOUR = {"FI": 16, "FF": 8}


@dataclass(frozen=True)
class CarryContractInput:
    """One dated-future input: symbol, daily mark frame, delivery instant.

    ``delivery`` may be a tz-aware ``datetime`` or an ISO-8601 string;
    ``data_label`` is the provenance tag sealed into the receipt (e.g.
    ``"kraken"`` for a live venue pull, ``"SYNTHETIC"`` for fixtures).
    """

    symbol: str
    frame: pl.DataFrame
    delivery: datetime | str
    data_label: str


def _to_utc_date(value: Any) -> date:
    """Normalize an event_time/date cell to a UTC calendar date."""
    if isinstance(value, datetime):
        aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return aware.astimezone(UTC).date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if len(text) == 10:
            return date.fromisoformat(text)
        return parse_time(text).date()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return parse_time(value).date()
    raise ValueError(f"unparseable event_time/date value: {value!r}")


def _series_from_frame(frame: pl.DataFrame, *, name: str) -> tuple[list[date], list[float]]:
    """Extract a strictly-increasing (date, close) series, failing closed."""
    if frame.is_empty():
        raise ValueError(f"{name}: frame is empty")
    columns = set(frame.columns)
    time_col = "event_time" if "event_time" in columns else "date" if "date" in columns else None
    if time_col is None:
        raise ValueError(f"{name}: no event_time or date column")
    if "close" not in columns:
        raise ValueError(f"{name}: no close column")
    dates: list[date] = []
    closes: list[float] = []
    for stamp, close_value in frame.select([time_col, "close"]).iter_rows():
        day = _to_utc_date(stamp)
        try:
            close = float(close_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: close is not numeric ({close_value!r})") from exc
        if not math.isfinite(close) or close <= 0.0:
            raise ValueError(f"{name}: close must be finite and positive ({close_value!r})")
        dates.append(day)
        closes.append(close)
    for earlier, later in zip(dates, dates[1:], strict=False):
        if later <= earlier:
            raise ValueError(f"{name}: event_time is not strictly increasing")
    return dates, closes


def _parse_delivery(value: datetime | str, *, symbol: str) -> datetime:
    """Normalize the delivery instant to an aware UTC datetime."""
    try:
        delivery = parse_time(value)
    except (TypeError, ValueError, OverflowError, OSError) as exc:
        raise ValueError(f"{symbol}: delivery instant is unparseable ({value!r})") from exc
    return delivery


def kraken_delivery_from_symbol(symbol: str) -> datetime:
    """Derive the delivery instant of a Kraken dated contract from its symbol.

    ``FI_<PAIR>_YYMMDD`` delivers 16:00 UTC and ``FF_<PAIR>_YYMMDD`` delivers
    08:00 UTC on the dated day (verified against the live instruments feed).
    Expired contracts are absent from the instruments snapshot, so this
    suffix parse is the only delivery source for already-settled history.
    """
    left, sep, tail = symbol.upper().rpartition("_")
    family = left.partition("_")[0]
    if not (
        sep
        and family in _KRAKEN_DELIVERY_UTC_HOUR
        and len(tail) == 6
        and tail.isdigit()
        and "_" in left
    ):
        raise ValueError(
            f"Kraken dated contract symbols look like 'FI_XBTUSD_261225', got {symbol!r}"
        )
    try:
        day = date(2000 + int(tail[:2]), int(tail[2:4]), int(tail[4:6]))
    except ValueError as exc:
        raise ValueError(f"Kraken contract {symbol!r} has an impossible date tail") from exc
    return datetime.combine(day, time(_KRAKEN_DELIVERY_UTC_HOUR[family], 0), tzinfo=UTC)


def _series_digest(dates: Sequence[date], closes: Sequence[float]) -> str:
    """Content digest of one price series: the (date, close) pairs, in order."""
    return hash_bytes(
        canonical_json_bytes([[d.isoformat(), c] for d, c in zip(dates, closes, strict=True)])
    )


def _bucket_key(bucket: float) -> str:
    return str(int(bucket)) if float(bucket).is_integer() else str(bucket)


def _contract_row(
    contract: CarryContractInput,
    *,
    spot_dates: Sequence[date],
    spot_by_date: Mapping[date, float],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure one contract against the spot series; errors become rows."""
    symbol = contract.symbol
    try:
        delivery = _parse_delivery(contract.delivery, symbol=symbol)
        fut_dates, fut_closes = _series_from_frame(contract.frame, name=symbol)
    except ValueError as exc:
        return {"contract": symbol, "status": "error", "error": str(exc)}
    fut_by_date = dict(zip(fut_dates, fut_closes, strict=True))
    merged = sorted(day for day in spot_dates if day in fut_by_date)
    min_overlap = int(params["min_overlap"])
    if len(merged) < min_overlap:
        return {
            "contract": symbol,
            "status": "error",
            "error": (f"insufficient_overlap: {len(merged)} shared dates < {min_overlap}"),
            "delivery": delivery.isoformat(),
        }
    closes_spot = [spot_by_date[day] for day in merged]
    closes_fut = [fut_by_date[day] for day in merged]
    basis = [(fut - spot) / spot for spot, fut in zip(closes_spot, closes_fut, strict=True)]
    midnight = time(0, 0)
    dte_days = [
        (delivery - datetime.combine(day, midnight, tzinfo=UTC)).total_seconds() / 86_400.0
        for day in merged
    ]
    dte_floor = float(params["dte_floor_days"])
    annualization = float(params["annualization_days"])
    ann_rows = [
        b * annualization / max(dte, _EPS_DTE)
        for b, dte in zip(basis, dte_days, strict=True)
        if dte >= dte_floor
    ]
    buckets: dict[str, dict[str, Any] | None] = {}
    for bucket in params["dte_buckets"]:
        # Nearest observation with dte <= bucket: the carry curve reading at
        # that anchor, or null when no observation is inside it.
        eligible = [
            (day, b, dte)
            for day, b, dte in zip(merged, basis, dte_days, strict=True)
            if dte <= bucket
        ]
        if not eligible:
            buckets[_bucket_key(bucket)] = None
            continue
        day, bucket_basis, bucket_dte = max(eligible, key=lambda item: item[2])
        buckets[_bucket_key(bucket)] = {
            "basis": bucket_basis,
            "ann_basis": (
                bucket_basis * annualization / max(bucket_dte, _EPS_DTE)
                if bucket_dte >= dte_floor
                else None
            ),
            "dte_days": bucket_dte,
            "date": day.isoformat(),
        }
    residual = basis[-1]
    residual_dte = dte_days[-1]
    basis_array = np.asarray(basis, dtype=float)
    ann_array = np.asarray(ann_rows, dtype=float)
    return {
        "contract": symbol,
        "status": "ok",
        "delivery": delivery.isoformat(),
        "n_dates": len(merged),
        "first_date": merged[0].isoformat(),
        "last_date": merged[-1].isoformat(),
        "delivered": merged[-1] >= delivery.date(),
        "residual_dte_days": residual_dte,
        "basis_first": basis[0],
        "basis_last": residual,
        "convergence_residual": residual,
        "basis_min": float(basis_array.min()),
        "basis_max": float(basis_array.max()),
        "basis_diff_std": float(np.std(np.diff(basis_array), ddof=1))
        if basis_array.size >= 3
        else None,
        "ann_basis_mean": float(ann_array.mean()) if ann_array.size else None,
        "ann_basis_std": float(ann_array.std(ddof=1)) if ann_array.size >= 2 else None,
        "ann_basis_n": int(ann_array.size),
        "dte_buckets": buckets,
    }


def _cross_contract(rows: Sequence[Mapping[str, Any]], *, tolerance: float) -> dict[str, Any]:
    """Distribution of per-contract convergence residuals — honest, no gate."""
    ok = [row for row in rows if row.get("status") == "ok"]
    residuals = [float(row["convergence_residual"]) for row in ok]
    delivered = sum(1 for row in ok if row.get("delivered") is True)
    array = np.asarray(residuals, dtype=float)
    within = sum(1 for residual in residuals if abs(residual) <= tolerance)
    return {
        "n_contracts": len(ok),
        "n_delivered": delivered,
        "residuals": residuals,
        "residual_mean": float(array.mean()) if array.size else None,
        "residual_std": float(array.std(ddof=1)) if array.size >= 2 else None,
        "residual_max_abs": float(np.abs(array).max()) if array.size else None,
        "frac_within_tolerance": within / len(residuals) if residuals else None,
        "tolerance": tolerance,
    }


def _derive_data_label(labels: Sequence[str]) -> str:
    """Provenance gate: every declared input label must agree, fail closed."""
    distinct = {label for label in labels}
    if len(distinct) > 1:
        raise ValueError(
            "basis-carry inputs carry mixed data_label values "
            f"{sorted(distinct)} — run mixed corpora as separate receipts"
        )
    return next(iter(distinct)) if distinct else "UNKNOWN"


def run_basis_carry(
    *,
    spot: pl.DataFrame,
    spot_label: str,
    contracts: Sequence[CarryContractInput],
    tolerance: float = DEFAULT_TOLERANCE,
    dte_buckets: Sequence[float] = DTE_BUCKETS,
    min_overlap: int = MIN_OVERLAP_DATES,
    dte_floor_days: float = DTE_FLOOR_DAYS,
    annualization_days: float = ANNUALIZATION_DAYS,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Run the settlement-anchored basis bench and return (frame, receipt).

    ``spot``/each contract frame need a ``event_time`` (or ``date``) column
    and a ``close`` column — collected source frames qualify as-is. Spot
    problems raise (shared input); per-contract problems record a
    ``status="error"`` result row so the sealed receipt documents the
    failure rather than dropping it.
    """
    if not contracts:
        raise ValueError("basis-carry needs at least one contract input")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must be a fraction in (0, 1)")
    if min_overlap < 2:
        raise ValueError("min_overlap must be >= 2")
    if dte_floor_days <= 0.0 or annualization_days <= 0.0:
        raise ValueError("dte_floor_days and annualization_days must be positive")
    buckets = tuple(sorted({float(b) for b in dte_buckets}, reverse=True))
    if not buckets or any(b <= 0 for b in buckets):
        raise ValueError("dte_buckets must be positive day counts")

    spot_dates, spot_closes = _series_from_frame(spot, name="spot")
    spot_by_date = dict(zip(spot_dates, spot_closes, strict=True))
    data_label = _derive_data_label([spot_label, *(contract.data_label for contract in contracts)])
    params: dict[str, Any] = {
        "tolerance": tolerance,
        "min_overlap": min_overlap,
        "dte_floor_days": dte_floor_days,
        "dte_buckets": list(buckets),
        "annualization_days": annualization_days,
        "convergence_note": (
            "convergence_residual is the basis at the last observed shared "
            "date; for contracts still listed it is measured "
            "residual_dte_days before delivery, not at settlement"
        ),
    }
    rows = [
        _contract_row(
            contract,
            spot_dates=spot_dates,
            spot_by_date=spot_by_date,
            params=params,
        )
        for contract in sorted(contracts, key=lambda item: item.symbol)
    ]
    cross = _cross_contract(rows, tolerance=tolerance)

    spot_digest = _series_digest(spot_dates, spot_closes)
    contract_meta: dict[str, Any] = {}
    contract_digests: dict[str, str] = {}
    for contract in sorted(contracts, key=lambda item: item.symbol):
        symbol = contract.symbol
        try:
            dates, closes = _series_from_frame(contract.frame, name=symbol)
            contract_digests[symbol] = _series_digest(dates, closes)
            meta: dict[str, Any] = {
                "sha256": contract_digests[symbol],
                "n_rows": len(dates),
                "first_date": dates[0].isoformat(),
                "last_date": dates[-1].isoformat(),
            }
        except ValueError:
            contract_digests[symbol] = "INVALID"
            meta = {"sha256": "INVALID", "n_rows": 0}
        meta["data_label"] = contract.data_label
        try:
            meta["delivery"] = _parse_delivery(contract.delivery, symbol=symbol).isoformat()
        except ValueError:
            meta["delivery"] = None
        contract_meta[symbol] = meta
    inputs_block = {
        "spot": {
            "sha256": spot_digest,
            "n_rows": len(spot_dates),
            "first_date": spot_dates[0].isoformat(),
            "last_date": spot_dates[-1].isoformat(),
            "data_label": spot_label,
        },
        "contracts": contract_meta,
        "params": params,
    }
    inputs_sha256 = hash_bytes(canonical_json_bytes(inputs_block))
    dataset_sha256 = hash_bytes(
        canonical_json_bytes({"spot": spot_digest, "contracts": contract_digests})
    )

    n_ok = cross["n_contracts"]
    n_delivered = cross["n_delivered"]
    frac = cross["frac_within_tolerance"]
    mean_residual = cross["residual_mean"]
    verdict = (
        f"{n_ok}/{len(rows)} contracts measured, {n_delivered} delivered; "
        f"residual mean {mean_residual:+.5%} max |residual| "
        f"{cross['residual_max_abs']:.5%}; "
        f"{frac:.0%} within tolerance {tolerance:.2%}"
        if n_ok
        and mean_residual is not None
        and cross["residual_max_abs"] is not None
        and frac is not None
        else f"{n_ok}/{len(rows)} contracts measured; no convergence residuals"
    )
    receipt: dict[str, Any] = {
        "schema": BASIS_CARRY_SCHEMA,
        "kind": BASIS_CARRY_KIND,
        "claim": "research_only",
        "research_only": True,
        "live_pnl_claim": False,
        "simulated_only": False,
        "data_label": data_label,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(),
        "params": params,
        "inputs": inputs_block,
        "inputs_sha256": inputs_sha256,
        "dataset_sha256": dataset_sha256,
        "n_rows": len(rows),
        "n_error_rows": sum(1 for row in rows if row["status"] != "ok"),
        "n_contracts": len(rows),
        "results": rows,
        "cross_contract": cross,
        "verdict": verdict,
    }
    frame = pl.DataFrame(
        {
            k: [row.get(k) for row in rows]
            for k in (
                "contract",
                "status",
                "delivery",
                "n_dates",
                "first_date",
                "last_date",
                "delivered",
                "residual_dte_days",
                "convergence_residual",
                "basis_diff_std",
                "ann_basis_mean",
                "ann_basis_std",
                "ann_basis_n",
            )
        }
    )
    return frame, receipt


def basis_carry_contract_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Fail-closed contract for a ``basis_carry.v1`` payload (writer + verifier)."""
    errors: list[str] = []
    research_blob = {key: value for key, value in receipt.items() if key != "live_pnl_claim"}
    if receipt.get("schema") != BASIS_CARRY_SCHEMA:
        errors.append("schema_not_basis_carry_v1")
    if receipt.get("kind") != BASIS_CARRY_KIND:
        errors.append("kind_not_basis_carry")
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
    spot_meta = inputs.get("spot")
    contract_meta = inputs.get("contracts")
    input_metas: list[Any] = []
    if isinstance(spot_meta, Mapping):
        input_metas.append(spot_meta)
    if isinstance(contract_meta, Mapping):
        input_metas.extend(m for m in contract_meta.values() if isinstance(m, Mapping))
    labels = {str(meta.get("data_label")) for meta in input_metas}
    if len(labels) != 1 or next(iter(labels), None) != receipt.get("data_label"):
        errors.append("data_label_not_derived")
    try:
        if hash_bytes(canonical_json_bytes(dict(inputs))) != receipt.get("inputs_sha256"):
            errors.append("inputs_sha256_mismatch")
    except (TypeError, ValueError):
        errors.append("inputs_sha256_uncomputable")
    if isinstance(spot_meta, Mapping) and isinstance(contract_meta, Mapping):
        try:
            expected_dataset = hash_bytes(
                canonical_json_bytes(
                    {
                        "spot": spot_meta.get("sha256"),
                        "contracts": {
                            str(sym): meta.get("sha256")
                            for sym, meta in contract_meta.items()
                            if isinstance(meta, Mapping)
                        },
                    }
                )
            )
        except (TypeError, ValueError):
            expected_dataset = "UNCOMPUTABLE"
        if expected_dataset != receipt.get("dataset_sha256"):
            errors.append("dataset_sha256_mismatch")
    else:
        errors.append("inputs_blocks_missing")

    results = receipt.get("results")
    if not isinstance(results, list) or not results:
        errors.append("results_missing_or_empty")
        return errors
    if receipt.get("n_rows") != len(results):
        errors.append("n_rows_mismatch")
    error_rows = [
        row for row in results if not isinstance(row, Mapping) or row.get("status") != "ok"
    ]
    if receipt.get("n_error_rows") != len(error_rows):
        errors.append("n_error_rows_mismatch")
    if receipt.get("n_contracts") != len(results):
        errors.append("n_contracts_mismatch")
    min_overlap = params.get("min_overlap")
    tolerance = params.get("tolerance")
    if not isinstance(min_overlap, int) or not isinstance(tolerance, (int, float)):
        errors.append("params_malformed")
        return errors
    for index, row in enumerate(results):
        if not isinstance(row, Mapping):
            errors.append(f"results[{index}]_not_object")
            continue
        symbol = row.get("contract")
        if row.get("status") != "ok":
            if not str(row.get("error", "")).strip():
                errors.append(f"results[{index}].error_empty")
            continue
        meta = contract_meta.get(str(symbol)) if isinstance(contract_meta, Mapping) else None
        if not isinstance(meta, Mapping) or meta.get("delivery") != row.get("delivery"):
            errors.append(f"results[{index}].delivery_mismatch")
        n_dates = row.get("n_dates")
        ann_n = row.get("ann_basis_n")
        if not isinstance(n_dates, int) or n_dates < min_overlap:
            errors.append(f"results[{index}].n_dates")
        if not isinstance(ann_n, int) or isinstance(n_dates, int) and ann_n > n_dates:
            errors.append(f"results[{index}].ann_basis_n")
        basis_lo, basis_hi = row.get("basis_min"), row.get("basis_max")
        for field in (
            "convergence_residual",
            "residual_dte_days",
            "basis_first",
            "basis_last",
            "basis_min",
            "basis_max",
        ):
            value = row.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                errors.append(f"results[{index}].{field}")
        if (
            isinstance(basis_lo, (int, float))
            and isinstance(basis_hi, (int, float))
            and not isinstance(basis_lo, bool)
            and not isinstance(basis_hi, bool)
        ):
            # Bucket means are interior observations: they must sit inside
            # the contract's own recorded basis range.
            for key, entry in (row.get("dte_buckets") or {}).items():
                if not isinstance(entry, Mapping):
                    if entry is not None:
                        errors.append(f"results[{index}].dte_buckets[{key}]")
                    continue
                bucket_basis = entry.get("basis")
                bucket_dte = entry.get("dte_days")
                if (
                    not isinstance(bucket_basis, (int, float))
                    or not math.isfinite(float(bucket_basis))
                    or not isinstance(bucket_dte, (int, float))
                    or not math.isfinite(float(bucket_dte))
                ):
                    errors.append(f"results[{index}].dte_buckets[{key}]_malformed")
                    continue
                try:
                    if bucket_dte > float(key) + 1e-9:
                        errors.append(f"results[{index}].dte_buckets[{key}]_dte")
                except ValueError:
                    errors.append(f"results[{index}].dte_buckets[{key}]_key")
                if not basis_lo - 1e-12 <= bucket_basis <= basis_hi + 1e-12:
                    errors.append(f"results[{index}].dte_buckets[{key}]_basis")
        delivery_raw = row.get("delivery")
        last_date_raw = row.get("last_date")
        first_date_raw = row.get("first_date")
        try:
            delivery_dt = parse_time(str(delivery_raw))
            last_day = date.fromisoformat(str(last_date_raw))
            first_day = date.fromisoformat(str(first_date_raw))
            if first_day > last_day:
                errors.append(f"results[{index}].date_order")
            expected_delivered = last_day >= delivery_dt.date()
            if row.get("delivered") is not expected_delivered:
                errors.append(f"results[{index}].delivered")
            expected_dte = (
                delivery_dt - datetime.combine(last_day, time(0, 0), tzinfo=UTC)
            ).total_seconds() / 86_400.0
            if not math.isclose(float(row["residual_dte_days"]), expected_dte, rel_tol=1e-9):
                errors.append(f"results[{index}].residual_dte_days")
        except (TypeError, ValueError, OverflowError, OSError, KeyError):
            errors.append(f"results[{index}].dates_unparseable")

    cross = receipt.get("cross_contract")
    if not isinstance(cross, Mapping):
        errors.append("cross_contract_missing")
        return errors
    ok_rows = [row for row in results if isinstance(row, Mapping) and row.get("status") == "ok"]
    residuals = cross.get("residuals")
    if not isinstance(residuals, list) or len(residuals) != len(ok_rows):
        errors.append("cross_contract_residuals")
    else:
        expected = [row.get("convergence_residual") for row in ok_rows]
        for index, (claimed, wanted) in enumerate(zip(residuals, expected, strict=False)):
            if (
                not isinstance(claimed, (int, float))
                or not isinstance(wanted, (int, float))
                or not math.isclose(float(claimed), float(wanted), rel_tol=1e-9)
            ):
                errors.append(f"cross_contract.residuals[{index}]")
        if all(isinstance(r, (int, float)) and not isinstance(r, bool) for r in residuals):
            array = np.asarray(residuals, dtype=float)
            for name, expected_value in (
                ("residual_mean", float(array.mean()) if array.size else None),
                ("residual_std", float(array.std(ddof=1)) if array.size >= 2 else None),
                ("residual_max_abs", float(np.abs(array).max()) if array.size else None),
            ):
                claimed = cross.get(name)
                if expected_value is None:
                    if claimed is not None:
                        errors.append(f"cross_contract.{name}")
                elif not isinstance(claimed, (int, float)) or not math.isclose(
                    float(claimed), expected_value, rel_tol=1e-9
                ):
                    errors.append(f"cross_contract.{name}")
            within = sum(1 for r in residuals if abs(float(r)) <= float(tolerance))
            expected_frac = within / len(residuals) if residuals else None
            claimed_frac = cross.get("frac_within_tolerance")
            if expected_frac is None:
                if claimed_frac is not None:
                    errors.append("cross_contract.frac_within_tolerance")
            elif not isinstance(claimed_frac, (int, float)) or not math.isclose(
                float(claimed_frac), expected_frac, rel_tol=1e-9
            ):
                errors.append("cross_contract.frac_within_tolerance")
    if cross.get("n_contracts") != len(ok_rows):
        errors.append("cross_contract.n_contracts")
    if cross.get("n_delivered") != sum(1 for row in ok_rows if row.get("delivered") is True):
        errors.append("cross_contract.n_delivered")
    return errors


def basis_carry_dataset_identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Input content digests bound by a v2 ``dataset_hash``."""
    inputs = receipt.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("basis-carry receipt has no inputs block")
    spot = inputs.get("spot")
    contracts = inputs.get("contracts")
    if not isinstance(spot, Mapping) or not isinstance(contracts, Mapping):
        raise ValueError("basis-carry receipt inputs lack spot/contracts blocks")
    return {
        "spot_sha256": spot.get("sha256"),
        "contracts": {
            str(symbol): meta.get("sha256")
            for symbol, meta in contracts.items()
            if isinstance(meta, Mapping)
        },
    }


def basis_carry_params(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """The run parameters bound by a v2 ``params_hash``."""
    params = receipt.get("params")
    return dict(params) if isinstance(params, Mapping) else {}


def basis_carry_verdict(receipt: Mapping[str, Any]) -> str:
    """``pass`` iff every contract was measured without an error row.

    This is an artifact-integrity verdict — whether the bench produced a
    complete measurement, not a market claim about the basis itself.
    """
    n_error_rows = receipt.get("n_error_rows")
    return "pass" if n_error_rows == 0 else "fail"


def basis_carry_receipt_v2(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap a ``basis_carry.v1`` payload in the unified ``receipt.v2`` envelope.

    The v1 payload is embedded verbatim under ``payload``; the envelope binds
    the input content digests, run params, this module's source hash, and the
    loaded numeric stack. Validates the v1 contract first — a malformed v1
    receipt is never wrapped.
    """
    from quant_fund.research.receipt_v2 import build_receipt_v2

    if basis_carry_contract_errors(receipt):
        raise ValueError("basis-carry receipt violates its research contract")
    return build_receipt_v2(
        kind=str(receipt["kind"]),
        data_label=str(receipt["data_label"]),
        dataset=basis_carry_dataset_identity(receipt),
        params=basis_carry_params(receipt),
        code_files=(Path(__file__),),
        verdict=basis_carry_verdict(receipt),
        payload=dict(receipt),
        generated_at=str(receipt["generated_at"]),
        revision=str(receipt["git_revision"]),
    )


def basis_carry_v2_consistency_errors(envelope: Mapping[str, Any]) -> list[str]:
    """Re-derive a basis-carry receipt.v2 envelope's bound digests from its payload."""
    errors: list[str] = []
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping):
        return ["payload_not_object"]
    contract_errors = basis_carry_contract_errors(payload)
    errors.extend(f"payload_{name}" for name in contract_errors)
    if contract_errors:
        return errors
    try:
        dataset = basis_carry_dataset_identity(payload)
    except ValueError as exc:
        return [*errors, f"payload_{exc}"]
    if hash_bytes(canonical_json_bytes(dataset)) != envelope.get("dataset_hash"):
        errors.append("dataset_hash_mismatch")
    if hash_bytes(canonical_json_bytes(basis_carry_params(payload))) != envelope.get("params_hash"):
        errors.append("params_hash_mismatch")
    if basis_carry_verdict(payload) != envelope.get("verdict"):
        errors.append("verdict_mismatch")
    return errors


def write_basis_carry_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
    *,
    receipt_version: int = 1,
) -> Path:
    """Seal a basis-carry receipt and write ``receipts/basis_carry_<hash>.json``.

    The filename hash is the sha256 of the canonical receipt payload; the
    same digest is embedded as ``receipt_sha256`` (the sealed-evidence
    convention). The write is atomic and fail-closed on tampering.
    ``receipt_version=2`` wraps the v1 payload in the unified ``receipt.v2``
    envelope before sealing.
    """
    from quant_fund.research.receipt_v2 import seal_receipt

    if receipt_version == 1:
        if basis_carry_contract_errors(receipt):
            raise ValueError("basis-carry receipt violates its research contract")
        body: Mapping[str, Any] = receipt
    elif receipt_version == 2:
        body = basis_carry_receipt_v2(receipt)
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    payload = seal_receipt(body)
    path = Path(receipts_dir) / f"basis_carry_{payload['receipt_sha256'][:16]}.json"
    publish_text_once(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


__all__ = [
    "ANNUALIZATION_DAYS",
    "BASIS_CARRY_KIND",
    "BASIS_CARRY_SCHEMA",
    "DEFAULT_TOLERANCE",
    "DTE_BUCKETS",
    "DTE_FLOOR_DAYS",
    "MIN_OVERLAP_DATES",
    "CarryContractInput",
    "basis_carry_contract_errors",
    "basis_carry_dataset_identity",
    "basis_carry_params",
    "basis_carry_receipt_v2",
    "basis_carry_v2_consistency_errors",
    "basis_carry_verdict",
    "kraken_delivery_from_symbol",
    "run_basis_carry",
    "write_basis_carry_receipt",
]
