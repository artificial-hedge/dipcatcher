"""Score user-supplied forecasts with proper scores and seal a receipt.v2.

``score_forecasts`` reads a CSV or Parquet file the operator supplies and
writes ``forecast_scoring_<sha16>.json`` under the chosen output directory.
The sealed envelope uses kind ``forecast_scoring.v1``, ``research_only``
true, and ``live_pnl_claim`` false. Headline numbers are proper scores only.

The receipt clock is the fixed replay timestamp ``1970-01-01T00:00:00+00:00``
so the same input bytes, arguments, code, and environment seal to the same
bytes. Paths are not part of the seal. Dataset identity is the file sha256,
size, row count, and column map, stored under names that are not the
tape-registry keys ``dataset_sha256`` / ``tape_manifest_sha256``. A
non-synthetic label plus either of those keys would fail ``verify-receipt``
because a user file is not a committed tape.

``available_time``, when provided, must be strictly earlier than the realized
timestamp on every row. Equal timestamps are lookahead and are refused.
Missing issue times still score, with a limitation that point-in-time
validity was not checked.

Failure is closed: missing files, bad extensions, samples shorter than 30
rows, files over 200 MB or 2,000,000 rows, non-finite values, non-positive
variances, probabilities outside ``[0, 1]``, non-binary probability
outcomes, crossing quantiles, and timestamps that are not strictly
increasing raise ``ForecastScoringError`` and do not leave a new receipt.
``run_selftest`` never raises.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import polars as pl

FORECAST_SCORING_KIND = "forecast_scoring.v1"
PROFILE_SCHEMA = "dataset_profile.v1"
REPLAY_GENERATED_AT = "1970-01-01T00:00:00+00:00"
TOOL_NAME = "quant_fund.research.forecast_scoring"

MIN_ROWS = 30
MAX_ROWS = 2_000_000
MAX_BYTES = 200_000_000
N_BOOTSTRAP = 1000
N_TIME_BLOCKS = 10
ECE_BINS = 10
LOG_LOSS_EPS = 1e-12
SEED_MAX = 2_147_483_647
# Ties within this tolerance are monotone; smaller negative gaps are crossings.
_CROSS_TOLERANCE = 1e-12
_SELFTEST_ROWS = 200
_QUANTILE_NAME = re.compile(r"^q(\d+)$")
_TIME_NAMES = ("timestamp", "event_time", "time", "date", "datetime")
_REALIZED_NAMES = ("realized", "realized_var", "rv", "target", "y")
_FORECAST_VAR_NAMES = ("forecast_var", "variance", "var_forecast", "forecast")
_FORECAST_PROB_NAMES = ("forecast_prob", "probability", "prob")
_AVAILABLE_NAMES = ("available_time", "available_at", "issue_time", "asof_time")
_TASKS = frozenset({"volatility", "quantile", "probability"})
_CONFIDENCE_NOTE = (
    "Suggested roles are column-name heuristics only. They are not validation. "
    "Pass explicit column flags to score-forecasts."
)
_PIT_LIMITATION = "Forecast issue time was not provided; point-in-time validity is NOT verified."


class ForecastScoringError(ValueError):
    """User data or arguments cannot be scored. The message names the column and row, not the frame."""


@dataclass(frozen=True)
class ScoringOutcome:
    """Sealed scoring result. ``scores`` holds finite headline proper scores only."""

    receipt_path: Path
    receipt_sha256: str
    scores: dict[str, float]
    rows: int
    data_label: str
    task: str


@dataclass(frozen=True)
class SelftestResult:
    """Verdict from ``run_selftest``. ``passed`` is false when verification or the tamper check fails."""

    passed: bool
    task: str
    receipt_sha256: str | None
    verified: bool
    detail: str
    duration_s: float


def _frozen_outcome(
    receipt_path: Path,
    receipt_sha256: str,
    scores: dict[str, float],
    rows: int,
    data_label: str,
    task: str,
) -> ScoringOutcome:
    return ScoringOutcome(receipt_path, receipt_sha256, dict(scores), rows, data_label, task)


def inspect_dataset(path: Path) -> dict[str, object]:
    """Read-only ``dataset_profile.v1`` profile. Writes nothing and does not score.

    Raises ``ForecastScoringError`` for a missing file, an unsupported
    extension, an empty file, a file over the size cap, or a table over the
    row cap. Samples shorter than 30 rows are still profiled.
    """
    data_path = Path(path)
    size = _require_file(data_path)
    frame = _read_table(data_path)
    _require_unique_columns(frame)
    return _profile(data_path, size, frame)


def score_forecasts(
    data_path: Path,
    *,
    task: str,
    out_dir: Path,
    time_col: str = "timestamp",
    realized_col: str = "realized",
    forecast_col: str | None = None,
    quantile_cols: str | None = None,
    available_time_col: str | None = None,
    seed: int = 0,
    receipt_version: int = 2,
) -> ScoringOutcome:
    """Score ``data_path`` and atomically write a sealed ``receipt.v2``.

    ``task`` is ``volatility``, ``quantile``, or ``probability``. ``quantile_cols``
    is a comma-separated ``q05=column`` list; omitted, quantile columns named
    ``q`` plus digits are detected. ``forecast_col`` defaults to ``forecast_var``
    or ``forecast_prob`` for those tasks. ``seed`` only affects the bootstrap
    interval. ``receipt_version`` must be 2.

    The same input bytes and arguments seal to the same receipt bytes on the
    same code and environment. An identical file already at the destination is
    left in place; a different file with that name is refused.
    """
    source = Path(data_path)
    destination = Path(out_dir)
    _require_task(task)
    _require_seed(seed)
    _require_version(receipt_version)
    _check_task_extras(task, forecast_col, quantile_cols)
    size = _require_file(source)
    frame = _read_table(source)
    _require_unique_columns(frame)
    _require_min_rows(frame.height)
    label = _resolve_label(source, frame)
    prepared = _prepare(
        frame,
        task=task,
        time_col=time_col,
        realized_col=realized_col,
        forecast_col=forecast_col,
        quantile_cols=quantile_cols,
        available_time_col=available_time_col,
    )
    scores, per_row, primary, extra, kupiec_gap = _dispatch(task, frame, prepared)
    diagnostics = _diagnostics(
        primary,
        per_row,
        seed,
        pit_column=prepared.available_time,
        pit_verified=prepared.available_time is not None,
        reliability=extra["reliability"],
        kupiec=extra["kupiec"],
        naive=prepared.naive_time,
    )
    limitations = _limitations(
        label=label,
        pit=prepared.available_time is not None,
        naive=prepared.naive_time,
        kupiec_undefined=kupiec_gap,
    )
    payload = _payload(
        source=source,
        size=size,
        frame=frame,
        label=label,
        task=task,
        rows=int(frame.height),
        prepared=prepared,
        scores=scores,
        diagnostics=diagnostics,
        limitations=limitations,
        seed=seed,
    )
    path, digest = _write_receipt(payload, destination)
    return _frozen_outcome(path, digest, scores, int(frame.height), label, task)


def run_selftest(task: str) -> SelftestResult:
    """Score a tiny deterministic SYNTHETIC fixture and verify the receipt.

    Also checks that a one-byte tamper of that receipt fails verification.
    Never raises. Unknown tasks return ``passed=False``.
    """
    started = time.perf_counter()
    try:
        passed, digest, verified, detail = _execute_selftest(task)
    except Exception as exc:
        passed = False
        digest = None
        verified = False
        detail = f"{exc.__class__.__name__}: {exc}"
    return SelftestResult(
        passed=passed,
        task=str(task),
        receipt_sha256=digest,
        verified=verified,
        detail=detail,
        duration_s=time.perf_counter() - started,
    )


def forecast_scoring_v2_consistency_errors(envelope: Mapping[str, Any]) -> list[str]:
    """Re-derive dataset and parameter digests for a ``forecast_scoring.v1`` envelope.

    Scores are not recomputed: the source rows are not inside the receipt.
    A byte edit that is not resealed still fails the envelope seal in
    ``verify_receipt_file``. This checker catches a reseal that drifts from
    the embedded dataset identity, parameter block, or honesty flags.
    """
    errors = _kind_errors(envelope)
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping):
        return [*errors, "payload_not_object"]
    errors.extend(_payload_flag_errors(payload, envelope))
    errors.extend(_binding_errors(payload, envelope))
    if not _forbidden_clean(payload):
        errors.append("forbidden_metric_keys")
    return errors


class _Prepared:
    __slots__ = (
        "time_name",
        "realized_name",
        "forecast_name",
        "available_time",
        "plan",
        "naive_time",
        "realized",
    )

    def __init__(
        self,
        time_name: str,
        realized_name: str,
        forecast_name: str | None,
        available_time: str | None,
        plan: list[tuple[str, float, str]],
        naive_time: bool,
        realized: np.ndarray,
    ) -> None:
        self.time_name = time_name
        self.realized_name = realized_name
        self.forecast_name = forecast_name
        self.available_time = available_time
        self.plan = plan
        self.naive_time = naive_time
        self.realized = realized


def _require_task(task: str) -> None:
    if task not in _TASKS:
        raise ForecastScoringError("task must be one of volatility, quantile, probability")


def _require_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ForecastScoringError("seed must be an integer in 0..2147483647")
    if seed < 0 or seed > SEED_MAX:
        raise ForecastScoringError("seed must be an integer in 0..2147483647")


def _require_version(receipt_version: int) -> None:
    if isinstance(receipt_version, bool) or receipt_version != 2:
        raise ForecastScoringError("receipt-version must be 2")


def _check_task_extras(task: str, forecast_col: str | None, quantile_cols: str | None) -> None:
    if task == "quantile" and forecast_col is not None:
        raise ForecastScoringError("forecast-col is not used for the quantile task")
    if task != "quantile" and quantile_cols is not None:
        raise ForecastScoringError("quantile-cols is only used for the quantile task")


def _require_file(path: Path) -> int:
    if not path.exists():
        raise ForecastScoringError(f"data file not found: {path}")
    if not path.is_file():
        raise ForecastScoringError(f"data path is not a file: {path}")
    size = int(path.stat().st_size)
    if size == 0:
        raise ForecastScoringError(f"data file is empty: {path}")
    if size > MAX_BYTES:
        raise ForecastScoringError(f"data file is {size} bytes; the limit is {MAX_BYTES} bytes")
    if path.suffix.lower() not in {".csv", ".parquet"}:
        raise ForecastScoringError(f"unsupported extension {path.suffix!r}; use .csv or .parquet")
    return size


def _read_table(path: Path) -> pl.DataFrame:
    import polars as pl

    _reject_duplicate_header(path)
    suffix = path.suffix.lower()
    try:
        frame = _read_supported(path, suffix)
    except pl.exceptions.DuplicateError:
        raise ForecastScoringError("data file has duplicate column names") from None
    except ForecastScoringError:
        raise
    except Exception as exc:
        raise ForecastScoringError(f"could not read {path.name}: {_short_error(exc)}") from None
    _require_max_rows(int(frame.height))
    return frame


def _reject_duplicate_header(path: Path) -> None:
    if path.suffix.lower() != ".csv":
        return
    header = _csv_header_line(path)
    if header is None:
        return
    names = next(csv.reader([header]))
    if len(names) == len(set(names)):
        return
    raise ForecastScoringError("data file has duplicate column names")


def _csv_header_line(path: Path) -> str | None:
    with path.open("rb") as handle:
        blob = handle.read(1 << 20)
    text = blob.decode("utf-8-sig", errors="replace")
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        return line
    return None


def _read_supported(path: Path, suffix: str) -> pl.DataFrame:
    import polars as pl

    if suffix == ".csv":
        return pl.read_csv(
            path,
            n_rows=MAX_ROWS + 1,
            infer_schema_length=None,
            try_parse_dates=True,
            comment_prefix="#",
        )
    return pl.scan_parquet(path).head(MAX_ROWS + 1).collect()


def _short_error(exc: BaseException) -> str:
    text = str(exc).splitlines()[0].strip() if str(exc) else ""
    if len(text) > 180:
        return text[:180] + "..."
    return text or exc.__class__.__name__


def _require_max_rows(n_rows: int) -> None:
    if n_rows > MAX_ROWS:
        raise ForecastScoringError(f"row count exceeds {MAX_ROWS}; found at least {n_rows} rows")


def _require_min_rows(n_rows: int) -> None:
    if n_rows < MIN_ROWS:
        raise ForecastScoringError(f"need at least {MIN_ROWS} rows; found {n_rows}")


def _require_unique_columns(frame: pl.DataFrame) -> None:
    names = [str(name) for name in frame.columns]
    if len(names) == len(set(names)):
        return
    raise ForecastScoringError("data file has duplicate column names")


def _resolve_label(path: Path, frame: pl.DataFrame) -> str:
    column = _column_label(frame)
    preamble = _csv_marked_synthetic(path)
    if column == "file" and preamble:
        raise ForecastScoringError(
            "data_label column says file but the file preamble says SYNTHETIC"
        )
    if column == "SYNTHETIC" or preamble:
        return "SYNTHETIC"
    return "file"


def _column_label(frame: pl.DataFrame) -> str | None:
    matches = [str(name) for name in frame.columns if str(name).lower() == "data_label"]
    if not matches:
        return None
    if len(matches) > 1:
        raise ForecastScoringError("multiple data_label columns")
    return _label_tokens(frame.get_column(matches[0]))


def _label_tokens(series: pl.Series) -> str | None:
    import polars as pl

    values = series.cast(pl.String)
    tokens = {
        str(item).strip().upper() for item in values.drop_nulls().to_list() if str(item).strip()
    }
    if not tokens:
        return None
    if tokens == {"SYNTHETIC"}:
        return "SYNTHETIC"
    if tokens == {"FILE"}:
        return "file"
    raise ForecastScoringError(
        "data_label column must be uniformly SYNTHETIC or file; other labels are refused"
    )


def _csv_marked_synthetic(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    with path.open("rb") as handle:
        blob = handle.read(8192)
    text = blob.decode("utf-8-sig", errors="replace")
    for line in text.splitlines():
        if not line.strip():
            continue
        if _line_marks_synthetic(line):
            return True
        if line.lstrip().startswith("#"):
            continue
        return False
    return False


def _line_marks_synthetic(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("#"):
        return False
    token = "".join(stripped[1:].split()).upper()
    return token in {"SYNTHETIC", "DATA_LABEL=SYNTHETIC"}


def _prepare(
    frame: pl.DataFrame,
    *,
    task: str,
    time_col: str,
    realized_col: str,
    forecast_col: str | None,
    quantile_cols: str | None,
    available_time_col: str | None,
) -> _Prepared:
    time_name = _clean_name(time_col, "time")
    realized_name = _clean_name(realized_col, "realized")
    issued_name = _optional_name(available_time_col, "available_time")
    forecast_name = _resolve_forecast_name(task, forecast_col)
    plan = _resolve_quantile_plan(task, frame, quantile_cols)
    stamps, naive_time = _load_time(frame, time_name)
    _require_increasing(stamps, time_name)
    naive_issued = _check_issued(frame, stamps, issued_name)
    realized = _float_column(frame, realized_name, "realized")
    return _Prepared(
        time_name,
        realized_name,
        forecast_name,
        issued_name,
        plan,
        naive_time or naive_issued,
        realized,
    )


def _optional_name(name: str | None, role: str) -> str | None:
    if name is None:
        return None
    return _clean_name(name, role)


def _clean_name(name: str, role: str) -> str:
    if not isinstance(name, str):
        raise ForecastScoringError(f"{role} column name must be a non-empty string")
    cleaned = name.strip()
    if not cleaned:
        raise ForecastScoringError(f"{role} column name must be a non-empty string")
    return cleaned


def _resolve_forecast_name(task: str, forecast_col: str | None) -> str | None:
    if task == "quantile":
        return None
    if forecast_col is None and task == "volatility":
        return "forecast_var"
    if forecast_col is None:
        return "forecast_prob"
    return _clean_name(forecast_col, "forecast")


def _resolve_quantile_plan(
    task: str, frame: pl.DataFrame, spec: str | None
) -> list[tuple[str, float, str]]:
    if task != "quantile":
        return []
    mapping = _autodetect_quantiles(frame) if spec is None else _parse_quantile_spec(spec)
    if not mapping:
        raise ForecastScoringError(
            "quantile task needs columns named like q05, or --quantile-cols q05=column"
        )
    return _order_quantiles(mapping)


def _autodetect_quantiles(frame: pl.DataFrame) -> dict[str, str]:
    found: dict[str, str] = {}
    for name in frame.columns:
        text = str(name)
        if _QUANTILE_NAME.fullmatch(text) is None:
            continue
        found[text] = text
    return found


def _parse_quantile_spec(spec: str) -> dict[str, str]:
    if not isinstance(spec, str) or not spec.strip():
        raise ForecastScoringError("quantile-cols must look like q05=column,q50=column")
    mapping: dict[str, str] = {}
    for part in spec.split(","):
        _add_quantile_pair(mapping, part)
    if not mapping:
        raise ForecastScoringError("quantile-cols did not name any columns")
    return mapping


def _add_quantile_pair(mapping: dict[str, str], part: str) -> None:
    piece = part.strip()
    if not piece:
        return
    token, sep, column = piece.partition("=")
    token = token.strip()
    column = column.strip()
    if not sep or not token or not column:
        raise ForecastScoringError("quantile-cols must look like q05=column,q50=column")
    if token in mapping:
        raise ForecastScoringError(f"duplicate quantile token {token!r}")
    _tau_from_token(token)
    mapping[token] = column


def _tau_from_token(token: str) -> float:
    """``q`` plus digits is that integer over 10^digits: q05 -> 0.05, q5 -> 0.5."""
    match = _QUANTILE_NAME.fullmatch(token)
    if match is None:
        raise ForecastScoringError(
            f"quantile token {token!r} must look like q05 (q followed by digits)"
        )
    digits = str(match.group(1))
    tau = float(int(digits) / (10 ** len(digits)))
    if not 0.0 < tau < 1.0:
        raise ForecastScoringError(f"quantile token {token!r} is outside (0, 1)")
    return tau


def _order_quantiles(mapping: dict[str, str]) -> list[tuple[str, float, str]]:
    rows: list[tuple[float, str, str]] = []
    seen: dict[str, str] = {}
    for token, column in mapping.items():
        tau = _tau_from_token(token)
        key = format(tau, ".12f")
        if key in seen:
            raise ForecastScoringError(
                f"quantile tokens {seen[key]!r} and {token!r} are the same tau"
            )
        seen[key] = token
        rows.append((tau, token, column))
    rows.sort()
    return [(token, tau, column) for tau, token, column in rows]


def _load_time(frame: pl.DataFrame, column: str) -> tuple[pl.Series, bool]:
    if column not in frame.columns:
        raise ForecastScoringError(f"missing column {column!r} for time")
    return _utc_series(frame.get_column(column), column)


def _check_issued(frame: pl.DataFrame, realized: pl.Series, issued_name: str | None) -> bool:
    if issued_name is None:
        return False
    issued, naive = _load_time(frame, issued_name)
    _require_earlier(issued, realized)
    return naive


def _utc_series(series: pl.Series, column: str) -> tuple[pl.Series, bool]:
    import polars as pl

    if int(series.null_count()) > 0:
        raise ForecastScoringError(_null_message(series, column))
    dtype = series.dtype
    if isinstance(dtype, pl.Datetime):
        return _datetime_to_utc(series), dtype.time_zone is None
    if dtype == pl.Date:
        stamped = series.cast(pl.Datetime(time_unit="us")).dt.replace_time_zone("UTC")
        return stamped, True
    if dtype == pl.String:
        return _string_to_utc(series, column)
    raise ForecastScoringError(
        f"column {column!r} must be a timestamp (datetime, date, or text); got {dtype}"
    )


def _datetime_to_utc(series: pl.Series) -> pl.Series:
    import polars as pl

    dtype = series.dtype
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
        return series.dt.replace_time_zone("UTC")
    return series.dt.convert_time_zone("UTC")


def _string_to_utc(series: pl.Series, column: str) -> tuple[pl.Series, bool]:
    naive = _strings_look_naive(series)
    parsed = _parse_timestamp_text(series, column)
    if int(parsed.null_count()):
        raise ForecastScoringError(_null_message(parsed, column))
    return parsed, naive


def _parse_timestamp_text(series: pl.Series, column: str) -> pl.Series:
    """Vectorized parse, then a per-value fallback for mixed offsets Polars rejects."""
    try:
        return series.str.to_datetime(time_zone="UTC", strict=True)
    except Exception:
        return _parse_timestamps_python(series, column)


def _parse_timestamps_python(series: pl.Series, column: str) -> pl.Series:
    import polars as pl

    values = [_one_timestamp(item, column, index) for index, item in enumerate(series.to_list())]
    return pl.Series(series.name, values, dtype=pl.Datetime(time_unit="us", time_zone="UTC"))


def _one_timestamp(item: object, column: str, index: int) -> object:
    from datetime import UTC, datetime

    text = str(item).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        raise ForecastScoringError(
            f"column {column!r} has values that are not timestamps (first at row {index})"
        ) from None
    if stamp.tzinfo is None:
        return stamp.replace(tzinfo=UTC)
    return stamp.astimezone(UTC)


def _strings_look_naive(series: pl.Series) -> bool:
    zoned = series.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})\s*$")
    filled = zoned.fill_null(False)
    return not bool(filled.all())


def _require_increasing(stamps: pl.Series, column: str) -> None:
    import numpy as np

    ns = _epoch_ns(stamps)
    if int(ns.size) < 2:
        return
    bad = np.flatnonzero(np.diff(ns) <= 0)
    if int(bad.size) == 0:
        return
    row = int(bad[0]) + 1
    raise ForecastScoringError(
        f"timestamps in {column!r} must be strictly increasing; "
        f"row {row} is not after row {row - 1}"
    )


def _require_earlier(issued: pl.Series, realized: pl.Series) -> None:
    import numpy as np

    bad = np.flatnonzero(_epoch_ns(issued) >= _epoch_ns(realized))
    if int(bad.size) == 0:
        return
    row = int(bad[0])
    raise ForecastScoringError(
        "available_time must be strictly earlier than the realized timestamp; "
        f"row {row} is not (lookahead)"
    )


def _epoch_ns(stamps: pl.Series) -> np.ndarray:
    import numpy as np

    return np.asarray(stamps.dt.timestamp("ns").to_numpy(), dtype=np.int64)


def _float_column(frame: pl.DataFrame, name: str, role: str) -> np.ndarray:
    if name not in frame.columns:
        raise ForecastScoringError(f"missing column {name!r} for {role}")
    series = frame.get_column(name)
    if int(series.null_count()):
        raise ForecastScoringError(_null_message(series, name))
    values = _coerce_float(series, name, role)
    _reject_nonfinite(values, name)
    return values


def _coerce_float(series: pl.Series, name: str, role: str) -> np.ndarray:
    if _dtype_is_numeric(series.dtype):
        return _as_float64(series, name, role)
    if _is_string_dtype(series.dtype):
        return _float_from_strings(series, name, role)
    raise ForecastScoringError(f"column {name!r} must be numeric ({role})")


def _is_string_dtype(dtype: object) -> bool:
    import polars as pl

    return dtype == pl.String


def _float_from_strings(series: pl.Series, name: str, role: str) -> np.ndarray:
    """Parse text numbers. ``nan``/``inf`` tokens stay non-finite so the finite check can refuse them."""
    import polars as pl

    casted = series.cast(pl.Float64, strict=False)
    if int(casted.null_count()):
        raise ForecastScoringError(f"column {name!r} must be numeric ({role})")
    return _as_float64(casted, name, role)


def _dtype_is_numeric(dtype: object) -> bool:
    check = getattr(dtype, "is_numeric", None)
    if not callable(check):
        return False
    return bool(check())


def _as_float64(series: pl.Series, name: str, role: str) -> np.ndarray:
    import numpy as np

    try:
        return np.asarray(series.to_numpy(), dtype=np.float64)
    except (TypeError, ValueError):
        raise ForecastScoringError(f"column {name!r} must be numeric ({role})") from None


def _null_message(series: pl.Series, column: str) -> str:
    count = int(series.null_count())
    first = _first_null_row(series)
    return f"column {column!r} has {count} nulls (first at row {first})"


def _first_null_row(series: pl.Series) -> int:
    import numpy as np

    flags = np.asarray(series.is_null().to_numpy(), dtype=bool)
    found = np.flatnonzero(flags)
    if int(found.size) == 0:
        return 0
    return int(found[0])


def _reject_nonfinite(values: np.ndarray, column: str) -> None:
    import numpy as np

    bad = ~np.isfinite(values)
    if not bool(bad.any()):
        return
    raise ForecastScoringError(
        "values must be finite (no NaN or inf); "
        f"column {column!r} has {int(bad.sum())} bad values "
        f"(first at row {int(np.flatnonzero(bad)[0])})"
    )


def _reject_nonpositive(values: np.ndarray, column: str, *, what: str) -> None:
    import numpy as np

    bad = ~(values > 0.0)
    if not bool(bad.any()):
        return
    raise ForecastScoringError(
        f"{what} must be positive; column {column!r} has {int(bad.sum())} "
        f"non-positive values (first at row {int(np.flatnonzero(bad)[0])})"
    )


def _reject_outside_unit(values: np.ndarray, column: str) -> None:
    import numpy as np

    bad = (values < 0.0) | (values > 1.0)
    if not bool(bad.any()):
        return
    raise ForecastScoringError(
        "probabilities must lie in [0, 1]; "
        f"column {column!r} has {int(bad.sum())} out-of-range values "
        f"(first at row {int(np.flatnonzero(bad)[0])})"
    )


def _reject_nonbinary(values: np.ndarray, column: str) -> None:
    import numpy as np

    bad = (values != 0.0) & (values != 1.0)
    if not bool(bad.any()):
        return
    raise ForecastScoringError(
        "probability outcomes must be 0 or 1; "
        f"column {column!r} has {int(bad.sum())} other values "
        f"(first at row {int(np.flatnonzero(bad)[0])})"
    )


def _dispatch(
    task: str, frame: pl.DataFrame, prepared: _Prepared
) -> tuple[dict[str, float], np.ndarray, str, dict[str, Any], bool]:
    if task == "volatility":
        return _volatility_bundle(frame, prepared)
    if task == "probability":
        return _probability_bundle(frame, prepared)
    return _quantile_bundle(frame, prepared)


def _volatility_bundle(
    frame: pl.DataFrame, prepared: _Prepared
) -> tuple[dict[str, float], np.ndarray, str, dict[str, Any], bool]:
    forecast_name = prepared.forecast_name
    if forecast_name is None:
        raise ForecastScoringError("volatility requires a forecast column")
    forecast = _float_column(frame, forecast_name, "forecast variance")
    _reject_nonpositive(prepared.realized, prepared.realized_name, what="realized variance proxies")
    _reject_nonpositive(forecast, forecast_name, what="variance forecasts")
    scores, per_row = _score_volatility(prepared.realized, forecast)
    _require_finite_scores(scores)
    return scores, per_row, "qlike", {"reliability": None, "kupiec": []}, False


def _probability_bundle(
    frame: pl.DataFrame, prepared: _Prepared
) -> tuple[dict[str, float], np.ndarray, str, dict[str, Any], bool]:
    forecast_name = prepared.forecast_name
    if forecast_name is None:
        raise ForecastScoringError("probability requires a forecast column")
    forecast = _float_column(frame, forecast_name, "forecast probability")
    _reject_nonbinary(prepared.realized, prepared.realized_name)
    _reject_outside_unit(forecast, forecast_name)
    scores, per_row, reliability = _score_probability(prepared.realized, forecast)
    _require_finite_scores(scores)
    return scores, per_row, "brier", {"reliability": reliability, "kupiec": []}, False


def _quantile_bundle(
    frame: pl.DataFrame, prepared: _Prepared
) -> tuple[dict[str, float], np.ndarray, str, dict[str, Any], bool]:
    matrix, taus, tokens = _quantile_matrix(frame, prepared.plan)
    scores, per_row, kupiec_rows = _score_quantiles(prepared.realized, matrix, taus, tokens)
    _require_finite_scores(scores)
    undefined = any(row.get("p_value") is None for row in kupiec_rows)
    return scores, per_row, "mean_pinball", {"reliability": None, "kupiec": kupiec_rows}, undefined


def _score_volatility(
    realized: np.ndarray, forecast: np.ndarray
) -> tuple[dict[str, float], np.ndarray]:
    import numpy as np

    from quant_fund.metrics.vol_eval import mse, qlike

    per_row = np.asarray(qlike(realized, forecast), dtype=np.float64)
    squared = np.asarray(mse(realized, forecast), dtype=np.float64)
    scores = {
        "qlike": float(np.mean(per_row)),
        "mse": float(np.mean(squared)),
        "calibration_ratio": float(np.mean(realized / forecast)),
    }
    return scores, per_row


def _score_probability(
    y: np.ndarray, prob: np.ndarray
) -> tuple[dict[str, float], np.ndarray, list[dict[str, object]]]:
    from quant_fund.metrics.probability import brier_score, expected_calibration_error, log_loss

    per_row = (prob - y) ** 2
    scores = {
        "brier": float(brier_score(prob, y)),
        "log_loss": float(log_loss(prob, y, eps=LOG_LOSS_EPS)),
        "ece": float(expected_calibration_error(prob, y, n_bins=ECE_BINS)),
    }
    return scores, per_row, _reliability(prob, y)


def _score_quantiles(
    y: np.ndarray, matrix: np.ndarray, taus: np.ndarray, tokens: list[str]
) -> tuple[dict[str, float], np.ndarray, list[dict[str, object]]]:
    import numpy as np

    per_tau = _per_tau_pinball(y, matrix, taus, tokens)
    scores: dict[str, float] = {}
    for token, loss in per_tau:
        scores[f"pinball_{token}"] = float(np.mean(loss))
    primary = np.mean(np.stack([loss for _, loss in per_tau], axis=0), axis=0)
    scores["mean_pinball"] = float(np.mean(primary))
    scores["crps"] = _crps(y, matrix, taus)
    for index, token in enumerate(tokens):
        scores[f"coverage_{token}"] = float(np.mean(y <= matrix[:, index]))
    kupiec_rows: list[dict[str, object]] = []
    for index, token in enumerate(tokens):
        row = _kupiec_row(token, float(taus[index]), y, matrix[:, index])
        if row is None:
            continue
        kupiec_rows.append(row)
        _add_kupiec_scores(scores, row)
    return scores, primary, kupiec_rows


def _per_tau_pinball(
    y: np.ndarray, matrix: np.ndarray, taus: np.ndarray, tokens: list[str]
) -> list[tuple[str, np.ndarray]]:
    import numpy as np

    from quant_fund.metrics.scoring import pinball_loss

    rows: list[tuple[str, np.ndarray]] = []
    for index, token in enumerate(tokens):
        loss = np.asarray(pinball_loss(y, matrix[:, index], float(taus[index])), dtype=np.float64)
        rows.append((token, loss))
    return rows


def _crps(y: np.ndarray, matrix: np.ndarray, taus: np.ndarray) -> float:
    from quant_fund.metrics.scoring import crps_from_quantiles

    return float(crps_from_quantiles(y, matrix, taus))


def _quantile_matrix(
    frame: pl.DataFrame, plan: list[tuple[str, float, str]]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import numpy as np

    tokens: list[str] = []
    taus: list[float] = []
    columns: list[np.ndarray] = []
    names: list[str] = []
    for token, tau, column in plan:
        columns.append(_float_column(frame, column, f"quantile {token}"))
        tokens.append(token)
        taus.append(tau)
        names.append(column)
    matrix = np.column_stack(columns)
    _reject_crossing(matrix, names)
    return matrix, np.asarray(taus, dtype=np.float64), tokens


def _reject_crossing(matrix: np.ndarray, names: list[str]) -> None:
    import numpy as np

    if matrix.shape[1] < 2:
        return
    bad = np.argwhere(np.diff(matrix, axis=1) < -_CROSS_TOLERANCE)
    if int(bad.size) == 0:
        return
    row = int(bad[0, 0])
    col = int(bad[0, 1])
    raise ForecastScoringError(
        f"quantile columns cross: {names[col]!r} is above {names[col + 1]!r} at row {row}"
    )


def _kupiec_row(token: str, tau: float, y: np.ndarray, q: np.ndarray) -> dict[str, object] | None:
    if tau == 0.5:
        return None
    hits, alpha, side = _tail_hits(tau, y, q)
    rate, statistic, p_value = _kupiec_numbers(hits, alpha)
    return {
        "token": token,
        "tau": tau,
        "alpha": alpha,
        "side": side,
        "hit_rate": _finite(rate),
        "statistic": _finite(statistic),
        "p_value": _finite(p_value),
    }


def _tail_hits(tau: float, y: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, float, str]:
    import numpy as np

    if tau < 0.5:
        return (y < q).astype(np.float64), tau, "lower"
    return (y > q).astype(np.float64), 1.0 - tau, "upper"


def _kupiec_numbers(hits: np.ndarray, alpha: float) -> tuple[float, float, float]:
    from quant_fund.metrics.probability import kupiec_pof

    rate, statistic, p_value = kupiec_pof(hits, alpha)
    return float(rate), float(statistic), float(p_value)


def _add_kupiec_scores(scores: dict[str, float], row: dict[str, object]) -> None:
    token = str(row["token"])
    statistic = row["statistic"]
    p_value = row["p_value"]
    if isinstance(statistic, float):
        scores[f"kupiec_lr_{token}"] = statistic
    if isinstance(p_value, float):
        scores[f"kupiec_pvalue_{token}"] = p_value


def _finite(value: float) -> float | None:
    number = float(value)
    if math.isfinite(number):
        return number
    return None


def _require_finite_scores(scores: dict[str, float]) -> None:
    for name, value in scores.items():
        if not math.isfinite(float(value)):
            raise ForecastScoringError(f"score {name} is not finite")


def _reliability(prob: np.ndarray, y: np.ndarray) -> list[dict[str, object]]:
    import numpy as np

    edges = np.linspace(0.0, 1.0, ECE_BINS + 1)
    rows: list[dict[str, object]] = []
    for index in range(ECE_BINS):
        rows.append(_reliability_row(prob, y, edges, index))
    return rows


def _reliability_row(
    prob: np.ndarray, y: np.ndarray, edges: np.ndarray, index: int
) -> dict[str, object]:
    low = float(edges[index])
    high = float(edges[index + 1])
    selected = _bin_mask(prob, low, high, last=index == ECE_BINS - 1)
    count = int(selected.sum())
    return {
        "bin": index,
        "low": low,
        "high": high,
        "count": count,
        "mean_forecast": None if count == 0 else float(prob[selected].mean()),
        "mean_outcome": None if count == 0 else float(y[selected].mean()),
    }


def _bin_mask(prob: np.ndarray, low: float, high: float, *, last: bool) -> np.ndarray:
    if last:
        return (prob >= low) & (prob <= high)
    return (prob >= low) & (prob < high)


def _diagnostics(
    primary_name: str,
    per_row: np.ndarray,
    seed: int,
    *,
    pit_column: str | None,
    pit_verified: bool,
    reliability: object,
    kupiec: object,
    naive: bool,
) -> dict[str, object]:
    import numpy as np

    blocks = _time_blocks(per_row)
    bootstrap = _bootstrap_interval(per_row, seed)
    bootstrap["primary"] = primary_name
    bootstrap["point"] = float(np.mean(per_row))
    return {
        "primary": primary_name,
        "blocks": blocks,
        "weakest_block": dict(_weakest_block(blocks)),
        "bootstrap": bootstrap,
        "point_in_time": {"column": pit_column, "verified": pit_verified},
        "reliability": reliability,
        "kupiec": kupiec,
        "naive_timestamps_as_utc": naive,
    }


def _time_blocks(losses: np.ndarray) -> list[dict[str, object]]:
    import numpy as np

    n = int(losses.shape[0])
    count = min(N_TIME_BLOCKS, n)
    base, extra = divmod(n, count)
    blocks: list[dict[str, object]] = []
    start = 0
    for index in range(count):
        length = base + (1 if index < extra else 0)
        stop = start + length
        blocks.append(
            {
                "index": index,
                "start": start,
                "end": stop,
                "n": length,
                "primary": float(np.mean(losses[start:stop])),
            }
        )
        start = stop
    return blocks


def _weakest_block(blocks: list[dict[str, object]]) -> dict[str, object]:
    chosen = blocks[0]
    for block in blocks[1:]:
        if _as_float(block.get("primary")) > _as_float(chosen.get("primary")):
            chosen = block
    return chosen


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ForecastScoringError("block score is not numeric")
    return float(value)


def _bootstrap_interval(losses: np.ndarray, seed: int) -> dict[str, object]:
    import numpy as np

    n = int(losses.shape[0])
    block = _block_length(n)
    per_resample = max(1, n // block)
    means = _circular_block_means(losses, block)
    rng = np.random.Generator(np.random.PCG64(seed))
    starts = rng.integers(0, n, size=(N_BOOTSTRAP, per_resample))
    samples = means[starts].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "method": "circular_block",
        "seed": int(seed),
        "n_resamples": N_BOOTSTRAP,
        "block_len": block,
        "blocks_per_resample": per_resample,
        "resampled_length": block * per_resample,
        "level": 0.95,
        "low": float(low),
        "high": float(high),
    }


def _block_length(n: int) -> int:
    # Hall / Horowitz / Jing rule of thumb: n^(1/3), then complete blocks only.
    block = int(round(n ** (1.0 / 3.0)))
    if block < 1:
        return 1
    if block > n:
        return n
    return block


def _circular_block_means(losses: np.ndarray, block: int) -> np.ndarray:
    import numpy as np

    n = int(losses.shape[0])
    tail = losses[: block - 1] if block > 1 else losses[:0]
    doubled = np.concatenate([losses, tail])
    cumulative = np.cumsum(doubled, dtype=np.float64)
    end = cumulative[block - 1 : block - 1 + n]
    previous = np.zeros(1, dtype=np.float64)
    if n > 1:
        previous = np.concatenate([previous, cumulative[: n - 1]])
    return (end - previous) / float(block)


def _limitations(*, label: str, pit: bool, naive: bool, kupiec_undefined: bool) -> list[str]:
    notes = [
        "Proper scores are research diagnostics. This receipt is not a live-trading claim.",
        (
            "The 95% interval is a seeded circular block bootstrap of complete blocks "
            "(block length round(n^(1/3))); it is a stability diagnostic, not a guarantee."
        ),
    ]
    if label == "SYNTHETIC":
        notes.append(
            "SYNTHETIC marks user-declared synthetic input. "
            "It is a correctness label, not market evidence."
        )
    else:
        notes.append("data_label file means operator-supplied bytes, not a pinned public tape.")
    if not pit:
        notes.append(_PIT_LIMITATION)
    if naive:
        notes.append("Naive timestamps were interpreted as UTC.")
    if kupiec_undefined:
        notes.append(
            "Kupiec p-value is undefined when a tail has zero or all hits; "
            "that tail's p-value is omitted."
        )
    return notes


def _payload(
    *,
    source: Path,
    size: int,
    frame: pl.DataFrame,
    label: str,
    task: str,
    rows: int,
    prepared: _Prepared,
    scores: dict[str, float],
    diagnostics: dict[str, object],
    limitations: list[str],
    seed: int,
) -> dict[str, Any]:
    dataset = _dataset_identity(source, size, rows, frame, prepared)
    params = _params(task, prepared, seed)
    return {
        "schema": FORECAST_SCORING_KIND,
        "kind": FORECAST_SCORING_KIND,
        "claim": "research_only",
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": label,
        "task": task,
        "rows": rows,
        "dataset": dataset,
        "params": params,
        "scores": scores,
        "diagnostics": diagnostics,
        "limitations": limitations,
        "tool": TOOL_NAME,
    }


def _dataset_identity(
    source: Path, size: int, rows: int, frame: pl.DataFrame, prepared: _Prepared
) -> dict[str, Any]:
    return {
        "file_sha256": _sha256_file(source),
        "size_bytes": size,
        "n_rows": rows,
        "file_columns": _file_columns(frame),
        "resolved_columns": _resolved_columns(prepared),
    }


def _file_columns(frame: pl.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, dtype in zip(frame.columns, frame.dtypes, strict=True):
        series = frame.get_column(name)
        rows.append({"name": str(name), "dtype": str(dtype), "nulls": int(series.null_count())})
    return rows


def _resolved_columns(prepared: _Prepared) -> dict[str, Any]:
    quantiles = [
        {"token": token, "tau": tau, "column": column} for token, tau, column in prepared.plan
    ]
    return {
        "time": prepared.time_name,
        "realized": prepared.realized_name,
        "forecast": prepared.forecast_name,
        "available_time": prepared.available_time,
        "quantiles": quantiles,
    }


def _params(task: str, prepared: _Prepared, seed: int) -> dict[str, Any]:
    quantile_cols = [
        {"token": token, "tau": tau, "column": column} for token, tau, column in prepared.plan
    ]
    return {
        "task": task,
        "time_col": prepared.time_name,
        "realized_col": prepared.realized_name,
        "forecast_col": prepared.forecast_name,
        "quantile_cols": quantile_cols,
        "available_time_col": prepared.available_time,
        "seed": int(seed),
        "receipt_version": 2,
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap": "circular_block",
        "block_length_rule": "round(n**(1/3))",
        "ece_bins": ECE_BINS,
        "log_loss_eps": LOG_LOSS_EPS,
        "min_rows": MIN_ROWS,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_receipt(payload: dict[str, Any], out_dir: Path) -> tuple[Path, str]:
    from quant_fund.schemas.receipt_v2_leaf import build_receipt_v2, seal_receipt
    from quant_fund.utils.reproducibility import git_revision

    plain = _as_object_dict(_plain(payload))
    dataset = _as_object_dict(plain["dataset"])
    params = _as_object_dict(plain["params"])
    envelope = build_receipt_v2(
        kind=FORECAST_SCORING_KIND,
        data_label=str(plain["data_label"]),
        dataset=dataset,
        params=params,
        code_files=(Path(__file__),),
        verdict="pass",
        payload=plain,
        generated_at=REPLAY_GENERATED_AT,
        revision=git_revision() or "UNKNOWN",
    )
    sealed = seal_receipt(envelope)
    digest = str(sealed["receipt_sha256"])
    path = _receipt_destination(out_dir, digest)
    _publish(path, _dump_receipt(sealed))
    _require_valid_receipt(path)
    return path, digest


def _as_object_dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ForecastScoringError("receipt block is not an object")
    return value


def _receipt_destination(out_dir: Path, digest: str) -> Path:
    name = f"forecast_scoring_{digest[:16]}.json"
    path = out_dir / name
    if path.name != name or path.parent != out_dir:
        raise ForecastScoringError("receipt path escaped the output directory")
    return path


def _dump_receipt(sealed: Mapping[str, Any]) -> str:
    try:
        return json.dumps(sealed, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise ForecastScoringError(f"receipt is not JSON-safe: {exc.__class__.__name__}") from None


def _require_valid_receipt(path: Path) -> None:
    from quant_fund.schemas.receipt_v2_leaf import verify_receipt_file

    result = verify_receipt_file(path)
    if result["valid"] is True:
        return
    path.unlink(missing_ok=True)
    summary = "; ".join(str(item) for item in result["errors"][:8])
    raise ForecastScoringError(f"receipt failed verification: {summary}")


def _publish(path: Path, content: str) -> None:
    _ensure_out_dir(path.parent)
    if path.is_symlink():
        raise ForecastScoringError(f"receipt path is a symlink: {path.name}")
    if path.exists():
        _require_same_text(path, content)
        return
    temporary = _write_temp(path, content)
    try:
        _install_temp(temporary, path, content)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_out_dir(out_dir: Path) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ForecastScoringError(
            f"could not create output directory {out_dir}: {exc.__class__.__name__}"
        ) from None


def _write_temp(path: Path, content: str) -> Path:
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)


def _install_temp(temporary: Path, path: Path, content: str) -> None:
    try:
        os.link(temporary, path)
    except FileExistsError:
        _require_same_text(path, content)
    except OSError:
        _move_if_absent(temporary, path, content)


def _move_if_absent(temporary: Path, path: Path, content: str) -> None:
    if path.exists() or path.is_symlink():
        _require_same_text(path, content)
        return
    os.replace(temporary, path)


def _require_same_text(path: Path, content: str) -> None:
    if path.is_symlink():
        raise ForecastScoringError(f"receipt path is a symlink: {path.name}")
    try:
        existing = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ForecastScoringError(
            f"receipt path is not readable: {path.name} ({exc.__class__.__name__})"
        ) from None
    if existing != content:
        raise ForecastScoringError(f"receipt already exists with different content: {path.name}")


def _plain(value: object) -> object:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    raise ForecastScoringError(f"receipt value is not JSON-safe: {type(value).__name__}")


def _profile(path: Path, size: int, frame: pl.DataFrame) -> dict[str, object]:
    columns = [str(name) for name in frame.columns]
    roles = _suggest_roles(columns)
    time_column = _suggested_column(roles, "time")
    return {
        "schema": PROFILE_SCHEMA,
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": size,
        "n_rows": int(frame.height),
        "columns": _file_columns(frame),
        "time": _time_section(frame, time_column),
        "suggested_roles": roles,
        "confidence_note": _CONFIDENCE_NOTE,
    }


def _suggested_column(roles: dict[str, object], key: str) -> str | None:
    role = roles.get(key)
    if not isinstance(role, dict):
        return None
    column = role.get("column")
    if isinstance(column, str):
        return column
    return None


def _suggest_roles(columns: list[str]) -> dict[str, object]:
    time_col, time_confidence = _match_name(columns, _TIME_NAMES)
    realized, realized_confidence = _match_name(columns, _REALIZED_NAMES)
    forecast_var, var_confidence = _match_name(columns, _FORECAST_VAR_NAMES)
    forecast_prob, prob_confidence = _match_name(columns, _FORECAST_PROB_NAMES)
    available, available_confidence = _match_name(columns, _AVAILABLE_NAMES)
    return {
        "time": _role(time_col, time_confidence, "time of the realized observation"),
        "realized": _role(
            realized, realized_confidence, "realized variance, value, or 0/1 outcome"
        ),
        "forecast_variance": _role(forecast_var, var_confidence, "variance forecast"),
        "forecast_probability": _role(forecast_prob, prob_confidence, "event probability"),
        "available_time": _role(
            available,
            available_confidence,
            "when the forecast was issued; must be earlier than time",
        ),
        "quantiles": _suggested_quantiles(columns),
    }


def _suggested_quantiles(columns: list[str]) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    for name in columns:
        tau = _optional_tau(name)
        if tau is None:
            continue
        found.append({"column": name, "tau": tau, "confidence": "name_pattern"})
    return found


def _optional_tau(name: str) -> float | None:
    if _QUANTILE_NAME.fullmatch(name) is None:
        return None
    try:
        return _tau_from_token(name)
    except ForecastScoringError:
        return None


def _match_name(columns: list[str], names: tuple[str, ...]) -> tuple[str | None, str]:
    lookup: dict[str, str] = {}
    for column in columns:
        lookup.setdefault(column.lower(), column)
    for index, name in enumerate(names):
        found = lookup.get(name)
        if found is None:
            continue
        confidence = "exact_name" if index == 0 else "alias"
        return found, confidence
    return None, "absent"


def _role(column: str | None, confidence: str, note: str) -> dict[str, object]:
    return {"column": column, "confidence": confidence, "note": note}


def _time_section(frame: pl.DataFrame, column: str | None) -> dict[str, object]:
    if column is None:
        return _empty_time()
    try:
        stamps, naive = _utc_series(frame.get_column(column), column)
    except ForecastScoringError as exc:
        section = _empty_time()
        section["column"] = column
        section["error"] = str(exc)
        return section
    return _time_stats(stamps, column, naive)


def _empty_time() -> dict[str, object]:
    return {
        "column": None,
        "parsed": False,
        "start": None,
        "end": None,
        "strictly_increasing": False,
        "duplicate_count": None,
        "naive_as_utc": None,
    }


def _time_stats(stamps: pl.Series, column: str, naive: bool) -> dict[str, object]:
    import numpy as np

    ns = _epoch_ns(stamps)
    n = int(ns.size)
    duplicate_count = int(n - np.unique(ns).size) if n else 0
    strictly = bool(n >= 2 and bool(np.all(np.diff(ns) > 0)))
    return {
        "column": column,
        "parsed": True,
        "start": _iso_ns(int(ns.min())) if n else None,
        "end": _iso_ns(int(ns.max())) if n else None,
        "strictly_increasing": strictly,
        "duplicate_count": duplicate_count,
        "naive_as_utc": naive,
    }


def _iso_ns(ns: int) -> str:
    from datetime import UTC, datetime

    seconds, rem = divmod(int(ns), 1_000_000_000)
    try:
        stamp = datetime.fromtimestamp(seconds, UTC)
    except (OverflowError, OSError, ValueError):
        return str(ns)
    if rem:
        stamp = stamp.replace(microsecond=rem // 1000)
        return stamp.isoformat()
    return stamp.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _execute_selftest(task: str) -> tuple[bool, str | None, bool, str]:
    if task not in _TASKS:
        return (False, None, False, "unknown task; expected volatility, quantile, or probability")
    with tempfile.TemporaryDirectory(prefix="fx1_forecast_scoring_") as tmp:
        root = Path(tmp)
        data_path = root / "fixture.csv"
        data_path.write_text(_selftest_csv(task), encoding="utf-8")
        outcome = score_forecasts(
            data_path,
            task=task,
            out_dir=root / "out",
            available_time_col="available_time",
            seed=0,
        )
        verified = _file_valid(outcome.receipt_path)
        tamper_rejected = _tamper_rejected(outcome.receipt_path)
        passed = _selftest_passed(outcome, task, verified, tamper_rejected)
        detail = (
            "verified; tampered copy rejected"
            if passed
            else _selftest_failure_detail(verified, tamper_rejected, outcome)
        )
        digest = outcome.receipt_sha256 if verified else None
        return passed, digest, verified, detail


def _selftest_passed(
    outcome: ScoringOutcome, task: str, verified: bool, tamper_rejected: bool
) -> bool:
    if outcome.data_label != "SYNTHETIC" or outcome.task != task:
        return False
    if outcome.rows != _SELFTEST_ROWS:
        return False
    return bool(verified and tamper_rejected)


def _selftest_failure_detail(verified: bool, tamper_rejected: bool, outcome: ScoringOutcome) -> str:
    return (
        f"verified={verified}; tamper_rejected={tamper_rejected}; "
        f"label={outcome.data_label}; rows={outcome.rows}"
    )


def _file_valid(path: Path) -> bool:
    from quant_fund.schemas.receipt_v2_leaf import verify_receipt_file

    return verify_receipt_file(path)["valid"] is True


def _tamper_rejected(path: Path) -> bool:
    from quant_fund.schemas.receipt_v2_leaf import verify_receipt_file

    flipped = bytearray(path.read_bytes())
    flipped[len(flipped) // 2] ^= 0x1
    tampered = path.with_name(path.name + ".tampered")
    tampered.write_bytes(flipped)
    return verify_receipt_file(tampered)["valid"] is False


def _selftest_csv(task: str) -> str:
    if task == "volatility":
        return _volatility_fixture()
    if task == "probability":
        return _probability_fixture()
    return _quantile_fixture()


def _fixture_stamps(index: int) -> tuple[str, str]:
    from datetime import UTC, datetime, timedelta

    realized = datetime(2020, 1, 1, tzinfo=UTC) + timedelta(hours=index)
    issued = realized - timedelta(hours=1)
    stamp = "%Y-%m-%dT%H:%M:%S+00:00"
    return issued.strftime(stamp), realized.strftime(stamp)


def _volatility_fixture() -> str:
    lines = ["timestamp,available_time,realized,forecast_var,data_label"]
    for index in range(_SELFTEST_ROWS):
        issued, realized_at = _fixture_stamps(index)
        rv = 0.02 + (index % 10) * 0.001
        forecast = 0.021 + (index % 7) * 0.0005
        lines.append(f"{realized_at},{issued},{rv},{forecast},SYNTHETIC")
    return "\n".join(lines) + "\n"


def _probability_fixture() -> str:
    lines = ["timestamp,available_time,realized,forecast_prob,data_label"]
    for index in range(_SELFTEST_ROWS):
        issued, realized_at = _fixture_stamps(index)
        outcome = 0 if index % 3 else 1
        probability = 0.3 + (index % 4) * 0.15
        lines.append(f"{realized_at},{issued},{outcome},{probability},SYNTHETIC")
    return "\n".join(lines) + "\n"


def _quantile_fixture() -> str:
    lines = ["timestamp,available_time,realized,q05,q50,q95,data_label"]
    for index in range(_SELFTEST_ROWS):
        issued, realized_at = _fixture_stamps(index)
        value = (index % 20) / 20
        lines.append(f"{realized_at},{issued},{value},0.10,0.50,0.90,SYNTHETIC")
    return "\n".join(lines) + "\n"


def _kind_errors(envelope: Mapping[str, Any]) -> list[str]:
    if envelope.get("kind") == FORECAST_SCORING_KIND:
        return []
    return ["kind_not_forecast_scoring"]


def _payload_flag_errors(payload: Mapping[str, Any], envelope: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("kind") != FORECAST_SCORING_KIND:
        errors.append("payload_kind_mismatch")
    if payload.get("schema") != FORECAST_SCORING_KIND:
        errors.append("payload_schema_mismatch")
    if payload.get("research_only") is not True:
        errors.append("research_only_not_true")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    if payload.get("data_label") != envelope.get("data_label"):
        errors.append("data_label_mismatch")
    return errors


def _binding_errors(payload: Mapping[str, Any], envelope: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    dataset = payload.get("dataset")
    params = payload.get("params")
    if not isinstance(dataset, Mapping):
        errors.append("dataset_missing")
    elif _digest(dataset) != envelope.get("dataset_hash"):
        errors.append("dataset_hash_mismatch")
    if not isinstance(params, Mapping):
        errors.append("params_missing")
    elif _digest(params) != envelope.get("params_hash"):
        errors.append("params_hash_mismatch")
    return errors


def _digest(value: Mapping[str, Any]) -> str:
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

    return hash_bytes(canonical_json_bytes(dict(value)))


def _forbidden_clean(payload: Mapping[str, Any]) -> bool:
    from quant_fund.research.catalog import family_blob_forbidden_metrics_absent

    return bool(family_blob_forbidden_metrics_absent(dict(payload)))


__all__ = [
    "ECE_BINS",
    "FORECAST_SCORING_KIND",
    "LOG_LOSS_EPS",
    "MAX_BYTES",
    "MAX_ROWS",
    "MIN_ROWS",
    "N_BOOTSTRAP",
    "PROFILE_SCHEMA",
    "REPLAY_GENERATED_AT",
    "ForecastScoringError",
    "ScoringOutcome",
    "SelftestResult",
    "forecast_scoring_v2_consistency_errors",
    "inspect_dataset",
    "run_selftest",
    "score_forecasts",
]
