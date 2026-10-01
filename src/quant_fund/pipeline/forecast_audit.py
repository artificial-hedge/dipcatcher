"""Adversarial contract audit of the live forecast pipeline.

``pipeline.forecast`` (~3.4k lines across decide, covariance, history,
conformal, garch, realized, artifacts, wrappee, state and the facade
``__init__``) turns gold panels into ``MarketState`` artifacts, target
weights and conformal intervals. This module runs a fixed battery of named
probes against the four contract families that pipeline must keep:

* **determinism** — identical inputs reproduce identical outputs and cache
  digests (day-index slices match ``filter``, prefix history matches the
  ``<=`` path, conformal cache keys are content fingerprints, never
  ``id(frame)``);
* **fail-closed edges** — empty/degenerate/mismatched inputs raise or stamp
  an honest ``unmeasured_reason`` / ``homoskedastic_proxy``, never fabricate
  (missing asof raises instead of latest-date fallback; named covariance
  paths refuse short history rather than silently substituting Ledoit-Wolf;
  null ``available_time`` is a ``PointInTimeError``);
* **causality** — outputs at ``asof`` are invariant to appending strictly
  later rows, calibration windows purge the last ``horizon`` sessions, and
  unpublished restatements cannot move trailing covariance;
* **honesty** — SYNTHETIC labeling, zero-weight challengers never size the
  book, and paper challengers are stamped but never blended.

``flag_*`` probes pin discovered warts. They assert the wart *exists* —
the point is a documented ledger, not a fix:

* ``flag_conformal_cache_total_eviction`` — ``_CONFORMAL_CACHE`` clears the
  whole store past 256 entries while ``_WRAPPEE_CACHE`` uses LRU eviction;
* ``flag_unsorted_index_concat_reorders`` — ``history_upto`` with a
  day_index on an unsorted frame preserves the row multiset but reorders
  rows vs the filter path (its own docstring warns about this);
* ``flag_quantile_maps_symmetric_sigma_rule`` — ``AssetForecast.quantiles``
  are ``alpha ± 1.65*vol_20``, a symmetric Gaussian-style rule, not the
  fitted conformal set;
* ``flag_confidence_floor_unreachable`` — ``np.clip(0.5 + |pct-0.5|, 0.2,
  1.0)`` can never reach the documented 0.2 floor; effective minimum
  confidence is 0.5;
* ``flag_heuristic_zero_alpha_silent`` — with no ranker artifact and no
  ``cs_pct_mom_20`` column the pipeline emits an all-zero alpha instead of
  failing;
* ``flag_w_prev_dict_missing_ids_zeroed`` — a dict ``w_prev`` silently
  zeroes missing security ids (documented, but a fat-fingered dict shrinks
  the book without a warning);
* ``flag_core_engine_mislabels_heuristic`` — ``core_engine`` is stamped
  ``"ridge"`` whenever robinhood_plus is enabled and unsized, even when no
  ridge (or any ranker) produced the scores.

References: Engle (2002) DCC; Ledoit & Wolf (2004); Hansen, Huang & Shek
(2012) Realized GARCH; Romano, Patterson & Candes (2019) conformalized
quantile regression.

Honesty: every probe runs on in-process synthetic panels only; nothing here
measures live performance and ``forecast_pipeline_audit_bench`` emits a
``data_label='SYNTHETIC'`` receipt with ``research_only=True`` and
``live_pnl_claim=False``. Composition: sibling of ``pipeline.doctor``
(config sanity) and the ``research.*_eval`` receipt lanes; this lane audits
the forecast artifact path itself. Kind ``forecast_pipeline_audit`` is
checked against ``EVALUE_FAMILY_KINDS`` and the lane/sim kinds so it does
not collide with any stricter verifier path.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.data.point_in_time import filter_trailing_returns_asof
from quant_fund.models.base import save_joblib_artifact
from quant_fund.models.calibration import ProbabilityCalibrator
from quant_fund.models.covariance import require_implemented_optimizer_covariance
from quant_fund.models.realized_garch import RealizedGARCHVol
from quant_fund.models.volatility import GARCH_DATE_LEVEL_SCOPE, GARCHVol
from quant_fund.pipeline.forecast import (
    _CONFORMAL_CACHE,
    AssetForecast,
    MarketState,
    _align_w_prev,
    _apply_forecast_interval_caps,
    _date_train_cal,
    _garch_history_digest,
    _garch_overlay_return_frame,
    _load_probability_calibrator,
    _load_rl_cached,
    _paper_challenger_stamp,
    _require_wrappee_resolve_inputs,
    _unmeasured_optimizer_covariance,
    apply_market_variance_overlay_to_covariance,
    build_causal_weight_panel,
    build_event_time_day_index,
    clear_forecast_caches,
    conformal_sets_asof,
    estimate_optimizer_covariance_asof,
    forecast_asof,
    garch_market_forecast_asof,
    garch_name_forecasts_asof,
    history_for_calibration,
    history_prefix_upto,
    history_upto,
    optimize_asof,
    overlay_covariance_with_garch_market,
    realized_garch_market_forecast_asof,
    resolve_market_variance_overlay_asof,
    slice_day,
    sort_for_history,
    wrappee_cal_fingerprint,
)
from quant_fund.pipeline.forecast.history import ForecastIntervals
from quant_fund.schemas.errors import OptimizationInfeasible, PointInTimeError
from quant_fund.schemas.forecast import (
    MARKET_RISK_OVERLAY_GARCH,
    MARKET_RISK_OVERLAY_REALIZED_GARCH,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["forecast_pipeline_audit", "forecast_pipeline_audit_bench"]

_AUDIT_START = datetime(2024, 1, 2, tzinfo=UTC)


def _frame(
    *,
    n_days: int = 40,
    n_names: int = 6,
    seed: int = 7,
    mom: bool = True,
    label: bool = True,
    ohlc: bool = False,
) -> pl.DataFrame:
    """Deterministic synthetic panel: rows sorted by (event_time, security_id)."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for day in range(n_days):
        stamp = _AUDIT_START + timedelta(days=day)
        market = float(rng.normal(0.0, 0.008))
        for name in range(n_names):
            vol = 0.010 + 0.0015 * name
            ret = market + float(rng.normal(0.0, vol))
            row: dict[str, Any] = {
                "event_time": stamp,
                "security_id": f"S{name:02d}",
                "symbol": f"S{name:02d}",
                "ret_1": ret,
                "vol_20": vol,
            }
            if mom:
                row["cs_pct_mom_20"] = 0.30 + 0.08 * name + float(rng.normal(0.0, 0.02))
            if label:
                row["future_log_return_5"] = float(rng.normal(0.0, 0.02))
            if ohlc:
                row["high"] = 100.0 * (1.0 + abs(ret) + 0.01)
                row["low"] = 100.0 * (1.0 - abs(ret) - 0.01)
            rows.append(row)
    return pl.DataFrame(rows)


def _asof(day: int) -> datetime:
    return _AUDIT_START + timedelta(days=day)


def _config(root: Path | None = None, **overrides: Any) -> AppConfig:
    """Fresh default config on an isolated data root; dotted override keys use ``__``."""
    cfg = AppConfig()
    cfg.data.root = Path(root) if root is not None else Path(tempfile.mkdtemp(prefix="audit_lake_"))
    cfg.data.source = "synthetic"
    for dotted, value in overrides.items():
        section, _, field_name = dotted.partition("__")
        setattr(getattr(cfg, section), field_name, value)
    return cfg


def _raises(exc_type: type[BaseException], fn: Callable[[], Any]) -> bool:
    try:
        fn()
    except exc_type:
        return True
    except Exception:
        return False
    return False


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def _probe_day_index_slice_equals_filter() -> bool:
    frame = _frame()
    index = build_event_time_day_index(frame)
    via_index = slice_day(frame, _asof(20), day_index=index)
    via_filter = frame.filter(pl.col("event_time") == _asof(20))
    miss = slice_day(frame, _asof(-3), day_index=index)
    return via_index.equals(via_filter) and miss.is_empty() and miss.columns == frame.columns


def _probe_history_prefix_equals_filter() -> bool:
    frame = _frame()
    asof = _asof(25)
    return history_prefix_upto(frame, asof).equals(frame.filter(pl.col("event_time") <= asof))


def _probe_history_upto_sorted_index_preserves_order() -> bool:
    frame = _frame()
    asof = _asof(25)
    index = build_event_time_day_index(frame)
    return history_upto(frame, asof, day_index=index).equals(
        frame.filter(pl.col("event_time") <= asof)
    )


def _probe_forecast_asof_repeat_call_identical() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp), fusion__skip_intervals=True)
        frame = _frame()
        first = forecast_asof(cfg, _asof(30), frame=frame)
        second = forecast_asof(cfg, _asof(30), frame=frame)
        return first is not second and first.model_dump(mode="json") == second.model_dump(
            mode="json"
        )


def _probe_optimize_asof_repeat_call_identical() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp), fusion__skip_intervals=True)
        frame = _frame()
        first = optimize_asof(cfg, _asof(30), persist=False, frame=frame)
        second = optimize_asof(cfg, _asof(30), persist=False, frame=frame)
        return first.equals(second) and first.height == 6


def _probe_conformal_sets_repeat_call_identical() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp))
        frame = _frame()
        try:
            first = conformal_sets_asof(frame, _asof(35), cfg)
            second = conformal_sets_asof(frame, _asof(35), cfg)
        finally:
            clear_forecast_caches()
        if first is None or second is None:
            return False
        return bool(
            first.lower == second.lower
            and first.upper == second.upper
            and first.cal_event_times == second.cal_event_times
            and first.method == second.method
        )


def _probe_conformal_cache_content_keyed() -> bool:
    """A different DataFrame object with identical bytes must hit the same entry."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp))
        frame = _frame()
        try:
            first = conformal_sets_asof(frame, _asof(35), cfg)
            second = conformal_sets_asof(frame.clone(), _asof(35), cfg)
        finally:
            clear_forecast_caches()
        return first is not None and first is second


def _probe_garch_history_digest_full_path() -> bool:
    dates = np.array([_AUDIT_START + timedelta(days=d) for d in range(30)], dtype=object)
    values = np.linspace(0.01, 0.02, 30)
    digest = _garch_history_digest(dates, values)
    mid_edit = values.copy()
    mid_edit[15] += 1e-9
    tail_edit = values.copy()
    tail_edit[-1] += 1e-9
    return (
        digest == _garch_history_digest(dates.copy(), values.copy())
        and digest != _garch_history_digest(dates, mid_edit)
        and digest != _garch_history_digest(dates, tail_edit)
    )


def _probe_wrappee_fingerprint_train_primary() -> bool:
    base = wrappee_cal_fingerprint(
        train_date_keys=("a", "b", "c"),
        cal_date_keys=("x",),
        alpha=0.1,
        n_tr=30,
        n_cal=10,
        y_tr_mean=0.0,
        scale_tr_mean=0.01,
        label="future_log_return_5",
        train_content_digest="deadbeef",
    )
    slid_cal = wrappee_cal_fingerprint(
        train_date_keys=("a", "b", "c"),
        cal_date_keys=("x", "y", "z"),
        alpha=0.1,
        n_tr=30,
        n_cal=12,
        y_tr_mean=0.0,
        scale_tr_mean=0.01,
        label="future_log_return_5",
        train_content_digest="deadbeef",
    )
    moved_train = wrappee_cal_fingerprint(
        train_date_keys=("a", "b", "d"),
        alpha=0.1,
        n_tr=30,
        y_tr_mean=0.0,
        scale_tr_mean=0.01,
        label="future_log_return_5",
        train_content_digest="deadbeef",
    )
    return base == slid_cal and base != moved_train


# ---------------------------------------------------------------------------
# fail-closed edges
# ---------------------------------------------------------------------------


def _probe_forecast_fail_closed_inputs() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp), fusion__skip_intervals=True)
        frame = _frame()
        missing_day = _raises(ValueError, lambda: forecast_asof(cfg, _asof(-5), frame=frame))
        empty_frame = _raises(TypeError, lambda: forecast_asof(cfg, frame=frame.head(0)))
        return missing_day and empty_frame


def _probe_w_prev_alignment_fail_closed() -> bool:
    align = _raises(ValueError, lambda: _align_w_prev(["S00", "S01"], np.array([0.5])))
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp), fusion__skip_intervals=True)
        optimize = _raises(
            ValueError,
            lambda: optimize_asof(
                cfg, _asof(30), persist=False, frame=_frame(), w_prev=np.zeros(2)
            ),
        )
    return align and optimize


def _probe_history_sort_contract_fail_closed() -> bool:
    frame = _frame()
    unsorted = frame.sample(fraction=1.0, seed=11, shuffle=True)
    unsorted_index = build_event_time_day_index(unsorted)
    lie = _raises(
        ValueError,
        lambda: history_upto(unsorted, _asof(20), day_index=unsorted_index, assume_sorted=True),
    )
    missing_keys = _raises(ValueError, lambda: sort_for_history(frame.drop("security_id")))
    bad_asof = _raises(
        TypeError,
        lambda: history_prefix_upto(frame, cast(datetime, "not-a-date")),
    )
    return lie and missing_keys and bad_asof


def _probe_calibration_history_bad_inputs_fail() -> bool:
    frame = _frame()
    asof = _asof(30)
    bad_horizon = _raises(ValueError, lambda: history_for_calibration(frame, asof, -1)) and _raises(
        ValueError,
        lambda: history_for_calibration(frame, asof, cast(int, 2.5)),
    )
    unsorted_times = _raises(
        ValueError,
        lambda: history_for_calibration(frame, asof, 5, event_times=[asof, _AUDIT_START]),
    )
    non_datetime_times = _raises(
        ValueError,
        lambda: history_for_calibration(
            frame, asof, 5, event_times=[asof, cast(datetime, "2024-01-01")]
        ),
    )
    return bad_horizon and unsorted_times and non_datetime_times


def _probe_garch_overlay_rejects_bad_sigma() -> bool:
    eye = np.eye(3)
    return (
        _raises(
            ValueError,
            lambda: overlay_covariance_with_garch_market(np.ones((2, 3)), 0.01),
        )
        and _raises(
            ValueError,
            lambda: overlay_covariance_with_garch_market(
                np.array([[np.nan, 0.0], [0.0, 1.0]]), 0.01
            ),
        )
        and _raises(ValueError, lambda: overlay_covariance_with_garch_market(eye, 0.0))
        and _raises(ValueError, lambda: overlay_covariance_with_garch_market(eye, float("nan")))
    )


def _probe_covariance_spec_fail_closed() -> bool:
    """Named DCC refuses short history; unimplemented specs refuse outright."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp))
        cfg.optimizer.covariance = "dcc_gaussian"
        frame = _frame(n_days=30)
        asof = _asof(29)
        ids = sorted(frame["security_id"].unique().to_list())
        hist = history_prefix_upto(frame, asof)
        short_dcc = _raises(
            ValueError,
            lambda: estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist),
        )
        cfg.optimizer.covariance = "ewma"
        short_frame = _frame(n_days=15)
        short_ewma = _raises(
            ValueError,
            lambda: estimate_optimizer_covariance_asof(
                cfg,
                short_frame,
                _asof(14),
                ids,
                history_prefix_upto(short_frame, _asof(14)),
            ),
        )
        unknown = _raises(
            ValueError,
            lambda: require_implemented_optimizer_covariance("covariance_crystal_ball"),
        )
    return short_dcc and short_ewma and unknown


def _probe_wrappee_resolve_input_validation() -> bool:
    y = np.linspace(-0.02, 0.02, 24)
    scale = np.full(24, 0.01)
    taus = [0.05, 0.95]
    return (
        _raises(
            ValueError,
            lambda: _require_wrappee_resolve_inputs(
                [], y, scale, y, scale, alpha=0.1, min_coverage=0.85
            ),
        )
        and _raises(
            ValueError,
            lambda: _require_wrappee_resolve_inputs(
                taus, y, scale[:-1], y, scale, alpha=0.1, min_coverage=0.85
            ),
        )
        and _raises(
            ValueError,
            lambda: _require_wrappee_resolve_inputs(
                taus, y, scale, y[:4], scale, alpha=0.1, min_coverage=0.85
            ),
        )
        and _raises(
            ValueError,
            lambda: _require_wrappee_resolve_inputs(
                taus, y, scale, y, scale, alpha=1.5, min_coverage=0.85
            ),
        )
        and _raises(
            ValueError,
            lambda: _require_wrappee_resolve_inputs(
                taus, y, scale, y, scale, alpha=0.1, min_coverage=0.0
            ),
        )
    )


def _probe_rl_malformed_artifact_fails() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        metadata = Path(tmp) / "metadata"
        metadata.mkdir(parents=True)
        save_joblib_artifact({"policy": object()}, metadata / "rl_thompson.joblib")
        cfg = _config(Path(tmp))
        try:
            return _raises(ValueError, lambda: _load_rl_cached(cfg))
        finally:
            clear_forecast_caches()


def _probe_calibrator_fail_closed() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp), fusion__apply_probability_calibration=True)
        missing = _raises(ValueError, lambda: _load_probability_calibrator(cfg))
        metadata = Path(tmp) / "metadata"
        metadata.mkdir(parents=True, exist_ok=True)
        grid = np.linspace(0.0, 1.0, 30)
        calibrator = ProbabilityCalibrator().fit(grid, (grid > 0.5).astype(float))
        calibrator.score_feature = "not_cs_pct_mom_20"
        calibrator.fit_start = calibrator.fit_end = "2024-01-01"
        calibrator.oos_start = calibrator.oos_end = "2024-02-01"
        calibrator.save(metadata / "calibrator_auto.joblib")
        wrong_identity = _raises(ValueError, lambda: _load_probability_calibrator(cfg))
        return missing and wrong_identity


def _probe_garch_spec_contract_fails() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        metadata = Path(tmp) / "metadata"
        metadata.mkdir(parents=True)
        GARCHVol(series_scope="other_universe").save(metadata / "vol_garch.joblib")
        cfg = _config(Path(tmp))
        frame = _frame()
        try:
            wrong_scope = _raises(
                ValueError, lambda: garch_market_forecast_asof(cfg, frame, _asof(30))
            )
            GARCHVol(min_obs=10, series_scope=GARCH_DATE_LEVEL_SCOPE).save(
                metadata / "vol_garch.joblib"
            )
            no_history = _raises(
                ValueError,
                lambda: garch_name_forecasts_asof(cfg, frame, _asof(30), ["NOPE"]),
            )
            dup_ids = _raises(
                ValueError,
                lambda: garch_name_forecasts_asof(cfg, frame, _asof(30), ["S00", "S00"]),
            )
            return wrong_scope and no_history and dup_ids
        finally:
            clear_forecast_caches()


def _probe_trailing_returns_null_availability_fails() -> bool:
    frame = _frame()
    asof = _asof(30)
    stamped = frame.with_columns(
        pl.when(pl.col("event_time") == _AUDIT_START)
        .then(None)
        .otherwise(pl.lit(asof))
        .alias("available_time")
    )
    direct = _raises(PointInTimeError, lambda: filter_trailing_returns_asof(stamped, asof))
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp), fusion__skip_intervals=True)
        via_optimize = _raises(
            PointInTimeError,
            lambda: optimize_asof(cfg, asof, persist=False, frame=stamped),
        )
    return direct and via_optimize


def _probe_universe_empty_membership_fails() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        silver = Path(tmp) / "silver"
        silver.mkdir(parents=True)
        pl.DataFrame(schema={"security_id": pl.Utf8, "asof": pl.Datetime}).write_parquet(
            silver / "universe.parquet"
        )
        cfg = _config(Path(tmp))
        return _raises(PointInTimeError, lambda: _garch_overlay_return_frame(cfg, _frame()))


def _probe_rgarch_missing_ohlc_fails() -> bool:
    """A present RGARCH artifact without OHLC raises — no silent GARCH fallback."""
    with tempfile.TemporaryDirectory() as tmp:
        metadata = Path(tmp) / "metadata"
        metadata.mkdir(parents=True)
        RealizedGARCHVol(min_obs=10, series_scope=GARCH_DATE_LEVEL_SCOPE).save(
            metadata / "vol_realized_garch.joblib"
        )
        cfg = _config(Path(tmp))
        frame = _frame(ohlc=False)
        try:
            direct = _raises(
                PointInTimeError,
                lambda: realized_garch_market_forecast_asof(cfg, frame, _asof(30)),
            )
            resolved = _raises(
                PointInTimeError,
                lambda: resolve_market_variance_overlay_asof(cfg, frame, _asof(30)),
            )
            return direct and resolved
        finally:
            clear_forecast_caches()


def _probe_interval_caps_turnover_infeasible_fails() -> bool:
    """Interval caps + hard turnover: an impossible minimum move fails closed."""
    asof = _asof(30)
    state = MarketState(
        asof=asof,
        forecasts=[
            AssetForecast(
                security_id="S00",
                symbol="S00",
                asof=asof,
                model_version="audit",
                interval_lo={"5d": -0.9},
                interval_hi={"5d": 0.9},
                interval_alpha=0.1,
                interval_method="split_cqr",
            )
        ],
    )
    cfg = _config()
    return _raises(
        OptimizationInfeasible,
        lambda: _apply_forecast_interval_caps(
            np.zeros(1), state, ["S00"], cfg, w_prev=np.array([0.9])
        ),
    )


def _probe_causal_weight_panel_empty_dates() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp))
        out = build_causal_weight_panel(cfg, dates=[])
        return out.is_empty() and set(out.columns) == {
            "event_time",
            "security_id",
            "target_weight",
            "alpha",
        }


# ---------------------------------------------------------------------------
# causality
# ---------------------------------------------------------------------------


def _probe_forecast_prefix_invariant() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp), fusion__skip_intervals=True)
        frame = _frame()
        asof = _asof(30)
        full = forecast_asof(cfg, asof, frame=frame)
        truncated = forecast_asof(cfg, asof, frame=frame.filter(pl.col("event_time") <= asof))
        return full.model_dump(mode="json") == truncated.model_dump(mode="json")


def _probe_optimize_prefix_invariant() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp), fusion__skip_intervals=True)
        frame = _frame()
        asof = _asof(30)
        full = optimize_asof(cfg, asof, persist=False, frame=frame)
        truncated = optimize_asof(
            cfg, asof, persist=False, frame=frame.filter(pl.col("event_time") <= asof)
        )
        return full.equals(truncated)


def _probe_conformal_prefix_invariant() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp))
        frame = _frame()
        asof = _asof(35)
        try:
            full = conformal_sets_asof(frame, asof, cfg)
            truncated = conformal_sets_asof(frame.filter(pl.col("event_time") <= asof), asof, cfg)
        finally:
            clear_forecast_caches()
        if full is None or truncated is None:
            return False
        return bool(
            full.lower == truncated.lower
            and full.upper == truncated.upper
            and full.cal_event_times == truncated.cal_event_times
        )


def _probe_conformal_cal_boundary_respected() -> bool:
    """Calibration labels are realized at asof: last `horizon` sessions excluded."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp))
        frame = _frame()
        asof = _asof(35)
        horizon = 5  # future_log_return_5
        try:
            result = conformal_sets_asof(frame, asof, cfg)
        finally:
            clear_forecast_caches()
        if result is None:
            return False
        times = sorted(set(frame["event_time"].to_list()))
        idx = next(i for i, t in enumerate(times) if t >= asof)
        cutoff_key = str(times[idx - horizon - 1])[:10]
        asof_key = str(asof)[:10]
        cal_keys = {str(stamp)[:10] for stamp in result.cal_event_times}
        return bool(
            cal_keys
            and all(key <= cutoff_key for key in cal_keys)
            and asof_key not in cal_keys
            and cutoff_key < asof_key
        )


def _probe_calibration_history_excludes_unrealized() -> bool:
    frame = _frame()
    asof = _asof(30)
    horizon = 5
    hist = history_for_calibration(frame, asof, horizon)
    times = sorted(set(frame["event_time"].to_list()))
    idx = next(i for i, t in enumerate(times) if t >= asof)
    cutoff = times[idx - horizon - 1]
    seen = set(hist["event_time"].to_list())
    excluded = set(times[idx - horizon : idx + 1])
    return bool(seen) and seen.isdisjoint(excluded) and max(seen) == cutoff


def _probe_date_train_cal_purges_boundary() -> bool:
    uniq = [_AUDIT_START + timedelta(days=d) for d in range(30)]
    dates = np.array([day for day in uniq for _ in range(6)], dtype=object)
    horizon = 5
    tr, cal = _date_train_cal(dates, horizon=horizon)
    cut = min(max(int(0.7 * len(uniq)), 3), len(uniq) - 2)
    train_days = {uniq[i // 6] for i, keep in enumerate(tr) if keep}
    cal_days = {uniq[i // 6] for i, keep in enumerate(cal) if keep}
    if not train_days or not cal_days or not train_days.isdisjoint(cal_days):
        return False
    gap = uniq.index(min(cal_days)) - uniq.index(max(train_days)) - 1
    return bool(
        gap == horizon
        and max(train_days) < uniq[cut] <= min(cal_days)
        and max(cal_days) == uniq[-1]
    )


def _probe_garch_overlay_strictly_prior() -> bool:
    """Rows at or after asof — however extreme — cannot move the overlay."""
    with tempfile.TemporaryDirectory() as tmp:
        metadata = Path(tmp) / "metadata"
        metadata.mkdir(parents=True)
        GARCHVol(min_obs=10, series_scope=GARCH_DATE_LEVEL_SCOPE).save(
            metadata / "vol_garch.joblib"
        )
        cfg = _config(Path(tmp))
        frame = _frame()
        asof = _asof(35)
        try:
            base = garch_market_forecast_asof(cfg, frame, asof)
            extreme = pl.DataFrame(
                {
                    "event_time": [asof, asof + timedelta(days=1)],
                    "security_id": ["X00", "X00"],
                    "symbol": ["X00", "X00"],
                    "ret_1": [9.9, -9.9],
                    "vol_20": [1.0, 1.0],
                }
            )
            polluted = garch_market_forecast_asof(
                cfg, pl.concat([frame, extreme], how="diagonal_relaxed"), asof
            )
        finally:
            clear_forecast_caches()
        if base is None or polluted is None:
            return False
        return bool(
            base.variance == polluted.variance
            and base.cumulative_variance == polluted.cumulative_variance
            and base.series_scope == GARCH_DATE_LEVEL_SCOPE
            and base.n_obs < frame["event_time"].n_unique()
        )


def _probe_trailing_returns_drops_unpublished() -> bool:
    """A restatement published after asof cannot enter trailing covariance inputs."""
    frame = _frame()
    asof = _asof(30)
    hist = history_prefix_upto(frame, asof)
    published_late = hist.with_columns(
        pl.when(pl.col("event_time") == _AUDIT_START)
        .then(pl.lit(asof + timedelta(days=9)))
        .otherwise(pl.lit(asof))
        .alias("available_time")
    )
    filtered = filter_trailing_returns_asof(published_late, asof)
    day_zero_dropped = filtered.filter(pl.col("event_time") == _AUDIT_START).is_empty()
    rest_kept = filtered.height == hist.height - 6
    return day_zero_dropped and rest_kept


# ---------------------------------------------------------------------------
# honesty
# ---------------------------------------------------------------------------


def _probe_synthetic_note_stamped() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp), fusion__skip_intervals=True)
        state = forecast_asof(cfg, _asof(30), frame=_frame())
        return "SYNTHETIC" in state.notes


def _probe_synthetic_note_absent_for_file_source() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp), fusion__skip_intervals=True)
        cfg.data.source = "file"
        state = forecast_asof(cfg, _asof(30), frame=_frame())
        return "SYNTHETIC" not in state.notes


def _probe_challenger_zero_blend_not_sized() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp), fusion__skip_intervals=True)
        state = forecast_asof(cfg, _asof(30), frame=_frame())
        blend_notes = [n for n in state.notes if n == "robinhood_plus_blend_weight=0.0"]
        engine_notes = [n for n in state.notes if n == "robinhood_plus"]
        statuses = {str(f.diagnostics.get("robinhood_plus_status")) for f in state.forecasts}
        return bool(
            "robinhood_plus_challenger" in state.notes
            and blend_notes
            and not engine_notes
            and statuses == {"challenger"}
        )


def _probe_paper_challengers_stamp_never_sizes() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp))
        note, diag = _paper_challenger_stamp(
            cfg,
            np.zeros((4, 1)),
            np.zeros(4),
        )
        return (
            bool(note.startswith("paper_challengers="))
            and float(diag["paper_challengers_blend"]) == 0.0
        )


def _probe_unmeasured_covariance_stamps_proxy() -> bool:
    estimate = _unmeasured_optimizer_covariance("audit_reason", ["S00", "S01"])
    return bool(
        estimate.estimator == "homoskedastic_proxy"
        and estimate.covariance_object == "diagonal_proxy"
        and estimate.spec == "diagonal_2pct"
        and estimate.unmeasured_reason == "audit_reason"
        and estimate.fallback_reason == "audit_reason"
        and estimate.sigma.shape == (2, 2)
    )


def _probe_named_trailing_specs_stamp_identity() -> bool:
    """sample/oas/ewma stamp their own estimator, never a silent substitute."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp))
        frame = _frame(n_days=60)
        asof = _asof(59)
        ids = sorted(frame["security_id"].unique().to_list())
        hist = history_prefix_upto(frame, asof)
        stamps: dict[str, str] = {}
        try:
            for name in ("sample", "oas", "ewma"):
                cfg.optimizer.covariance = name
                estimate = estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)
                stamps[name] = str(estimate.estimator)
        finally:
            clear_forecast_caches()
        return stamps == {"sample": "sample", "oas": "oas", "ewma": "ewma"}


def _probe_overlay_precedence_rgarch_then_garch() -> bool:
    """A present RGARCH artifact is preferred; removing it falls back to GARCH."""
    with tempfile.TemporaryDirectory() as tmp:
        metadata = Path(tmp) / "metadata"
        metadata.mkdir(parents=True)
        GARCHVol(min_obs=10, series_scope=GARCH_DATE_LEVEL_SCOPE).save(
            metadata / "vol_garch.joblib"
        )
        RealizedGARCHVol(min_obs=10, series_scope=GARCH_DATE_LEVEL_SCOPE).save(
            metadata / "vol_realized_garch.joblib"
        )
        cfg = _config(Path(tmp))
        frame = _frame(ohlc=True)
        try:
            overlay, kind = resolve_market_variance_overlay_asof(cfg, frame, _asof(35))
            preferred = overlay is not None and kind == MARKET_RISK_OVERLAY_REALIZED_GARCH
            (metadata / "vol_realized_garch.joblib").unlink()
            clear_forecast_caches()
            overlay2, kind2 = resolve_market_variance_overlay_asof(cfg, frame, _asof(35))
            return bool(preferred and overlay2 is not None and kind2 == MARKET_RISK_OVERLAY_GARCH)
        finally:
            clear_forecast_caches()


def _probe_market_overlay_absent_returns_none() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp))
        overlay, kind = resolve_market_variance_overlay_asof(cfg, _frame(), _asof(30))
        sigma, stamped_overlay, stamped_kind = apply_market_variance_overlay_to_covariance(
            cfg, _frame(), _asof(30), np.eye(2)
        )
        return (
            overlay is None
            and kind is None
            and stamped_overlay is None
            and stamped_kind is None
            and np.array_equal(sigma, np.eye(2))
        )


# ---------------------------------------------------------------------------
# warts (flag_*) — pinned, deliberately not fixed
# ---------------------------------------------------------------------------


def _probe_flag_conformal_cache_total_eviction() -> bool:
    """_CONFORMAL_CACHE clears the entire store past 256 entries (no LRU)."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp))
        try:
            _CONFORMAL_CACHE.update(
                {("audit_sentinel", i): cast(ForecastIntervals, None) for i in range(300)}
            )
            result = conformal_sets_asof(_frame(), _asof(35), cfg)
            return result is not None and len(_CONFORMAL_CACHE) <= 4
        finally:
            clear_forecast_caches()


def _probe_flag_unsorted_index_concat_reorders() -> bool:
    """day-index concat on an unsorted frame keeps rows but changes order."""
    frame = _frame().sample(fraction=1.0, seed=23, shuffle=True)
    asof = _asof(20)
    index = build_event_time_day_index(frame)
    via_index = history_upto(frame, asof, day_index=index)
    via_filter = frame.filter(pl.col("event_time") <= asof)
    same_rows = via_index.sort(["event_time", "security_id"]).equals(
        via_filter.sort(["event_time", "security_id"])
    )
    reordered = not via_index.equals(via_filter)
    return same_rows and reordered


def _probe_flag_quantile_maps_symmetric_sigma_rule() -> bool:
    """AssetForecast.quantiles are alpha ± 1.65*vol, not the fitted conformal set."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp), fusion__skip_intervals=True)
        state = forecast_asof(cfg, _asof(30), frame=_frame())
        for forecast in state.forecasts:
            quantiles = forecast.quantiles.get("5d", {})
            vol = forecast.volatility.get("5d", float("nan"))
            alpha = forecast.alpha.get("5d", float("nan"))
            if not quantiles or not np.isfinite(vol) or not np.isfinite(alpha):
                return False
            q05, q50, q95 = quantiles[0.05], quantiles[0.5], quantiles[0.95]
            if abs(q50 - alpha) > 1e-12 or abs((q95 - q50) - 1.65 * vol) > 1e-12:
                return False
            if abs((q95 - q50) - (q50 - q05)) > 1e-12:
                return False
        return True


def _probe_flag_confidence_floor_unreachable() -> bool:
    """clip(0.5 + |pct-0.5|, 0.2, 1.0): the documented 0.2 floor never binds."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp), fusion__skip_intervals=True)
        state = forecast_asof(cfg, _asof(30), frame=_frame())
        confidences = [f.confidence["5d"] for f in state.forecasts]
        return bool(confidences) and min(confidences) >= 0.5


def _probe_flag_heuristic_zero_alpha_silent() -> bool:
    """No ranker + no cs_pct_mom_20 -> all-zero alpha instead of an error."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp), fusion__skip_intervals=True)
        state = forecast_asof(cfg, _asof(30), frame=_frame(mom=False))
        alphas = [f.alpha["5d"] for f in state.forecasts]
        scores = [f.rank_score["5d"] for f in state.forecasts]
        silent = not any(
            marker in note
            for note in state.notes
            for marker in ("heuristic", "zero_alpha", "no_ranker")
        )
        return all(a == 0.0 for a in alphas) and all(s == 0.0 for s in scores) and silent


def _probe_flag_w_prev_dict_missing_ids_zeroed() -> bool:
    aligned = _align_w_prev(["S00", "S01", "S02"], {"S00": 0.7})
    return bool(aligned[0] == 0.7 and aligned[1] == 0.0 and aligned[2] == 0.0)


def _probe_flag_core_engine_mislabels_heuristic() -> bool:
    """core_engine='ridge' stamped even when the momentum heuristic scored."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp), fusion__skip_intervals=True)
        frame = _frame()
        state = forecast_asof(cfg, _asof(30), frame=frame)
        no_ranker = not (Path(tmp) / "metadata").exists() or all(
            not name.startswith("ranker_")
            for name in (p.name for p in (Path(tmp) / "metadata").iterdir())
        )
        engines = {str(f.diagnostics.get("core_engine")) for f in state.forecasts}
        return no_ranker and engines == {"ridge"}


_PROBES: dict[str, Callable[[], bool]] = {
    # determinism
    "day_index_slice_equals_filter": _probe_day_index_slice_equals_filter,
    "history_prefix_equals_filter": _probe_history_prefix_equals_filter,
    "history_upto_sorted_index_preserves_order": (_probe_history_upto_sorted_index_preserves_order),
    "forecast_asof_repeat_call_identical": _probe_forecast_asof_repeat_call_identical,
    "optimize_asof_repeat_call_identical": _probe_optimize_asof_repeat_call_identical,
    "conformal_sets_repeat_call_identical": _probe_conformal_sets_repeat_call_identical,
    "conformal_cache_content_keyed": _probe_conformal_cache_content_keyed,
    "garch_history_digest_full_path": _probe_garch_history_digest_full_path,
    "wrappee_fingerprint_train_primary": _probe_wrappee_fingerprint_train_primary,
    # fail-closed edges
    "forecast_fail_closed_inputs": _probe_forecast_fail_closed_inputs,
    "w_prev_alignment_fail_closed": _probe_w_prev_alignment_fail_closed,
    "history_sort_contract_fail_closed": _probe_history_sort_contract_fail_closed,
    "calibration_history_bad_inputs_fail": _probe_calibration_history_bad_inputs_fail,
    "garch_overlay_rejects_bad_sigma": _probe_garch_overlay_rejects_bad_sigma,
    "covariance_spec_fail_closed": _probe_covariance_spec_fail_closed,
    "wrappee_resolve_input_validation": _probe_wrappee_resolve_input_validation,
    "rl_malformed_artifact_fails": _probe_rl_malformed_artifact_fails,
    "calibrator_fail_closed": _probe_calibrator_fail_closed,
    "garch_spec_contract_fails": _probe_garch_spec_contract_fails,
    "trailing_returns_null_availability_fails": (_probe_trailing_returns_null_availability_fails),
    "universe_empty_membership_fails": _probe_universe_empty_membership_fails,
    "rgarch_missing_ohlc_fails": _probe_rgarch_missing_ohlc_fails,
    "interval_caps_turnover_infeasible_fails": (_probe_interval_caps_turnover_infeasible_fails),
    "causal_weight_panel_empty_dates": _probe_causal_weight_panel_empty_dates,
    # causality
    "forecast_prefix_invariant": _probe_forecast_prefix_invariant,
    "optimize_prefix_invariant": _probe_optimize_prefix_invariant,
    "conformal_prefix_invariant": _probe_conformal_prefix_invariant,
    "conformal_cal_boundary_respected": _probe_conformal_cal_boundary_respected,
    "calibration_history_excludes_unrealized": (_probe_calibration_history_excludes_unrealized),
    "date_train_cal_purges_boundary": _probe_date_train_cal_purges_boundary,
    "garch_overlay_strictly_prior": _probe_garch_overlay_strictly_prior,
    "trailing_returns_drops_unpublished": _probe_trailing_returns_drops_unpublished,
    # honesty
    "synthetic_note_stamped": _probe_synthetic_note_stamped,
    "synthetic_note_absent_for_file_source": _probe_synthetic_note_absent_for_file_source,
    "challenger_zero_blend_not_sized": _probe_challenger_zero_blend_not_sized,
    "paper_challengers_stamp_never_sizes": _probe_paper_challengers_stamp_never_sizes,
    "unmeasured_covariance_stamps_proxy": _probe_unmeasured_covariance_stamps_proxy,
    "named_trailing_specs_stamp_identity": _probe_named_trailing_specs_stamp_identity,
    "overlay_precedence_rgarch_then_garch": (_probe_overlay_precedence_rgarch_then_garch),
    "market_overlay_absent_returns_none": _probe_market_overlay_absent_returns_none,
    # warts, pinned not fixed
    "flag_conformal_cache_total_eviction": _probe_flag_conformal_cache_total_eviction,
    "flag_unsorted_index_concat_reorders": _probe_flag_unsorted_index_concat_reorders,
    "flag_quantile_maps_symmetric_sigma_rule": (_probe_flag_quantile_maps_symmetric_sigma_rule),
    "flag_confidence_floor_unreachable": _probe_flag_confidence_floor_unreachable,
    "flag_heuristic_zero_alpha_silent": _probe_flag_heuristic_zero_alpha_silent,
    "flag_w_prev_dict_missing_ids_zeroed": _probe_flag_w_prev_dict_missing_ids_zeroed,
    "flag_core_engine_mislabels_heuristic": _probe_flag_core_engine_mislabels_heuristic,
}


def forecast_pipeline_audit() -> dict[str, bool]:
    """Run every named probe; True = contract holds (or wart pinned).

    A probe that raises is recorded False rather than propagated: the audit
    is itself a contract surface and must degrade to a ledger of failures.
    """
    results: dict[str, bool] = {}
    for name, probe in _PROBES.items():
        try:
            results[name] = bool(probe())
        except Exception:
            results[name] = False
    return results


def forecast_pipeline_audit_bench() -> dict[str, Any]:
    """Run the audit and seal the outcome as a verifiable receipt.

    ``receipt_sha256`` hashes the canonical payload *before* the field is
    added, so ``verify_receipt_payload`` can recompute the seal over the
    body minus the digest.
    """
    results = forecast_pipeline_audit()
    n_passed = sum(1 for held in results.values() if held)
    ok = all(results.values())
    payload: dict[str, Any] = {
        "kind": "forecast_pipeline_audit",
        "schema": "forecast_pipeline_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "results": results,
            "ok": ok,
            "n_probes": len(results),
            "n_passed": n_passed,
        },
        "interpretation": (
            "All probes hold: the forecast pipeline is deterministic, fails "
            "closed on degenerate inputs, never reads past asof, and stamps "
            "SYNTHETIC data honestly. Flagged warts are documented behavior, "
            "deliberately not fixed."
            if ok
            else "Some forecast-pipeline contract probes failed; inspect claim.results."
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
