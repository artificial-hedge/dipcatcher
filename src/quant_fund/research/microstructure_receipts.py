"""Internal consistency contracts for the pre-envelope microstructure lanes.

These checks rederive relationships from the embedded, often rounded summaries.
They do not authenticate tape provenance, reproduce a fit, replay a simulation,
establish point-in-time availability, or turn MIXED evidence into market proof.
In particular, a valid seal alone is insufficient for the registered schemas.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, TypeGuard

import numpy as np

Body = Mapping[str, Any]
Checker = Callable[[Body], list[str]]
_ARMS = ("iid", "regime", "split")
_SYNTHETIC = frozenset({"exec_cost_split", "glft_bench", "split_flow"})
_FRACTIONS = frozenset(
    {
        "mo_fraction",
        "fill_fraction_mean",
        "misclass_rate",
        "revision_rate",
        "p_continue",
        "p_move",
        "p_ge_5",
        "p_ge_10",
        "vpin_mean",
        "vpin_p90",
        "recovery_share",
    }
)


def _obj(value: object, name: str, errors: list[str]) -> Body:
    if not isinstance(value, Mapping) or not value:
        errors.append(f"{name}:missing_object")
        return {}
    return value


def _list(value: object, name: str, errors: list[str]) -> list[Any]:
    if not isinstance(value, list) or not value:
        errors.append(f"{name}:missing_list")
        return []
    return value


def _finite(value: object) -> TypeGuard[float | int]:
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _num(row: Body, key: str, errors: list[str], *, low: float = -math.inf) -> float:
    value = row.get(key)
    if not _finite(value) or float(value) < low:
        errors.append(f"{key}:number_domain")
        return 0.0
    return float(value)


def _count(row: Body, key: str, errors: list[str], *, positive: bool = False) -> int:
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < int(positive):
        errors.append(f"{key}:count_domain")
        return 0
    return value


def _eq(actual: object, expected: object, key: str, errors: list[str]) -> None:
    # Python's True == 1 must not validate a forged boolean/count.
    if type(actual) is not type(expected) or actual != expected:
        errors.append(f"{key}:rederive_mismatch")


def _near(
    actual: object, expected: float | None, key: str, errors: list[str], tolerance: float = 1e-8
) -> None:
    if expected is None:
        if actual is not None:
            errors.append(f"{key}:rederive_mismatch")
    elif not _finite(actual) or not math.isclose(
        float(actual), expected, rel_tol=1e-9, abs_tol=tolerance
    ):
        errors.append(f"{key}:rederive_mismatch")


def _tree_domains(value: object, errors: list[str], path: str = "", depth: int = 0) -> None:
    """Bound traversal and reject malformed numeric domains without a JSON seal."""
    if depth > 32:
        errors.append(f"{path}:nesting_limit")
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool) and not _finite(value):
        errors.append(f"{path}:not_finite")
        return
    if isinstance(value, Mapping):
        if len(value) > 4096:
            errors.append(f"{path}:object_limit")
            return
        for key, item in value.items():
            name = str(key)
            loc = f"{path}.{name}" if path else name
            if (
                name
                in {"ok", "present", "stationary", "converged", "mechanism_present", "use_split"}
                and type(item) is not bool
            ):
                errors.append(f"{loc}:not_bool")
            if (
                name.startswith("n_")
                and item is not None
                and (not isinstance(item, int) or isinstance(item, bool) or item < 0)
            ):
                errors.append(f"{loc}:count_domain")
            if (
                item is not None
                and not isinstance(item, Mapping)
                and (name in _FRACTIONS or name.startswith("share_") or name.endswith("_share"))
                and (not _finite(item) or not 0 <= float(item) <= 1)
            ):
                errors.append(f"{loc}:fraction_domain")
            if isinstance(item, (int, float)) and not isinstance(item, bool) and not _finite(item):
                errors.append(f"{loc}:not_finite")
            _tree_domains(item, errors, loc, depth + 1)
    elif isinstance(value, list):
        if len(value) > 4096:
            errors.append(f"{path}:list_limit")
            return
        for index, item in enumerate(value):
            _tree_domains(item, errors, f"{path}[{index}]", depth + 1)


def _blocks(payload: Body, errors: list[str]) -> dict[str, Body]:
    blocks = {"real": _obj(payload.get("real"), "real", errors)}
    if "sim_arms" in payload:
        arms = _obj(payload.get("sim_arms"), "sim_arms", errors)
        if set(arms) != set(_ARMS):
            errors.append("sim_arms:identity")
        blocks.update({name: _obj(arms.get(name), name, errors) for name in _ARMS})
    elif "sim" in payload:
        blocks["sim"] = _obj(payload.get("sim"), "sim", errors)
    return blocks


def _simplex(value: object, name: str, errors: list[str], tolerance: float = 1e-8) -> Body:
    row = _obj(value, name, errors)
    if any(not _finite(v) or not 0 <= float(v) <= 1 for v in row.values()):
        errors.append(f"{name}:fraction_domain")
    elif row and abs(sum(float(v) for v in row.values()) - 1.0) > tolerance:
        errors.append(f"{name}:not_simplex")
    return row


def _hist_counts(value: object, expected: int, name: str, errors: list[str]) -> None:
    row = _obj(value, name, errors)
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in row.values()):
        errors.append(f"{name}:count_domain")
    elif row and sum(row.values()) != expected:
        errors.append(f"{name}:sum_mismatch")


def _abc(payload: Body) -> list[str]:
    errors: list[str] = []
    abc = _obj(payload.get("abc"), "abc", errors)
    target = _obj(abc.get("target"), "target", errors)
    accepted = _list(abc.get("accepted"), "accepted", errors)
    keep = _count(abc, "keep", errors, positive=True)
    draws = _count(abc, "n_draws", errors, positive=True)
    _count(abc, "horizon", errors, positive=True)
    if keep > draws or len(accepted) != keep:
        errors.append("abc:accepted_count")
    floors = {
        "sign_lag1": 0.05,
        "mo_fraction": 0.01,
        "spread_ticks_median": 0.5,
        "mid_move_std": 0.05,
    }
    distances: list[float] = []
    split_count = 0
    for index, item in enumerate(accepted):
        row = _obj(item, f"accepted_{index}", errors)
        measured = _obj(row.get("measured"), "measured", errors)
        params = _obj(row.get("params"), "params", errors)
        d2 = 0.0
        for metric, floor in floors.items():
            t = _num(target, metric, errors)
            v = _num(measured, metric, errors)
            d2 += ((v - t) / max(abs(t), floor)) ** 2
        expected = math.sqrt(d2 / len(floors))
        _near(row.get("distance"), expected, "accepted_distance", errors)
        distances.append(expected)
        if type(params.get("use_split")) is not bool:
            errors.append("use_split:not_bool")
        split_count += int(params.get("use_split") is True)
        for key in ("lam", "mu", "theta_cxl", "intensity_mult", "p_start", "density_exponent"):
            _num(params, key, errors, low=0)
        for key in ("band", "k_min"):
            _count(params, key, errors, positive=True)
    if distances:
        if distances != sorted(distances):
            errors.append("abc:accepted_not_sorted")
        _near(abc.get("min_distance"), min(distances), "min_distance", errors)
    if keep:
        _near(abc.get("use_split_share"), split_count / keep, "use_split_share", errors)
    median = _num(abc, "median_draw_distance", errors, low=0)
    claims = _obj(payload.get("claims"), "claims", errors)
    _eq(
        claims.get("split_selected_in_posterior"),
        split_count / max(keep, 1) > 0.5,
        "split_claim",
        errors,
    )
    _eq(
        claims.get("min_distance_under_median"),
        bool(distances and min(distances) < median),
        "distance_claim",
        errors,
    )
    full = _obj(payload.get("refit_full_horizon"), "refit_full_horizon", errors)
    for metric in floors:
        _num(full, metric, errors)
    # Rejected draws are not embedded; the median itself cannot be rederived.
    return errors


def _cancel(payload: Body) -> list[str]:
    errors: list[str] = []
    windows = _list(payload.get("windows_s"), "windows_s", errors)
    if any(not _finite(w) or float(w) <= 0 for w in windows):
        return errors + ["windows_s:domain"]
    for name, block in _blocks(payload, errors).items():
        _count(block, "n_execs", errors)
        sides = ("buy_side", "sell_side") if name == "real" else ("both_sides",)
        for side in sides:
            row = _obj(block.get(side), side, errors)
            if row.get("ok") is False:
                continue
            base = _num(row, "baseline_cxl_per_s", errors, low=0)
            for w in windows:
                post = _num(row, f"post_cxl_per_s_{float(w):g}", errors, low=0)
                _near(
                    row.get(f"lift_{float(w):g}s"),
                    post / base if base > 0 else None,
                    "cancel_lift",
                    errors,
                    8e-5,
                )
        if name != "real":
            _eq(block.get("mode"), "exec_on_exec_lift", "sim_cancel_proxy", errors)
    return errors


def _deep_microprice(payload: Body) -> list[str]:
    errors: list[str] = []
    max_depth = _count(payload, "max_depth", errors, positive=True)
    horizon = _count(payload, "fwd_events", errors, positive=True)
    blocks = _blocks(payload, errors)
    for block in blocks.values():
        if block.get("ok") is False:
            continue
        _count(block, "n", errors, positive=True)
        _eq(block.get("fwd_events"), horizon, "fwd_events", errors)
        rows = _list(block.get("per_depth"), "per_depth", errors)
        if len(rows) != max_depth or max_depth > 128:
            errors.append("per_depth:dimension")
        valid: list[Body] = []
        for index, item in enumerate(rows):
            row = _obj(item, "depth_row", errors)
            _eq(row.get("levels"), index + 1, "levels", errors)
            if row.get("ok") is False:
                continue
            corr = _num(row, "corr", errors)
            if abs(corr) > 1:
                errors.append("corr:domain")
            _near(row.get("r2"), corr * corr, "r2", errors)
            _num(row, "slope", errors)
            valid.append(row)
        if valid:
            best = max(valid, key=lambda row: float(row["r2"]))
            _eq(block.get("best_depth"), best["levels"], "best_depth", errors)
            _near(block.get("best_r2"), float(best["r2"]), "best_r2", errors)
    real = blocks["real"]
    expected = [
        f"{name}_bestdepth_{arm['best_depth']}_vs_{real['best_depth']}"
        for name, arm in blocks.items()
        if name != "real"
        and real.get("ok")
        and arm.get("ok")
        and arm.get("best_depth") != real.get("best_depth")
    ]
    _divergences(payload, expected, errors)
    return errors


def _depth_consumption(payload: Body) -> list[str]:
    errors: list[str] = []
    blocks = _blocks(payload, errors)
    for row in blocks.values():
        if row.get("ok") is False:
            continue
        _count(row, "n_fills", errors, positive=True)
        for key in (
            "mean_consumption",
            "median_consumption",
            "p90_consumption",
            "full_sweep_share",
            "over_sweep_share",
        ):
            _num(row, key, errors, low=0)
        if row.get("over_sweep_share", 0) > row.get("full_sweep_share", 0):
            errors.append("over_sweep_share:exceeds_full")
    real = blocks["real"]
    _eq(real.get("n_with_depth"), real.get("n_fills"), "n_with_depth", errors)
    if _count(real, "n_with_depth", errors) > _count(real, "n_exec_total", errors):
        errors.append("n_with_depth:exceeds_execs")
    gap = float(real.get("full_sweep_share", 0)) - float(blocks["sim"].get("full_sweep_share", 0))
    _divergences(payload, [f"full_sweep_share_gap_{gap:+.2f}"] if abs(gap) > 0.2 else [], errors)
    return errors


def _event_burst(payload: Body) -> list[str]:
    errors: list[str] = []
    blocks = _blocks(payload, errors)
    for block in blocks.values():
        for value in block.values():
            row = _obj(value, "burst_row", errors)
            if row.get("ok") is False:
                continue
            _count(row, "n_gaps", errors, positive=True)
            cv = _num(row, "cv", errors, low=0)
            _near(
                row.get("burstiness_B"),
                (cv * cv - 1) / (cv * cv + 1),
                "burstiness_B",
                errors,
                1.5e-4,
            )
            for key in ("mean_gap_ms", "median_gap_ms", "p99_gap_ms"):
                _num(row, key, errors, low=0)
    rb = blocks["real"].get("all", {}).get("burstiness_B")
    expected = []
    for name in _ARMS:
        ab = blocks.get(name, {}).get("all", {}).get("burstiness_B")
        if _finite(rb) and _finite(ab) and abs(ab - rb) > 0.15:
            expected.append(f"{name}_B_{ab}_vs_{rb}")
    _divergences(payload, expected, errors)
    return errors


def _event_granger(payload: Body) -> list[str]:
    errors: list[str] = []
    step = _num(payload, "bin_s", errors, low=0)
    max_lag = _num(payload, "max_lag_s", errors, low=0)
    if step <= 0 or max_lag < step:
        errors.append("lag_configuration:domain")
    blocks = _blocks(payload, errors)
    pairs = {
        f"{a}->{b}"
        for a in ("submit", "cancel_partial", "delete", "exec")
        for b in ("submit", "cancel_partial", "delete", "exec")
        if a != b
    }
    for block in blocks.values():
        if block.get("ok") is False:
            continue
        _count(block, "n_events", errors, positive=True)
        if not pairs.issubset(block):
            errors.append("granger_pairs:dimension")
        for pair in pairs:
            row = _obj(block.get(pair), pair, errors)
            for key in ("lag0_corr", "peak_corr", "peak_corr_lead"):
                if abs(_num(row, key, errors)) > 1:
                    errors.append(f"{key}:correlation_domain")
            lag = _count(row, "peak_lag_bins", errors)
            _near(row.get("peak_lag_s"), lag * step, "peak_lag_s", errors)
            if lag * step > max_lag + 1e-9:
                errors.append("peak_lag_s:out_of_window")
            lead = _num(row, "peak_lag_lead_s", errors, low=0)
            if lead < step or lead > max_lag + 1e-9:
                errors.append("peak_lag_lead_s:out_of_window")
    expected = []
    real = blocks["real"]
    if real.get("ok"):
        for name in _ARMS:
            arm = blocks.get(name, {})
            if not arm.get("ok"):
                continue
            for pair, vals in real.items():
                if isinstance(vals, Mapping) and "peak_corr" in vals:
                    s = arm.get(pair, {}).get("peak_corr")
                    if _finite(s) and abs(vals["peak_corr"] - s) > 0.15:
                        expected.append(f"{name}:{pair}_{s:.2f}_vs_{vals['peak_corr']:.2f}")
    _divergences(payload, expected, errors, unordered=True)
    return errors


def _event_matrix(payload: Body) -> list[str]:
    errors: list[str] = []
    for name, row in _blocks(payload, errors).items():
        n = _count(row, "n_events", errors, positive=True)
        span = _num(row, "span_s", errors, low=0)
        rates = _obj(row.get("events_per_s"), "events_per_s", errors)
        for key in rates:
            _num(rates, str(key), errors, low=0)
        if span <= 0:
            errors.append("span_s:not_positive")
            continue
        if name == "real":
            shares = _simplex(row.get("mix_share"), "mix_share", errors, 5e-4)
            for key, share in shares.items():
                _near(share, float(rates.get(key, 0)) * span / n, "mix_share", errors, 1e-4)
                if f"{key}|buy" in rates and f"{key}|sell" in rates:
                    _near(
                        rates.get(key),
                        rates[f"{key}|buy"] + rates[f"{key}|sell"],
                        "side_rates",
                        errors,
                        1.5e-4,
                    )
        else:
            _near(row.get("steps_per_s"), n / span, "steps_per_s", errors, 6e-4)
            _near(
                rates.get("exec"),
                rates.get("exec_buy", 0) + rates.get("exec_sell", 0),
                "exec_sides",
                errors,
                1.5e-4,
            )
    _divergences(payload, [], errors)
    return errors


def _exec_cost_real(payload: Body) -> list[str]:
    errors: list[str] = []
    for row in _blocks(payload, errors).values():
        n = _count(row, "n_execs", errors, positive=True)
        for key in ("mean_ticks_beyond_mid", "median_ticks", "share_negative_fills"):
            _num(row, key, errors)
        buckets = _obj(row.get("size_buckets"), "size_buckets", errors)
        count = 0
        for value in buckets.values():
            bucket = _obj(value, "size_bucket", errors)
            count += _count(bucket, "n", errors, positive=True)
            for key in ("mean_ticks_beyond_mid", "p90_ticks"):
                _num(bucket, key, errors)
        # Percentile endpoints can exclude some fills; equality is not claimed.
        if count > n:
            errors.append("size_buckets:exceed_execs")
    return errors


def _exec_cost_split(payload: Body) -> list[str]:
    errors: list[str] = []
    for key in ("parent_size", "n_children", "events_per_child"):
        _count(payload, key, errors, positive=True)
    arms = _obj(payload.get("arms"), "arms", errors)
    if set(arms) != set(_ARMS):
        errors.append("arms:identity")
    for arm in arms.values():
        row = _obj(arm, "arm", errors)
        _count(row, "n_episodes", errors, positive=True)
        for key in ("shortfall_ticks_mean", "mid_drift_ticks_mean", "fill_fraction_mean"):
            _num(row, key, errors)
        _num(row, "shortfall_ticks_sd", errors, low=0)
    claims = _obj(payload.get("claims"), "claims", errors)
    if all(name in arms for name in ("iid", "split")):
        _eq(
            claims.get("correlated_flow_raises_cost"),
            arms["split"]["shortfall_ticks_mean"] > arms["iid"]["shortfall_ticks_mean"],
            "cost_claim",
            errors,
        )
    return errors


def _glft(payload: Body) -> list[str]:
    errors: list[str] = []
    for key in ("A", "k", "sigma", "tick"):
        if _num(payload, key, errors, low=0) <= 0:
            errors.append(f"{key}:not_positive")
    _count(payload, "q_max", errors, positive=True)
    cells = _list(payload.get("cells"), "cells", errors)
    small: list[Body] = []
    for item in cells:
        row = _obj(item, "cell", errors)
        for key in (
            "gamma",
            "sigma",
            "T",
            "xi",
            "theta_max_abs_ode",
            "theta_max_abs_lin",
            "quote_err_ticks_max",
            "quote_err_ticks_mean",
        ):
            _num(row, key, errors, low=0)
        if row.get("T") not in payload.get("Ts", []):
            errors.append("cell:T_not_in_grid")
        _near(row.get("sigma"), float(payload.get("sigma", 0)), "cell_sigma", errors)
        if row.get("quote_err_ticks_mean", 0) > row.get("quote_err_ticks_max", 0):
            errors.append("quote_error:mean_exceeds_max")
        if row.get("xi", 1) < 0.05:
            small.append(row)
    _eq(
        payload.get("ok"),
        bool(small) and all(row["quote_err_ticks_max"] < 1 for row in small),
        "glft_ok",
        errors,
    )
    mc = _obj(payload.get("mc_check"), "mc_check", errors)
    pde = _num(mc, "value_pde", errors)
    value = _num(mc, "value_mc", errors)
    _num(mc, "mc_se", errors, low=0)
    _near(mc.get("abs_diff"), abs(value - pde), "mc_abs_diff", errors)
    return errors


def _hawkes_real(payload: Body) -> list[str]:
    errors: list[str] = []
    for row in _blocks(payload, errors).values():
        if row.get("ok") is False:
            continue
        _count(row, "n", errors, positive=True)
        eta = _num(row, "branching_ratio_eta", errors, low=0)
        if eta >= 1:
            errors.append("branching_ratio_eta:not_stationary")
        for key in ("mu", "decay_beta", "horizon_s"):
            if _num(row, key, errors, low=0) <= 0:
                errors.append(f"{key}:not_positive")
        hnll = _num(row, "hawkes_neg_loglik", errors)
        pnll = _num(row, "poisson_neg_loglik", errors)
        _near(row.get("loglik_gain_vs_poisson"), pnll - hnll, "loglik_gain", errors, 0.0015)
    return errors


def _hawkes_mv(payload: Body) -> list[str]:
    errors: list[str] = []
    marks = payload.get("marks")
    expected_marks = {"LO_ask_add", "LO_bid_add", "MO_buy", "MO_sell", "cancel_ask", "cancel_bid"}
    if not isinstance(marks, list) or len(marks) != 6 or set(marks) != expected_marks:
        return ["marks:identity"]
    blocks = _blocks(payload, errors)
    for row in blocks.values():
        counts = _obj(row.get("mark_counts"), "mark_counts", errors)
        if set(counts) != expected_marks:
            errors.append("mark_counts:dimension")
        _eq(row.get("n_events"), sum(_count(counts, m, errors) for m in marks), "n_events", errors)
        matrix = _obj(row.get("branching_matrix"), "branching_matrix", errors)
        if set(matrix) != expected_marks:
            errors.append("branching_matrix:dimension")
        b = np.zeros((6, 6), dtype=np.float64)
        for i, m in enumerate(marks):
            cells = _obj(matrix.get(m), "matrix_row", errors)
            if set(cells) != expected_marks:
                errors.append("branching_matrix:dimension")
            for j, mark in enumerate(marks):
                b[i, j] = _num(cells, mark, errors, low=0)
        # Perron monotonicity gives a real rounding bound for a nonnegative
        # matrix, including nonnormal matrices with ill-conditioned eigenvalues.
        # Both the matrix entries and the claimed rho are rounded to 5 decimals.
        half_unit = 5.0001e-6
        lower = float(np.max(np.abs(np.linalg.eigvals(np.maximum(b - half_unit, 0)))))
        upper = float(np.max(np.abs(np.linalg.eigvals(b + half_unit))))
        reported = _num(row, "rho_branching", errors, low=0)
        if reported + half_unit < lower or reported - half_unit > upper:
            errors.append("rho_branching:rederive_mismatch")
        # A boundary-straddling interval cannot determine stationarity from
        # rounded entries alone. Outside it, a contradictory flag must fail.
        if upper < 0.999:
            _eq(row.get("stationary"), True, "stationary", errors)
        elif lower >= 0.999:
            _eq(row.get("stationary"), False, "stationary", errors)
        shares = _obj(row.get("excitation_share"), "excitation_share", errors)
        endo = _obj(row.get("endo_share"), "endo_share", errors)
        for m in marks:
            values = _obj(shares.get(m), "excitation_row", errors)
            if set(values) != expected_marks:
                errors.append("excitation_share:dimension")
            total = sum(_num(values, mark, errors, low=0) for mark in marks)
            _near(endo.get(m), total, "endo_share", errors, 8e-5)
        for item in _list(row.get("dominant_pairs"), "dominant_pairs", errors):
            pair = _obj(item, "dominant_pair", errors)
            excited, excitor = pair.get("excited"), pair.get("excitor")
            if excited not in expected_marks or excitor not in expected_marks:
                errors.append("dominant_pair:mark_identity")
                continue
            _eq(pair.get("pair"), f"{excitor}->{excited}", "pair", errors)
            _eq(pair.get("self"), excited == excitor, "self", errors)
            _near(pair.get("b_mj"), matrix[excited][excitor], "b_mj", errors)
    expected_probes = {
        "real_tape_fit_ok": _finite(blocks["real"].get("loglik")),
        "real_tape_stationary": blocks["real"].get("stationary"),
        "real_tape_events_sufficient": blocks["real"].get("n_events", 0) >= 200,
        **{f"{name}_fit_ok": _finite(blocks.get(name, {}).get("loglik")) for name in _ARMS},
        **{f"{name}_stationary": blocks.get(name, {}).get("stationary") for name in _ARMS},
    }
    probes = _list(payload.get("probes"), "probes", errors)
    probe_names = []
    passed = 0
    for item in probes:
        probe = _obj(item, "probe", errors)
        probe_name = probe.get("name")
        probe_names.append(probe_name)
        if type(probe.get("passed")) is not bool:
            errors.append("probe:passed_not_bool")
        if probe_name not in expected_probes:
            errors.append("probe:name_identity")
        else:
            _eq(probe.get("passed"), expected_probes[probe_name], "probe_result", errors)
        passed += int(probe.get("passed") is True)
    if len(probe_names) != len(expected_probes) or set(probe_names) != set(expected_probes):
        errors.append("probes:identity")
    claim = _obj(payload.get("claim"), "claim", errors)
    _eq(claim.get("n_passed"), passed, "n_passed", errors)
    _eq(claim.get("n_probes"), len(probes), "n_probes", errors)
    _eq(claim.get("ok"), passed == len(probes), "hawkes_claim_ok", errors)
    results = _obj(claim.get("results"), "claim_results", errors)
    for name, row in blocks.items():
        _near(results.get(f"rho_{name}"), _num(row, "rho_branching", errors), "claim_rho", errors)
    top_pairs = blocks["real"].get("dominant_pairs", [])
    _eq(
        results.get("top_pair"),
        top_pairs[0].get("pair") if top_pairs else None,
        "claim_top_pair",
        errors,
    )
    parity = _obj(payload.get("parity_per_pair"), "parity_per_pair", errors)
    if set(parity) != {f"{source}->{target}" for source in marks for target in marks}:
        errors.append("parity_per_pair:dimension")
    for source in marks:
        for target in marks:
            pair = _obj(parity.get(f"{source}->{target}"), "parity_pair", errors)
            if set(pair) != set(blocks):
                errors.append("parity_pair:arm_identity")
            for name, row in blocks.items():
                _near(
                    pair.get(name),
                    row.get("branching_matrix", {}).get(target, {}).get(source),
                    "parity_value",
                    errors,
                )
    realrho = float(blocks["real"].get("rho_branching", 0))
    deltas = _obj(payload.get("delta_rho_vs_real"), "delta_rho_vs_real", errors)
    expected = []
    for name in _ARMS:
        rho = float(blocks.get(name, {}).get("rho_branching", 0))
        _near(deltas.get(name), realrho - rho, "delta_rho_vs_real", errors, 1.5e-5)
        if abs(rho - realrho) > 0.3:
            expected.append(f"{name}_rho_far_from_real")
    if realrho < 1e-3:
        expected.append("real_tape_no_excitation")
    _divergences(payload, expected, errors)
    return errors


def _hidden(payload: Body) -> list[str]:
    errors: list[str] = []
    blocks = _blocks(payload, errors)
    real, sim = blocks["real"], blocks["sim"]
    total = _count(real, "n_execs", errors, positive=True)
    hidden = _count(real, "n_hidden_execs", errors)
    if hidden > total:
        errors.append("n_hidden_execs:exceeds_execs")
    _near(
        real.get("hidden_trade_share"), hidden / max(total, 1), "hidden_trade_share", errors, 5.1e-5
    )
    _eq(_obj(real.get("hidden"), "hidden", errors).get("n"), hidden, "hidden_n", errors)
    _eq(_obj(real.get("visible"), "visible", errors).get("n"), total - hidden, "visible_n", errors)
    _eq(sim.get("mechanism_present"), False, "sim_hidden_mechanism", errors)
    _near(sim.get("hidden_trade_share"), 0.0, "sim_hidden_trade_share", errors)
    _near(sim.get("hidden_volume_share"), 0.0, "sim_hidden_volume_share", errors)
    _divergences(
        payload,
        ["sim_has_no_hidden_liquidity_mechanism"]
        if real.get("hidden_trade_share", 0) > 0.02
        else [],
        errors,
    )
    return errors


def _impact(payload: Body) -> list[str]:
    errors: list[str] = []
    blocks = _blocks(payload, errors)
    for row in blocks.values():
        if row.get("ok") is False:
            continue
        n = _count(row, "n", errors, positive=True)
        bins = _list(row.get("size_bins"), "size_bins", errors)
        count = 0
        for item in bins:
            cell = _obj(item, "size_bin", errors)
            count += _count(cell, "n", errors, positive=True)
            lo = _num(cell, "size_lo", errors, low=0)
            hi = _num(cell, "size_hi", errors, low=0)
            size = _num(cell, "mean_size", errors, low=0)
            if not lo <= size <= hi or lo >= hi:
                errors.append("size_bin:bounds")
        if count > n:
            errors.append("size_bins:exceed_observations")
        signed = _num(row, "mean_signed_dmid_ticks", errors)
        absolute = _num(row, "mean_abs_dmid_ticks", errors, low=0)
        if abs(signed) > absolute + 1e-9:
            errors.append("impact:signed_exceeds_absolute")
    expected = []
    real = blocks["real"]
    if real.get("ok"):
        small = next(
            (b for b in real["size_bins"] if b["size_lo"] <= 1.0 < b["size_hi"]),
            real["size_bins"][0],
        )
        for name in _ARMS:
            arm = blocks.get(name, {})
            if (
                arm.get("ok")
                and abs(small["mean_abs_dmid_ticks"] - arm["mean_abs_dmid_ticks"]) > 0.5
            ):
                expected.append(
                    f"{name}_unit_impact_{arm['mean_abs_dmid_ticks']:.2f}_vs_{small['mean_abs_dmid_ticks']:.2f}"
                )
        if real.get("alpha_loglog") is not None:
            expected.extend(
                f"{name}_no_size_dimension"
                for name in _ARMS
                if blocks.get(name, {}).get("ok") and blocks[name].get("alpha_loglog") is None
            )
    _divergences(payload, expected, errors)
    return errors


def _u_shape(profile: list[float]) -> float | None:
    if len(profile) < 3:
        return None
    k = max(len(profile) // 3, 1)
    mid = sum(profile[k:-k]) / len(profile[k:-k])
    return (sum(profile[:k]) + sum(profile[-k:])) / (2 * k * mid) if mid > 0 else None


def _intraday(payload: Body) -> list[str]:
    errors: list[str] = []
    blocks = _blocks(payload, errors)
    real = blocks["real"]
    bins = _count(real, "n_bins", errors, positive=True)
    profiles = {}
    for key in ("events_per_bin", "execs_per_bin", "mean_spread_per_bin"):
        profile = _list(real.get(key), key, errors)
        if len(profile) != bins or any(not _finite(v) or v < 0 for v in profile):
            errors.append(f"{key}:dimension_or_domain")
            continue
        profiles[key] = [float(v) for v in profile]
    if all(k in profiles for k in ("events_per_bin", "execs_per_bin")):
        events, execs = profiles["events_per_bin"], profiles["execs_per_bin"]
        if any(x > e for e, x in zip(events, execs, strict=True)):
            errors.append("execs_per_bin:exceed_events")
        _near(
            real.get("activity_u_shape"),
            _u_shape(execs if sum(execs) else events),
            "activity_u_shape",
            errors,
        )
    if "mean_spread_per_bin" in profiles:
        _near(
            real.get("spread_u_shape"),
            _u_shape([x for x in profiles["mean_spread_per_bin"] if x > 0]),
            "spread_u_shape",
            errors,
            1e-5,
        )
    return errors


def _lob_exec(payload: Body) -> list[str]:
    errors: list[str] = []
    cells = _obj(payload.get("cells"), "cells", errors)
    expected = {f"{size}_twap{steps}" for size in ("small", "medium", "large") for steps in (5, 20)}
    if set(cells) != expected:
        errors.append("lob_exec:cell_identity")
    for item in cells.values():
        row = _obj(item, "cell", errors)
        _count(row, "n_episodes", errors, positive=True)
        _count(row, "resyncs_total", errors)
        mean = _num(row, "is_total_ticks_mean", errors)
        low = _num(row, "is_total_ticks_min", errors)
        high = _num(row, "is_total_ticks_max", errors)
        if not low <= mean <= high:
            errors.append("shortfall:mean_outside_range")
        _num(row, "liquidity_cost_ticks_mean", errors, low=0)
        _num(row, "fill_fraction_mean", errors, low=0)
    return errors


def _resilience(payload: Body) -> list[str]:
    errors: list[str] = []
    blocks = _blocks(payload, errors)
    for name, block in blocks.items():
        rows = (
            [_obj(block.get(side), side, errors) for side in ("ask", "bid")]
            if name == "real"
            else [block]
        )
        for row in rows:
            if row.get("ok") is False:
                continue
            n = _count(row, "n_depletions", errors, positive=True)
            recovered = _count(row, "n_recovered", errors)
            unresolved = _count(row, "n_unrecovered", errors)
            _eq(n, recovered + unresolved, "n_depletions", errors)
            _near(
                row.get("recovery_share"), recovered / max(n, 1), "recovery_share", errors, 5.1e-5
            )
            for key in ("mean_refill_s", "median_refill_s", "p90_refill_s"):
                value = row.get(key)
                if value is not None:
                    _num(row, key, errors, low=0)
    realmedian = blocks["real"].get("ask", {}).get("median_refill_s")
    expected = []
    for name in _ARMS:
        amedian = blocks.get(name, {}).get("median_refill_s")
        if realmedian and amedian and abs(amedian - realmedian) / realmedian > 0.8:
            expected.append(f"{name}_refill_ratio_{amedian / realmedian:.2f}")
    _divergences(payload, expected, errors)
    return errors


def _marketable(payload: Body) -> list[str]:
    errors: list[str] = []
    blocks = _blocks(payload, errors)
    keys = {"behind", "at_own_touch", "inside", "marketable", "crossing"}
    for row in blocks.values():
        if row.get("ok") is False:
            continue
        _count(row, "n", errors, positive=True)
        for key in ("share", "volume_share"):
            fractions = _obj(row.get(key), key, errors)
            if set(fractions) != keys or any(
                not _finite(v) or not 0 <= v <= 1 for v in fractions.values()
            ):
                errors.append(f"{key}:identity_or_domain")
            else:
                # crossing is a subset of marketable, not a fifth partition.
                _near(sum(fractions[k] for k in keys - {"crossing"}), 1.0, f"{key}:simplex", errors)
                if fractions["crossing"] > fractions["marketable"]:
                    errors.append(f"{key}:crossing_exceeds_marketable")
    expected = []
    real = blocks["real"]
    if real.get("ok"):
        for key in ("marketable", "inside"):
            r = real["share"].get(key)
            for name in _ARMS:
                arm = blocks.get(name, {})
                s = arm.get("share", {}).get(key)
                if arm.get("ok") and _finite(r) and _finite(s) and abs(r - s) > 0.05:
                    expected.append(f"{name}_{key}_{s:.3f}_vs_{r:.3f}")
    _divergences(payload, expected, errors)
    return errors


def _metaorder(payload: Body) -> list[str]:
    errors: list[str] = []
    blocks = _blocks(payload, errors)
    for row in blocks.values():
        _count(row, "n_execs", errors, positive=True)
        _count(row, "n_metaorders", errors)
        for key in (
            "mean_fills_per_meta",
            "max_fills_per_meta",
            "p95_fills_per_meta",
            "mean_duration_s",
            "buy_share",
            "clustered_fill_share",
        ):
            _num(row, key, errors, low=0)
        if row.get("n_metaorders", 0) > row.get("n_execs", 0):
            errors.append("n_metaorders:exceeds_execs")
    detector = _obj(payload.get("detector"), "detector", errors)
    _num(detector, "gap_s", errors, low=0)
    _count(detector, "min_len", errors, positive=True)
    gap = blocks["real"].get("clustered_fill_share", 0) - blocks["sim"].get(
        "clustered_fill_share", 0
    )
    _divergences(
        payload, [f"clustered_fill_share_gap_{gap:+.2f}"] if abs(gap) > 0.15 else [], errors
    )
    return errors


def _mid_jump(payload: Body) -> list[str]:
    errors: list[str] = []
    blocks = _blocks(payload, errors)
    for row in blocks.values():
        n = _count(row, "n_obs", errors, positive=True)
        moves = _count(row, "n_mid_moves", errors)
        _near(row.get("move_share"), moves / max(n, 1), "move_share", errors, 5.1e-5)
        hist = _simplex(row.get("jump_hist"), "jump_hist", errors, 5e-4)
        _near(hist.get("0"), 1 - moves / max(n, 1), "zero_jump_share", errors, 5.1e-5)
        for key in ("mean_abs_jump_ticks", "p95_jump_ticks", "mean_move_gap_s"):
            _num(row, key, errors, low=0)
    r = blocks["real"].get("mean_abs_jump_ticks")
    expected = []
    for name in _ARMS:
        a = blocks.get(name, {}).get("mean_abs_jump_ticks")
        if r and a and abs(r - a) / r > 0.5:
            expected.append(f"{name}_jump_mean_ratio_{a / r:.2f}")
    _divergences(payload, expected, errors)
    return errors


def _lifetime(payload: Body) -> list[str]:
    errors: list[str] = []
    blocks = _blocks(payload, errors)
    real = blocks["real"]
    resolved = sum(_count(real, k, errors) for k in ("n_executed", "n_canceled", "n_deleted"))
    censored = _count(real, "n_open_at_end_censored", errors)
    _eq(real.get("n_orders_tracked"), resolved + censored, "n_orders_tracked", errors)
    _hist_counts(real.get("lifetime_hist_s"), resolved, "lifetime_hist_s", errors)
    for key in (
        "lifetime_s_p50_executed",
        "lifetime_s_p50_canceled",
        "lifetime_s_p50_deleted",
        "lifetime_s_p90_all",
    ):
        if real.get(key) is not None:
            _num(real, key, errors, low=0)
    for name in _ARMS:
        row = blocks.get(name, {})
        # Anonymous simulated queue-age proxies have no order census.
        if not row:
            errors.append(f"{name}:missing_proxy")
    return errors


def _order_revision(payload: Body) -> list[str]:
    errors: list[str] = []
    blocks = _blocks(payload, errors)
    for name, row in blocks.items():
        if row.get("ok") is False:
            continue
        for key in ("n_cancels", "n_submits"):
            _count(row, key, errors, positive=True)
        for key in (
            "revision_rate",
            "within_3_ticks_share",
            "median_latency_ms",
            "step_ticks_abs_median",
        ):
            _num(row, key, errors, low=0)
        if name == "real":
            for key in ("same_price_share", "toward_mid_share", "away_mid_share", "step_ticks_p90"):
                _num(row, key, errors, low=0)
            _near(
                sum(
                    row.get(k, 0)
                    for k in ("same_price_share", "toward_mid_share", "away_mid_share")
                ),
                1.0,
                "revision_direction_partition",
                errors,
            )
        if row.get("median_latency_ms", 0) > float(payload.get("window_s", 0)) * 1000:
            errors.append("median_latency_ms:outside_window")
    r = blocks["real"].get("revision_rate", 0)
    expected = [
        f"{name}_rate_{blocks[name]['revision_rate']:.3f}_vs_{r:.3f}"
        for name in _ARMS
        if blocks.get(name, {}).get("ok") and abs(blocks[name]["revision_rate"] - r) > 0.1
    ]
    _divergences(payload, expected, errors)
    return errors


def _drift(payload: Body) -> list[str]:
    errors: list[str] = []
    horizons = _list(payload.get("event_horizons"), "event_horizons", errors)
    blocks = _blocks(payload, errors)
    for block in blocks.values():
        if block.get("ok") is False:
            continue
        allrow = _obj(block.get("all"), "all", errors)
        movers = _obj(block.get("movers"), "movers", errors)
        nonmovers = _obj(block.get("nonmovers"), "nonmovers", errors)
        _eq(
            allrow.get("n"),
            _count(movers, "n", errors) + _count(nonmovers, "n", errors),
            "drift_partition_n",
            errors,
        )
        _eq(block.get("n_execs"), allrow.get("n"), "drift_n_execs", errors)
        for row in (allrow, movers, nonmovers):
            kernel = _obj(row.get("kernel"), "kernel", errors)
            if set(kernel) != {str(h) for h in horizons}:
                errors.append("kernel:horizon_identity")
            for cell in kernel.values():
                k = _obj(cell, "kernel_cell", errors)
                for key in ("mean_ticks", "median_ticks", "share_positive"):
                    _num(k, key, errors)
    expected = []
    real = blocks["real"]
    for name in _ARMS:
        arm = blocks.get(name, {})
        if not real.get("ok") or not arm.get("ok"):
            continue
        for h in horizons:
            r = real["all"]["kernel"][str(h)]["mean_ticks"]
            s = arm["all"]["kernel"][str(h)]["mean_ticks"]
            if abs(r - s) > 0.5:
                expected.append(f"{name}@{h}ev_{s:.2f}_vs_{r:.2f}")
    _divergences(payload, expected, errors)
    return errors


def _price_improvement(payload: Body) -> list[str]:
    errors: list[str] = []
    real = _obj(payload.get("real"), "real", errors)
    hidden = _count(real, "n_hidden", errors)
    priced = _count(real, "n_priced", errors)
    if priced > hidden:
        errors.append("n_priced:exceeds_hidden")
    _near(
        sum(
            _num(real, k, errors, low=0)
            for k in ("improved_share", "at_touch_share", "worse_share")
        ),
        1.0,
        "improvement_partition",
        errors,
        1.5e-4,
    )
    for key in ("mean_improvement_ticks", "median_improvement_ticks", "p90_improvement_ticks"):
        _num(real, key, errors)
    _divergences(
        payload,
        ["sim_has_no_hidden_mechanism_to_price_improve"]
        if real.get("improved_share", 0) > 0.1
        else [],
        errors,
    )
    return errors


def _propagator(payload: Body) -> list[str]:
    errors: list[str] = []
    table = _obj(payload.get("table"), "table", errors)
    real = _obj(table.get("real"), "real", errors)
    arms = _obj(table.get("sim_arms"), "sim_arms", errors)
    lags = _list(payload.get("lags_events"), "lags_events", errors)
    for row in (real, *arms.values()):
        block = _obj(row, "propagator_block", errors)
        _count(block, "n_events", errors, positive=True)
        _count(block, "n_trades", errors, positive=True)
        curve = _obj(block.get("response_ticks"), "response_ticks", errors)
        if set(curve) != {str(lag) for lag in lags}:
            errors.append("response_ticks:lag_identity")
        for key in curve:
            _num(curve, str(key), errors)
    if set(arms) != set(_ARMS):
        errors.append("sim_arms:identity")
    r = real.get("response_ticks", {}).get("1")
    expected = []
    for name, arm in arms.items():
        s = arm.get("response_ticks", {}).get("1")
        if _finite(r) and _finite(s) and abs(s) > 1e-9:
            # Six-decimal curve rounding is strongly amplified by near-zero
            # denominators. Bound that propagation rather than trust a ratio.
            tolerance = (
                5e-5 + 5.1e-7 / abs(s) + abs(r) * 5.1e-7 / (abs(s) * max(abs(s) - 5.1e-7, 1e-12))
            )
            _near(
                table.get(f"ratio_real_over_{name}@1"), r / s, "response_ratio", errors, tolerance
            )
        elif f"ratio_real_over_{name}@1" in table:
            errors.append("response_ratio:undefined")
        if r is None or s is None or abs(r - s) > 0.5 * max(abs(r), 0.05):
            expected.append(f"{name}_lag1_response_off")
    _divergences(payload, expected, errors)
    return errors


def _quote_place(payload: Body) -> list[str]:
    errors: list[str] = []
    blocks = _blocks(payload, errors)
    real = blocks["real"]
    n = _count(real, "n_submissions", errors)
    _hist_counts(real.get("dist_hist_ticks"), n, "dist_hist_ticks", errors)
    _near(
        sum(
            _num(real, key, errors, low=0)
            for key in ("share_improves_spread", "share_at_touch", "share_behind_touch")
        ),
        1.0,
        "placement_partition",
        errors,
        1.5e-4,
    )
    return errors


def _round_lot(payload: Body) -> list[str]:
    errors: list[str] = []
    blocks = _blocks(payload, errors)
    for row in blocks.values():
        n = _count(row, "n_trades", errors, positive=True)
        _hist_counts(row.get("size_hist"), n, "size_hist", errors)
        for key in (
            "mean_size",
            "median_size",
            "p95_size",
            "round_lot_share",
            "multiple_of_100_share",
        ):
            _num(row, key, errors, low=0)
        if row.get("multiple_of_100_share", 0) > row.get("round_lot_share", 0):
            errors.append("multiple_of_100_share:exceeds_round_lot")
    r = blocks["real"].get("round_lot_share", 0)
    expected = []
    for name in _ARMS:
        s = blocks.get(name, {}).get("round_lot_share", 0)
        if abs(r - s) > 0.15:
            expected.append(f"{name}_round_lot_share_gap_{r - s:+.2f}")
    _divergences(payload, expected, errors)
    return errors


def _sign_autocorr(payload: Body) -> list[str]:
    errors: list[str] = []
    lags = _list(payload.get("lags"), "lags", errors)
    blocks = _blocks(payload, errors)
    for row in blocks.values():
        n = _count(row, "n_execs", errors, positive=True)
        curve = _obj(row.get("curve"), "curve", errors)
        if set(curve) != {f"lag{lag}" for lag in lags if n >= lag + 2}:
            errors.append("curve:lag_identity")
        for key in curve:
            if abs(_num(curve, str(key), errors)) > 1:
                errors.append("curve:correlation_domain")
        positives = sum(v > 0 for v in curve.values())
        _eq(row.get("n_positive_lags"), positives, "n_positive_lags", errors)
        # The rounded curve is insufficient for an exact log-fit replay.
        if positives < 4 and row.get("exponent") is not None:
            errors.append("exponent:too_few_positive_lags")
    r = blocks["real"].get("exponent")
    expected = []
    for name in _ARMS:
        s = blocks.get(name, {}).get("exponent")
        if _finite(r) and _finite(s) and abs(s - r) > 0.2:
            expected.append(f"{name}_exponent_gap_{abs(s - r):.2f}")
    _divergences(payload, expected, errors)
    return errors


def _sign_predict(payload: Body) -> list[str]:
    errors: list[str] = []
    max_k = _count(payload, "k_max", errors, positive=True)
    blocks = _blocks(payload, errors)
    for row in blocks.values():
        n = _count(row, "n_signs", errors, positive=True)
        curve = _obj(row.get("continuation"), "continuation", errors)
        prev = n
        for key in sorted(curve, key=int):
            if not 1 <= int(key) <= max_k:
                errors.append("continuation:k_domain")
            cell = _obj(curve[key], "continuation_cell", errors)
            count = _count(cell, "n", errors, positive=True)
            if count >= n or count > prev:
                errors.append("continuation:count_order")
            if int(key) == 1:
                _eq(count, n - 1, "continuation_k1", errors)
            prev = count
            _num(cell, "p_continue", errors, low=0)
    r = blocks["real"].get("continuation", {}).get("5", {}).get("p_continue")
    expected = []
    for name in _ARMS:
        s = blocks.get(name, {}).get("continuation", {}).get("5", {}).get("p_continue")
        if r and s is not None and abs(s - r) > 0.1:
            expected.append(f"{name}_p5_{s}_vs_{r}")
    _divergences(payload, expected, errors)
    return errors


def _sim_real_ledger(payload: Body) -> list[str]:
    errors: list[str] = []
    rows = {
        name: _obj(payload.get(name), name, errors) for name in ("real", "sim_calm", "sim_regime")
    }
    for row in rows.values():
        _count(row, "n_events", errors, positive=True)
        _count(row, "n_trades", errors, positive=True)
        depth = _list(row.get("depth_l1_5_mean"), "depth_l1_5_mean", errors)
        if len(depth) != 5 or any(not _finite(v) or v < 0 for v in depth):
            errors.append("depth_l1_5_mean:dimension_or_domain")
        elif depth:
            _near(row.get("hump_level"), float(int(np.argmax(depth)) + 1), "hump_level", errors)
    table = _obj(payload.get("table"), "table", errors)
    for metric in ("sign_lag1", "mo_fraction", "spread_ticks_median", "mid_move_std"):
        values = _obj(table.get(metric), metric, errors)
        r = _num(rows["real"], metric, errors)
        for name in rows:
            _near(values.get(name), _num(rows[name], metric, errors), "ledger_table_value", errors)
        for side in ("calm", "regime"):
            s = _num(rows[f"sim_{side}"], metric, errors)
            _near(values.get(f"{side}_over_real"), s / r if r else None, "ledger_ratio", errors)
    return errors


def _split_flow(payload: Body) -> list[str]:
    errors: list[str] = []
    horizon = _count(payload, "horizon", errors, positive=True)
    lags = _list(payload.get("lags"), "lags", errors)
    arms: dict[str, Body] = {}
    for item in _list(payload.get("arms"), "arms", errors):
        row = _obj(item, "arm", errors)
        name = row.get("name")
        if name not in _ARMS or name in arms:
            errors.append("arms:identity")
            continue
        arms[name] = row
        _near(
            row.get("mo_fraction"),
            _count(row, "n_trades", errors) / max(horizon, 1),
            "mo_fraction",
            errors,
        )
        curve = _obj(row.get("autocorr"), "autocorr", errors)
        if set(curve) != {f"lag{lag}" for lag in lags}:
            errors.append("autocorr:lag_identity")
        for key in curve:
            if abs(_num(curve, str(key), errors)) > 1:
                errors.append("autocorr:correlation_domain")
    if set(arms) != set(_ARMS):
        errors.append("arms:identity")
    else:
        claims = _obj(payload.get("claims"), "claims", errors)
        _eq(
            claims.get("split_produces_persistence"),
            arms["split"]["autocorr"]["lag1"] > 0.2,
            "persistence_claim",
            errors,
        )
    return errors


def _spread_dynamics(payload: Body) -> list[str]:
    errors: list[str] = []
    blocks = _blocks(payload, errors)
    for row in blocks.values():
        _count(row, "n_obs", errors, positive=True)
        hist = _simplex(row.get("spread_occupancy"), "spread_occupancy", errors, 5e-4)
        _near(
            row.get("tight_share_le2"),
            hist.get("1-1", 0) + hist.get("2-2", 0),
            "tight_share_le2",
            errors,
            1.5e-4,
        )
        for key in ("mean_spread_ticks", "median_spread_ticks", "p90_spread_ticks"):
            _num(row, key, errors, low=0)
    simmean = blocks["sim"].get("mean_spread_ticks", 0)
    ratio = blocks["real"].get("mean_spread_ticks", 0) / simmean if simmean else 0
    _divergences(
        payload, [f"mean_spread_ratio_{ratio:.2f}"] if abs(ratio - 1) > 0.5 else [], errors
    )
    return errors


def _spread_response(payload: Body) -> list[str]:
    errors: list[str] = []
    horizons = _list(payload.get("horizons_s"), "horizons_s", errors)
    blocks = _blocks(payload, errors)
    for block in blocks.values():
        if block.get("ok") is False:
            continue
        sides = _obj(block.get("by_dir"), "by_dir", errors)
        for side in ("buy_initiated", "sell_initiated", "pooled"):
            row = _obj(sides.get(side), side, errors)
            _count(row, "n", errors)
            kernel = _obj(row.get("kernel"), "kernel", errors)
            if set(kernel) != {f"{h:g}s" for h in horizons}:
                errors.append("kernel:horizon_identity")
            for item in kernel.values():
                cell = _obj(item, "cell", errors)
                for key in ("mean_delta", "median_delta", "share_wider"):
                    _num(cell, key, errors)
        if all(s in sides for s in ("buy_initiated", "sell_initiated", "pooled")):
            _eq(
                sides["pooled"]["n"],
                sides["buy_initiated"]["n"] + sides["sell_initiated"]["n"],
                "response_partition_n",
                errors,
            )
    expected = []
    real = blocks["real"]
    for name in _ARMS:
        arm = blocks.get(name, {})
        if real.get("ok") and arm.get("ok"):
            for h in horizons:
                r = real["by_dir"]["pooled"]["kernel"][f"{h:g}s"]["median_delta"]
                s = arm["by_dir"]["pooled"]["kernel"][f"{h:g}s"]["median_delta"]
                if abs(r - s) > 2:
                    expected.append(f"{name}@{h}s_{s:.1f}_vs_{r:.1f}")
    _divergences(payload, expected, errors)
    return errors


def _stale_quote(payload: Body) -> list[str]:
    errors: list[str] = []
    blocks = _blocks(payload, errors)
    for row in blocks.values():
        if row.get("ok") is False:
            continue
        n = _count(row, "n", errors, positive=True)
        if _count(row, "n_with_drift", errors) > n:
            errors.append("n_with_drift:exceeds_n")
        bins = _list(row.get("age_bins"), "age_bins", errors)
        total = 0
        for item in bins:
            cell = _obj(item, "age_bin", errors)
            count = _count(cell, "n", errors)
            total += count
            _near(cell.get("share"), count / max(n, 1), "age_bin_share", errors)
        _eq(total, n, "age_bin_total", errors)
    real = blocks["real"]
    r = real.get("share_age_gt_5s", 0)
    expected = [
        f"{name}_oldfill_{blocks[name]['share_age_gt_5s']:.3f}_vs_{r:.3f}"
        for name in _ARMS
        if blocks.get(name, {}).get("ok") and abs(blocks[name]["share_age_gt_5s"] - r) > 0.1
    ]
    _divergences(payload, expected, errors)
    return errors


def _streak(payload: Body) -> list[str]:
    errors: list[str] = []
    blocks = _blocks(payload, errors)
    geo_tail = sum(0.5**k for k in range(6, 21)) / sum(0.5**k for k in range(1, 21))
    for row in blocks.values():
        n = _count(row, "n_runs", errors, positive=True)
        executions = _count(row, "n_execs", errors, positive=True)
        _near(row.get("mean_run"), executions / max(n, 1), "mean_run", errors, 5.1e-4)
        hist = _simplex(row.get("run_hist"), "run_hist", errors, 0.001)
        if set(hist) != {str(k) for k in range(1, 21)}:
            errors.append("run_hist:bin_identity")
            continue
        _near(row.get("p_ge_5"), sum(hist[str(k)] for k in range(5, 21)), "p_ge_5", errors, 9e-4)
        _near(row.get("p_ge_10"), sum(hist[str(k)] for k in range(10, 21)), "p_ge_10", errors, 7e-4)
        _near(
            row.get("excess_mass_gt5_vs_geo"),
            sum(hist[str(k)] for k in range(6, 21)) - geo_tail,
            "excess_mass",
            errors,
            9e-4,
        )
    r = blocks["real"].get("excess_mass_gt5_vs_geo")
    expected = []
    for name in _ARMS:
        s = blocks.get(name, {}).get("excess_mass_gt5_vs_geo")
        if _finite(r) and _finite(s) and abs(r - s) > 0.05:
            expected.append(f"{name}_run_tail_gap_{r - s:+.3f}")
    _divergences(payload, expected, errors)
    return errors


def _tick_rule(payload: Body) -> list[str]:
    errors: list[str] = []
    for row in _blocks(payload, errors).values():
        n = _count(row, "n_execs", errors, positive=True)
        for key in ("quote_rule", "tick_rule"):
            rule = _obj(row.get(key), key, errors)
            _eq(rule.get("rule"), key, "rule_identity", errors)
            _eq(rule.get("n"), n, "rule_n", errors)
            if _count(rule, "undecided", errors) > n:
                errors.append("undecided:exceeds_n")
            _num(rule, "misclass_rate", errors, low=0)
    return errors


def _vol_signature(payload: Body) -> list[str]:
    errors: list[str] = []
    taus = _list(payload.get("taus_s"), "taus_s", errors)
    if any(not _finite(t) or t <= 0 for t in taus):
        return errors + ["taus_s:domain"]
    blocks = _blocks(payload, errors)
    for row in blocks.values():
        _count(row, "n_samples", errors, positive=True)
        rv = _obj(row.get("rv_per_tau"), "rv_per_tau", errors)
        if set(rv) != {str(float(t)) for t in taus}:
            errors.append("rv_per_tau:tau_identity")
        values = [_num(rv, str(float(t)), errors, low=0) for t in taus]
        if values:
            _near(
                row.get("rv_fine_to_coarse_ratio"),
                values[0] / values[-1] if values[-1] else None,
                "rv_ratio",
                errors,
                5.1e-5,
            )
    r = blocks["real"].get("rv_fine_to_coarse_ratio")
    expected = []
    for name in _ARMS:
        s = blocks.get(name, {}).get("rv_fine_to_coarse_ratio")
        if r and s and abs(s - r) / r > 0.8:
            expected.append(f"{name}_signature_ratio_{s:.2f}_vs_{r}")
    _divergences(payload, expected, errors)
    return errors


def _vpin(payload: Body) -> list[str]:
    errors: list[str] = []
    for row in _blocks(payload, errors).values():
        if row.get("ok") is False:
            continue
        _count(row, "n_buckets", errors, positive=True)
        if _num(row, "bucket_volume", errors, low=0) <= 0:
            errors.append("bucket_volume:not_positive")
        for key in ("vpin_mean", "vpin_p90"):
            _num(row, key, errors, low=0)
        if abs(_num(row, "vpin_fwd_vol_corr", errors)) > 1:
            errors.append("vpin_fwd_vol_corr:domain")
    return errors


def _tape_digest(payload: Body) -> list[str]:
    errors: list[str] = []
    lanes = _obj(payload.get("lanes"), "lanes", errors)
    scores = {name: 0 for name in _ARMS}
    present = scored = 0
    uncovered = []
    for name, item in lanes.items():
        row = _obj(item, "lane", errors)
        if type(row.get("present")) is not bool:
            errors.append("lane_present:not_bool")
        if not row.get("present"):
            continue
        present += 1
        if not isinstance(row.get("receipt_sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", row["receipt_sha256"]
        ):
            errors.append("lane_receipt_sha256:domain")
        statuses = _obj(row.get("arm_status"), "arm_status", errors)
        if set(statuses) != set(_ARMS) or any(
            v not in ("covered", "diverged", "unscored") for v in statuses.values()
        ):
            errors.append("arm_status:identity_or_domain")
        scored += int(all(s != "unscored" for s in statuses.values()))
        for arm in _ARMS:
            scores[arm] += int(statuses.get(arm) == "covered")
        if all(statuses.get(arm) == "diverged" for arm in _ARMS):
            uncovered.append(name)
    _eq(payload.get("n_lanes_present"), present, "n_lanes_present", errors)
    _eq(payload.get("n_lanes_scored"), scored, "n_lanes_scored", errors)
    _eq(payload.get("arm_coverage_score"), scores, "arm_coverage_score", errors)
    _eq(
        payload.get("best_arm"),
        max(scores, key=lambda k: scores[k]) if scored else None,
        "best_arm",
        errors,
    )
    claimed_uncovered = payload.get("uncovered_lanes")
    if (
        not isinstance(claimed_uncovered, list)
        or any(not isinstance(v, str) for v in claimed_uncovered)
        or sorted(claimed_uncovered) != sorted(uncovered)
    ):
        errors.append("uncovered_lanes:rederive_mismatch")
    # Child hashes are attestations here; no child content is embedded.
    return errors


def _divergences(
    payload: Body, expected: list[str], errors: list[str], *, unordered: bool = False
) -> None:
    claimed = payload.get("divergences")
    if not isinstance(claimed, list) or any(not isinstance(v, str) for v in claimed):
        errors.append("divergences:missing_list")
    elif (sorted(claimed) if unordered else claimed) != (
        sorted(expected) if unordered else expected
    ):
        errors.append("divergences:rederive_mismatch")


_CHECKS: dict[str, Checker] = {
    "abc_calibrate": _abc,
    "cancel_cluster": _cancel,
    "deep_microprice": _deep_microprice,
    "depth_consumption": _depth_consumption,
    "event_burst": _event_burst,
    "event_granger": _event_granger,
    "event_matrix": _event_matrix,
    "exec_cost_real": _exec_cost_real,
    "exec_cost_split": _exec_cost_split,
    "glft_bench": _glft,
    "hawkes_mv": _hawkes_mv,
    "hawkes_real": _hawkes_real,
    "hidden_depth": _hidden,
    "impact_instant": _impact,
    "intraday_shape": _intraday,
    "lob_exec": _lob_exec,
    "lob_resilience": _resilience,
    "marketable_limit": _marketable,
    "metaorder_detect": _metaorder,
    "mid_jump": _mid_jump,
    "order_lifetime": _lifetime,
    "order_revision": _order_revision,
    "post_trade_drift": _drift,
    "price_improvement": _price_improvement,
    "propagator_real": _propagator,
    "quote_place": _quote_place,
    "round_lot": _round_lot,
    "sign_autocorr_real": _sign_autocorr,
    "sign_predict": _sign_predict,
    "sim_real_ledger": _sim_real_ledger,
    "split_flow": _split_flow,
    "spread_dynamics": _spread_dynamics,
    "spread_response": _spread_response,
    "stale_quote": _stale_quote,
    "streak_stats": _streak,
    "tape_digest": _tape_digest,
    "tick_rule": _tick_rule,
    "vol_signature": _vol_signature,
    "vpin": _vpin,
}


def _check(name: str, payload: Body) -> list[str]:
    errors: list[str] = []
    _eq(payload.get("schema"), f"{name}.v1", "schema", errors)
    _eq(payload.get("kind"), "split_flow_bench" if name == "split_flow" else name, "kind", errors)
    _eq(payload.get("research_only"), True, "research_only", errors)
    expected_labels = {"SYNTHETIC"} if name in _SYNTHETIC else {"MIXED"}
    if name == "price_improvement":
        expected_labels = {"REAL"}
    if name == "hawkes_mv":
        expected_labels.add("SYNTHETIC")
    if payload.get("data_label") not in expected_labels:
        errors.append("data_label:lane_honesty")
    revision = payload.get("git_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        errors.append("git_revision:domain")
    if payload.get("live_pnl_claim", False) is not False:
        errors.append("live_pnl_claim:not_false")
    self_contained = {
        "abc_calibrate",
        "exec_cost_split",
        "glft_bench",
        "lob_exec",
        "price_improvement",
        "propagator_real",
        "sim_real_ledger",
        "split_flow",
        "tape_digest",
    }
    if name not in self_contained:
        for key in (
            "real",
            "sim"
            if name in {"depth_consumption", "hidden_depth", "metaorder_detect", "spread_dynamics"}
            else "sim_arms",
        ):
            _obj(payload.get(key), key, errors)
    _tree_domains(payload, errors)
    if errors:
        return errors
    try:
        errors.extend(_CHECKS[name](payload))
    except (
        KeyError,
        TypeError,
        ValueError,
        ZeroDivisionError,
        OverflowError,
        np.linalg.LinAlgError,
    ):
        errors.append(f"{name}:malformed_detail")
    return errors


MICROSTRUCTURE_RECEIPT_CONTRACTS: dict[str, Checker] = {
    f"{name}.v1": partial(_check, name) for name in _CHECKS
}


def microstructure_receipt_contract_errors(payload: Body) -> list[str]:
    """Dispatch by declared schema or recognized kind, preserving rename guards."""
    schema = payload.get("schema")
    if isinstance(schema, str) and schema in MICROSTRUCTURE_RECEIPT_CONTRACTS:
        return MICROSTRUCTURE_RECEIPT_CONTRACTS[schema](payload)
    kind = payload.get("kind")
    name = "split_flow" if kind == "split_flow_bench" else kind
    if isinstance(name, str) and name in _CHECKS:
        return _check(name, payload)
    return []
