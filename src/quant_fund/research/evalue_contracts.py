"""Contract checks for the anytime-valid/sequential receipt family.

Forward-compatible deep verification for kinds emitted by the
e-process lanes (``evalue_promotion.v1``, ``fleet_race.v1``,
``corpus_inference.v1``, ``online_fdr.v1``): every embedded claim is
re-derived from the payload itself — a receipt that declares a promotion
whose arithmetic doesn't hold fails closed, seal or no seal.

Hooked into ``verify_receipt_payload`` by ``receipt_v2`` on the ``kind``
field, so a v1-shaped receipt of one of these kinds gets the contract
before the seal check alone could pass it.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from typing import Any

import numpy as np

EVALUE_FAMILY_KINDS = frozenset(
    {
        "evalue_promotion.v1",
        "fleet_race.v1",
        "corpus_inference.v1",
        "online_fdr.v1",
        "calibration_audit.v1",
        "loss_cs.v1",
        "changepoint_localize.v1",
        "coverage_audit.v1",
        "coverage_cs.v1",
        "winner_curse.v1",
        "drift_alarm.v1",
        "conformal_monitor.v1",
        # these writers emit kind="tail_audit"/"lane_power" (schema on `schema`)
        "tail_audit",
        "lane_power",
        "honest_verdict.v1",
        "monitor_run",
        "suite_health",
        "mcs_seq",
        "serial_watch",
        # drill receipts emit bare kinds (schema carries the .v1 tag)
        "coverage_cs",
        "coverage_audit",
        "emerge_drill.v1",
        "panel_audit.v1",
    }
)

_SHA_LEN = 64


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA_LEN
        and all(character in "0123456789abcdef" for character in value)
    )


def _num(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _finite(value: object) -> bool:
    v = _num(value)
    return v is not None and bool(np.isfinite(v))


def _evalue_promotion_errors(p: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("alpha", "lam"):
        v = _num(p.get(field))
        if v is None or not (0.0 < v < 1.0):
            errors.append(f"{field}_out_of_unit_interval")
    n_origins = p.get("n_origins")
    if not isinstance(n_origins, int) or isinstance(n_origins, bool) or n_origins < 1:
        errors.append("n_origins_not_positive_int")
    ev = _num(p.get("final_evalue"))
    ap = _num(p.get("anytime_p"))
    if ev is None or ev <= 0.0 or not np.isfinite(ev):
        errors.append("final_evalue_not_positive_finite")
    if ap is None or not (0.0 < ap <= 1.0):
        errors.append("anytime_p_out_of_unit_interval")
    if ev is not None and ap is not None and ev > 0.0:
        expected = min(1.0, 1.0 / ev)
        if not np.isclose(ap, expected, rtol=1e-6, atol=1e-9):
            errors.append("anytime_p_not_reciprocal_of_evalue")
    origin = p.get("promotion_origin")
    promoted = p.get("promoted")
    if promoted not in (True, False):
        errors.append("promoted_not_bool")
    elif isinstance(n_origins, int):
        if promoted and (
            not isinstance(origin, int)
            or isinstance(origin, bool)
            or origin < 0
            or origin >= n_origins
        ):
            errors.append("promotion_origin_out_of_range")
        if not promoted and origin is not None:
            errors.append("promotion_origin_set_without_promotion")
    return errors


def _fleet_race_errors(p: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    shard_meta = p.get("shard_meta")
    if isinstance(shard_meta, Mapping) and shard_meta:
        distinct_fr = {
            str(m.get("data_label") or "UNKNOWN")
            for m in shard_meta.values()
            if isinstance(m, Mapping)
        }
        expected_fr = next(iter(distinct_fr)) if len(distinct_fr) == 1 else None
        if expected_fr is None or p.get("data_label") != expected_fr:
            errors.append("data_label_mismatches_shard_labels")
    elif p.get("data_label") != "SYNTHETIC":
        errors.append("data_label_not_synthetic")
    if p.get("research_only") is not True:
        errors.append("research_only_not_true")
    if p.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    if not _is_sha256(p.get("inputs_sha256")):
        errors.append("inputs_sha256_invalid")
    params = p.get("params")
    if not isinstance(params, Mapping):
        errors.append("params_missing")
    else:
        for field in ("n_train", "n_eval", "n_chunks"):
            v = params.get(field)
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                errors.append(f"params_{field}_not_positive_int")
        alpha = _num(params.get("alpha"))
        if alpha is None or not (0.0 < alpha < 1.0):
            errors.append("params_alpha_out_of_unit_interval")
        taus = params.get("taus")
        if isinstance(taus, list):
            tv = [float(t) for t in taus if isinstance(t, (int, float))]
            if (
                len(tv) != len(taus)
                or any(t <= 0.0 or t >= 1.0 for t in tv)
                or any(b <= a for a, b in pairwise(tv))
            ):
                errors.append("params_taus_not_strictly_increasing_unit_grid")
        else:
            errors.append("params_taus_not_list")
    winners = p.get("shard_winners")
    if not isinstance(winners, Mapping):
        errors.append("shard_winners_not_mapping")
    n_shards = p.get("n_shards")
    if not isinstance(n_shards, int) or isinstance(n_shards, bool) or n_shards < 1:
        errors.append("n_shards_not_positive_int")
    n_models = p.get("n_models")
    if not isinstance(n_models, int) or isinstance(n_models, bool) or n_models < 1:
        errors.append("n_models_not_positive_int")
    return errors


def _corpus_inference_errors(p: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("n_receipts", "n_p_findings", "n_e_findings", "n_survivors"):
        v = p.get(field)
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            errors.append(f"{field}_not_nonnegative_int")
    if not _is_sha256(p.get("inputs_sha256")):
        errors.append("inputs_sha256_invalid")
    params = p.get("params")
    q = _num(params.get("q")) if isinstance(params, Mapping) else None
    if q is None or not (0.0 < q < 1.0):
        errors.append("params_q_out_of_unit_interval")
    input_labels_ci = params.get("input_labels") if isinstance(params, Mapping) else None
    if isinstance(input_labels_ci, Mapping) and input_labels_ci:
        distinct_ci = set(input_labels_ci.values())
        expected_ci = next(iter(distinct_ci)) if len(distinct_ci) == 1 else "MIXED"
        if p.get("data_label") != expected_ci:
            errors.append("data_label_mismatches_input_labels")
    parse_errors = p.get("parse_errors")
    n_err = p.get("n_parse_errors")
    if isinstance(parse_errors, list) and isinstance(n_err, int):
        if len(parse_errors) != n_err:
            errors.append("parse_errors_length_mismatch")
    elif parse_errors is not None:
        errors.append("parse_errors_not_list")
    survivors = p.get("surviving_claims")
    n_surv = p.get("n_survivors")
    if isinstance(survivors, list):
        if isinstance(n_surv, int) and len(survivors) != n_surv:
            errors.append("surviving_claims_length_mismatch")
    elif survivors is not None:
        errors.append("surviving_claims_not_list")
    ce = _num(p.get("corpus_evalue"))
    n_e = p.get("n_e_findings")
    if ce is None or ce <= 0.0 or not np.isfinite(ce):
        errors.append("corpus_evalue_not_positive_finite")
    reject = p.get("corpus_reject_at_alpha")
    if reject not in (True, False):
        errors.append("corpus_reject_not_bool")
    elif ce is not None and q is not None and isinstance(n_e, int) and n_e > 0 and ce > 0.0:
        expected = ce >= 1.0 / q
        if reject != expected:
            errors.append("corpus_reject_inconsistent_with_evalue")
    return errors


def _online_fdr_errors(p: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    n_tests = p.get("n_tests")
    n_rej = p.get("n_rejections")
    for name, v in (("n_tests", n_tests), ("n_rejections", n_rej)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            errors.append(f"{name}_not_nonnegative_int")
    if isinstance(n_tests, int) and isinstance(n_rej, int) and n_rej > n_tests:
        errors.append("n_rejections_exceeds_n_tests")
    indices = p.get("rejection_indices")
    if isinstance(indices, list):
        if any(not isinstance(i, int) or isinstance(i, bool) for i in indices):
            errors.append("rejection_indices_not_ints")
        elif any(b <= a for a, b in pairwise(indices)):
            errors.append("rejection_indices_not_strictly_increasing")
        elif isinstance(n_tests, int) and any(i < 0 or i >= n_tests for i in indices):
            errors.append("rejection_indices_out_of_range")
        elif isinstance(n_rej, int) and len(indices) != n_rej:
            errors.append("rejection_indices_length_mismatch")
    elif indices is not None:
        errors.append("rejection_indices_not_list")
    wealth = _num(p.get("final_wealth"))
    if wealth is None or wealth < 0.0 or not np.isfinite(wealth):
        errors.append("final_wealth_not_nonnegative_finite")
    level = _num(p.get("level"))
    if level is None or not (0.0 < level < 1.0):
        errors.append("level_out_of_unit_interval")
    return errors


def _calibration_audit_errors(p: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    params = p.get("params")
    if not isinstance(params, Mapping):
        return ["missing_params"]
    for field in ("alpha",):
        v = _num(params.get(field))
        if v is None or not (0.0 < v < 1.0):
            errors.append(f"{field}_out_of_unit_interval")
    channels = params.get("channels")
    if not isinstance(channels, list) or not channels:
        errors.append("channels_empty")
    if not _is_sha256(p.get("inputs_sha256")):
        errors.append("inputs_sha256_malformed")
    return errors


def _loss_cs_errors(p: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("alpha", "lam", "bound"):
        v = _num(p.get(field))
        if v is None or not (0.0 < v < 1.0 if field != "bound" else v > 0.0):
            errors.append(f"{field}_invalid")
    lo, hi = _num(p.get("cs_low")), _num(p.get("cs_high"))
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
        errors.append("cs_bounds_malformed")
    n = p.get("n")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        errors.append("n_not_positive_int")
    if p.get("interpretation") not in {
        "challenger_better",
        "incumbent_better",
        "inconclusive",
    }:
        errors.append("interpretation_unknown")
    return errors


def _changepoint_localize_errors(p: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    params = p.get("params")
    if not isinstance(params, Mapping):
        return ["missing_params"]
    for field in ("alpha",):
        v = _num(params.get(field))
        if v is None or not (0.0 < v < 1.0):
            errors.append(f"{field}_out_of_unit_interval")
    for field in ("window", "min_left"):
        v = params.get(field)
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            errors.append(f"{field}_not_positive_int")
    result = p.get("result")
    if isinstance(result, Mapping):
        lo, hi = _num(result.get("cs_lo")), _num(result.get("cs_hi"))
        n = _num(result.get("n"))
        if lo is not None and hi is not None and lo > hi:
            errors.append("cs_bounds_inverted")
        if n is None or n < 1:
            errors.append("n_not_positive")
    return errors


def _coverage_audit_errors(p: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    params = p.get("params")
    if not isinstance(params, Mapping):
        return ["missing_params"]
    alpha = _num(params.get("alpha"))
    if alpha is None or not (0.0 < alpha < 1.0):
        errors.append("alpha_out_of_unit_interval")
    levels = params.get("levels")
    if not isinstance(levels, list) or not all(
        isinstance(lv, (int, float)) and 0.0 < float(lv) < 1.0 for lv in levels
    ):
        errors.append("levels_malformed")
    if not _is_sha256(p.get("inputs_sha256")):
        errors.append("inputs_sha256_malformed")
    return errors


def _coverage_cs_errors(p: Mapping[str, Any]) -> list[str]:
    # same envelope shape as coverage_audit
    return _coverage_audit_errors(p)


def _winner_curse_errors(p: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if p.get("data_label") != "SYNTHETIC":
        errors.append("data_label_not_synthetic")
    if p.get("research_only") is not True:
        errors.append("research_only_not_true")
    if p.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    for field in ("n_obs", "n_heads", "n_boot"):
        v = p.get(field)
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            errors.append(f"{field}_not_positive_int")
    for field in ("naive_score", "selection_bias", "corrected_score", "honest_score"):
        if not _finite(p.get(field)):
            errors.append(f"{field}_not_finite")
    bias = _num(p.get("selection_bias"))
    verdict = p.get("verdict")
    if verdict not in ("bias_material", "bias_negligible"):
        errors.append("verdict_not_in_enum")
    elif bias is not None:
        # writer: bias_material iff selection_bias > 1e-12
        if verdict == "bias_material" and bias <= 0.0:
            errors.append("bias_material_without_bias")
        if verdict == "bias_negligible" and bias < 0.0:
            errors.append("bias_negligible_with_negative_bias")
    for field in ("naive_ci", "selection_aware_ci"):
        ci = p.get(field)
        if not (isinstance(ci, list) and len(ci) == 2 and all(_finite(x) for x in ci)):
            errors.append(f"{field}_not_finite_pair")
        elif ci[1] < ci[0]:
            errors.append(f"{field}_inverted")
    return errors


def _drift_alarm_errors(p: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if p.get("research_only") is not True:
        errors.append("research_only_not_true")
    if p.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    alpha = _num(p.get("alpha"))
    if alpha is None or not (0.0 < alpha < 1.0):
        errors.append("alpha_out_of_unit_interval")
    n_obs = p.get("n_obs")
    if not isinstance(n_obs, int) or isinstance(n_obs, bool) or n_obs < 1:
        errors.append("n_obs_not_positive_int")
    eproc = p.get("eprocess")
    if not isinstance(eproc, Mapping):
        errors.append("eprocess_not_mapping")
    else:
        lam = _num(eproc.get("lam"))
        if lam is None or not (0.0 < lam < 1.0):
            errors.append("eprocess_lam_out_of_unit_interval")
        ev = _num(eproc.get("final_evalue"))
        if ev is None or ev <= 0.0 or not np.isfinite(ev):
            errors.append("final_evalue_not_positive_finite")
        alarmed = eproc.get("alarmed")
        if alarmed not in (True, False):
            errors.append("eprocess_alarmed_not_bool")
        idx = eproc.get("alarm_index")
        if alarmed is True:
            if (
                not isinstance(idx, int)
                or isinstance(idx, bool)
                or idx < 0
                or (isinstance(n_obs, int) and idx >= n_obs)
            ):
                errors.append("alarm_index_out_of_range")
        elif idx is not None:
            errors.append("alarm_index_set_without_alarm")
    return errors


def _conformal_monitor_errors(p: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if p.get("research_only") is not True:
        errors.append("research_only_not_true")
    if p.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    alpha = _num(p.get("alpha"))
    if alpha is None or not (0.0 < alpha < 1.0):
        errors.append("alpha_out_of_unit_interval")
    kappa = _num(p.get("kappa"))
    if kappa is None or kappa <= 0.0 or not np.isfinite(kappa):
        errors.append("kappa_not_positive_finite")
    window = p.get("window")
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        errors.append("window_not_positive_int")
    n_obs = p.get("n_obs")
    if not isinstance(n_obs, int) or isinstance(n_obs, bool) or n_obs < 1:
        errors.append("n_obs_not_positive_int")
    m = _num(p.get("final_martingale"))
    if m is None or m < 0.0 or not np.isfinite(m):
        errors.append("final_martingale_not_nonnegative_finite")
    alarmed = p.get("alarmed")
    if alarmed not in (True, False):
        errors.append("alarmed_not_bool")
    idx = p.get("alarm_index")
    if alarmed is True and (
        not isinstance(idx, int)
        or isinstance(idx, bool)
        or idx < 0
        or (isinstance(n_obs, int) and idx >= n_obs)
    ):
        errors.append("alarm_index_out_of_range")
    return errors


def _tail_audit_errors(p: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not _is_sha256(p.get("inputs_sha256")):
        errors.append("inputs_sha256_invalid")
    params = p.get("params")
    if not isinstance(params, Mapping):
        errors.append("params_missing")
        return errors
    cell = params.get("cell")
    if not (
        isinstance(cell, list)
        and len(cell) == 2
        and all(_finite(x) for x in cell)
        and 0.0 < float(cell[0]) < float(cell[1]) < 1.0
    ):
        errors.append("params_cell_not_adjacent_unit_pair")
    p0 = _num(params.get("p0_deep_share"))
    if p0 is None or not (0.0 < p0 < 1.0):
        errors.append("params_p0_deep_share_out_of_unit_interval")
    elif isinstance(cell, list) and len(cell) == 2 and all(_finite(x) for x in cell):
        # nested-quantile identity: p0 must equal tau_lo / tau_hi exactly
        expected = float(cell[0]) / float(cell[1])
        if not np.isclose(p0, expected, rtol=1e-9, atol=1e-12):
            errors.append("p0_deep_share_not_tau_ratio")
    alpha = _num(params.get("alpha"))
    if alpha is None or not (0.0 < alpha < 1.0):
        errors.append("params_alpha_out_of_unit_interval")
    grid = params.get("alt_grid")
    if not (isinstance(grid, list) and grid and all(_finite(x) and float(x) > 0.0 for x in grid)):
        errors.append("params_alt_grid_not_positive_list")
    for field in ("n_train", "n_eval"):
        v = params.get(field)
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            errors.append(f"params_{field}_not_positive_int")
    return errors


def _lane_power_errors(p: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not _is_sha256(p.get("inputs_sha256")):
        errors.append("inputs_sha256_invalid")
    params = p.get("params")
    if not isinstance(params, Mapping):
        errors.append("params_missing")
        return errors
    lanes = params.get("lanes")
    if not (isinstance(lanes, list) and lanes and all(isinstance(x, str) for x in lanes)):
        errors.append("params_lanes_not_str_list")
    defects = params.get("defects")
    if not (
        isinstance(defects, list)
        and defects
        and all(_finite(x) for x in defects)
        and 0.0 in [float(x) for x in defects]
    ):
        errors.append("params_defects_must_include_zero")
    for field in ("n_steps", "n_seeds"):
        v = params.get(field)
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            errors.append(f"params_{field}_not_positive_int")
    alpha = _num(params.get("alpha"))
    if alpha is None or not (0.0 < alpha < 1.0):
        errors.append("params_alpha_out_of_unit_interval")
    n_ok = p.get("n_lanes_ok")
    if not isinstance(n_ok, int) or isinstance(n_ok, bool) or n_ok < 0:
        errors.append("n_lanes_ok_not_nonnegative_int")
    elif isinstance(lanes, list) and n_ok > len(lanes):
        errors.append("n_lanes_ok_exceeds_lanes")
    rates = p.get("null_alarm_rate")
    if not isinstance(rates, Mapping):
        errors.append("null_alarm_rate_not_mapping")
    else:
        for lane, rate in rates.items():
            if isinstance(lanes, list) and lane not in lanes:
                errors.append(f"null_alarm_rate_unknown_lane:{lane}")
            rv = _num(rate)
            if rv is None or not (0.0 <= rv <= 1.0):
                errors.append(f"null_alarm_rate_out_of_bounds:{lane}")
    return errors


def _honest_verdict_errors(p: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    run = p.get("run")
    run_params = run.get("params") if isinstance(run, Mapping) else None
    data_labels = run_params.get("data_labels") if isinstance(run_params, Mapping) else None
    if isinstance(data_labels, Mapping) and data_labels:
        distinct = set(data_labels.values())
        if len(distinct) != 1 or p.get("data_label") != distinct.pop():
            errors.append("data_label_mismatches_shard_labels")
    elif p.get("data_label") != "SYNTHETIC":
        errors.append("data_label_not_synthetic")
    if p.get("research_only") is not True:
        errors.append("research_only_not_true")
    if p.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    if not _is_sha256(p.get("inputs_sha256")):
        errors.append("inputs_sha256_invalid")
    verdict = p.get("verdict")
    if verdict not in (
        "confirmed",
        "supported_with_caveats",
        "not_supported",
        "inconclusive",
    ):
        errors.append("verdict_not_in_enum")
    alpha = _num(p.get("alpha"))
    if alpha is None or not (0.0 < alpha < 1.0):
        errors.append("alpha_out_of_unit_interval")
    for field in ("n_obs", "n_heads"):
        v = p.get(field)
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            errors.append(f"{field}_not_positive_int")
    winner = p.get("winner")
    if not isinstance(winner, str) or not winner.strip():
        errors.append("winner_not_nonempty_str")
    components = p.get("components")
    if not isinstance(components, Mapping):
        errors.append("components_not_mapping")
        components = {}
    unavailable = p.get("unavailable_lanes")
    if not isinstance(unavailable, list) or not all(isinstance(lane, str) for lane in unavailable):
        errors.append("unavailable_lanes_not_str_list")
        unavailable = []
    for lane in unavailable:
        if lane not in components:
            errors.append(f"unavailable_lane_unknown:{lane}")
    core_missing = {"winner_curse", "promotion", "drift"} & set(unavailable)
    if core_missing and verdict != "inconclusive":
        errors.append("core_lane_missing_but_verdict_not_inconclusive")
    promotion_detail = components.get("promotion")
    if verdict == "confirmed" and (
        not isinstance(promotion_detail, Mapping) or promotion_detail.get("promoted") is not True
    ):
        errors.append("confirmed_without_promotion_flag")
    return errors


_MONITOR_LANES = frozenset({"coverage", "tail", "calibration", "conformal", "drift", "emerge"})


def _monitor_run_errors(p: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    params = p.get("params")
    data_labels = params.get("data_labels") if isinstance(params, Mapping) else None
    if isinstance(data_labels, Mapping) and data_labels:
        distinct = set(data_labels.values())
        if len(distinct) != 1 or p.get("data_label") != distinct.pop():
            errors.append("data_label_mismatches_shard_labels")
    elif p.get("data_label") != "SYNTHETIC":
        errors.append("data_label_not_synthetic")
    if p.get("research_only") is not True:
        errors.append("research_only_not_true")
    if p.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    if not _is_sha256(p.get("inputs_sha256")):
        errors.append("inputs_sha256_invalid")
    n_rows = p.get("n_rows")
    if not isinstance(n_rows, int) or isinstance(n_rows, bool) or n_rows < 1:
        errors.append("n_rows_not_positive_int")
        n_rows = None
    n_alarms = p.get("n_alarm_rows")
    if not isinstance(n_alarms, int) or isinstance(n_alarms, bool) or n_alarms < 0:
        errors.append("n_alarm_rows_not_nonnegative_int")
    elif n_rows is not None and n_alarms > n_rows:
        errors.append("n_alarm_rows_exceeds_n_rows")
    lanes = p.get("lanes_available")
    if not isinstance(lanes, Mapping):
        errors.append("lanes_available_not_mapping")
    else:
        for lane in lanes:
            if lane not in _MONITOR_LANES:
                errors.append(f"lanes_available_unknown:{lane}")
        for lane, flag in lanes.items():
            if not isinstance(flag, bool):
                errors.append(f"lane_flag_not_bool:{lane}")
    params = p.get("params")
    if not isinstance(params, Mapping):
        errors.append("params_not_mapping")
    else:
        alpha = _num(params.get("alpha"))
        level = _num(params.get("level"))
        if alpha is None or not (0.0 < alpha < 1.0):
            errors.append("params.alpha_out_of_unit_interval")
        if level is None or not (0.0 < level < 1.0):
            errors.append("params.level_out_of_unit_interval")
        cell = params.get("tail_cell")
        if (
            not isinstance(cell, list)
            or len(cell) != 2
            or not all(_finite(x) for x in cell)
            or not (0.0 < cell[0] < cell[1] < 1.0)
        ):
            errors.append("tail_cell_not_ordered_pair")
        for field in ("n_train", "n_eval", "seed"):
            v = params.get(field)
            if not isinstance(v, int) or isinstance(v, bool) or (field != "seed" and v < 1):
                errors.append(f"params.{field}_bad")
        for field in ("heads", "shards"):
            v = params.get(field)
            if not isinstance(v, list) or not v or not all(isinstance(x, str) for x in v):
                errors.append(f"params.{field}_not_str_list")
    return errors


def _suite_health_errors(p: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    params_sh = p.get("params")
    input_labels = params_sh.get("input_labels") if isinstance(params_sh, Mapping) else None
    if isinstance(input_labels, Mapping) and input_labels:
        distinct_in = set(input_labels.values())
        expected = next(iter(distinct_in)) if len(distinct_in) == 1 else "MIXED"
        if p.get("data_label") != expected:
            errors.append("data_label_mismatches_input_labels")
    elif p.get("data_label") != "SYNTHETIC":
        errors.append("data_label_not_synthetic")
    if p.get("research_only") is not True:
        errors.append("research_only_not_true")
    if p.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    if not _is_sha256(p.get("inputs_sha256")):
        errors.append("inputs_sha256_invalid")
    params = p.get("params")
    if not isinstance(params, Mapping):
        errors.append("params_missing")
        return errors
    if not isinstance(params.get("receipts_dir"), str):
        errors.append("params_receipts_dir_not_str")
    alpha = _num(params.get("alpha"))
    if alpha is None or not (0.0 < alpha < 1.0):
        errors.append("params_alpha_out_of_range")
    n_files = params.get("n_files")
    if not isinstance(n_files, int) or isinstance(n_files, bool) or n_files < 1:
        errors.append("params_n_files_not_positive_int")
        n_files = None
    n_receipts = p.get("n_receipts")
    n_ok = p.get("n_ok")
    n_failed = p.get("n_failed")
    for name, v in (("n_receipts", n_receipts), ("n_ok", n_ok), ("n_failed", n_failed)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            errors.append(f"{name}_not_nonnegative_int")
    if (
        isinstance(n_receipts, int)
        and isinstance(n_ok, int)
        and isinstance(n_failed, int)
        and not (
            isinstance(n_ok, bool) or isinstance(n_failed, bool) or isinstance(n_receipts, bool)
        )
    ):
        if n_ok + n_failed != n_receipts:
            errors.append("ok_plus_failed_neq_receipts")
        if n_files is not None and n_receipts != n_files:
            errors.append("n_receipts_neq_n_files")
    if not isinstance(p.get("corpus_lane_available"), bool):
        errors.append("corpus_lane_available_not_bool")
    epoch_state = p.get("corpus_epoch")
    if epoch_state is not None:
        if not isinstance(epoch_state, Mapping):
            errors.append("corpus_epoch_not_mapping")
        elif epoch_state.get("available") is True:
            for name in ("errors",):
                if not isinstance(epoch_state.get(name), list):
                    errors.append(f"corpus_epoch_{name}_not_list")
            if not isinstance(epoch_state.get("chain_ok"), bool):
                errors.append("corpus_epoch_chain_ok_not_bool")
            root_val = epoch_state.get("head_epoch_root")
            if root_val is not None and not _is_sha256(root_val):
                errors.append("corpus_epoch_head_epoch_root_invalid")
            if epoch_state.get("chain_ok") is True and any(
                e != "no_epoch_receipts" for e in epoch_state.get("errors") or []
            ):
                errors.append("corpus_epoch_chain_ok_lies")
    n_findings = p.get("n_findings_harvested")
    n_pooled = p.get("n_evalues_pooled")
    for name, v in (("n_findings_harvested", n_findings), ("n_evalues_pooled", n_pooled)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            errors.append(f"{name}_not_nonnegative_int")
    if (
        isinstance(n_pooled, int)
        and isinstance(n_findings, int)
        and not isinstance(n_pooled, bool)
        and not isinstance(n_findings, bool)
        and n_pooled > n_findings
    ):
        errors.append("n_evalues_pooled_exceeds_findings")
    pooled = p.get("pooled_evalue")
    if pooled is not None and not (_finite(pooled) and float(pooled) > 0.0):
        errors.append("pooled_evalue_not_positive_finite")
    # fail-closed invariant: any unverifiable input must withhold the pool
    if (
        isinstance(n_failed, int)
        and not isinstance(n_failed, bool)
        and n_failed > 0
        and pooled is not None
    ):
        errors.append("pooled_evalue_not_withheld_on_failures")
    pooled_alarmed = p.get("pooled_alarmed")
    if not isinstance(pooled_alarmed, bool):
        errors.append("pooled_alarmed_not_bool")
    elif (
        pooled_alarmed and pooled is not None and alpha is not None and float(pooled) < 1.0 / alpha
    ):
        errors.append("pooled_alarmed_below_threshold")
    elif pooled_alarmed and pooled is None:
        errors.append("pooled_alarmed_without_pooled_evalue")
    return errors


def _mcs_seq_errors(p: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if p.get("research_only") is not True:
        errors.append("research_only_not_true")
    if p.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    alpha = _num(p.get("alpha"))
    if alpha is None or not (0.0 < alpha < 1.0):
        errors.append("alpha_out_of_unit_interval")
    lam = _num(p.get("lam"))
    if lam is None or not (0.0 < lam < 1.0):
        errors.append("lam_out_of_unit_interval")
    n_heads = p.get("n_heads")
    if not isinstance(n_heads, int) or isinstance(n_heads, bool) or n_heads < 2:
        errors.append("n_heads_lt_2")
        n_heads = None
    n_origins = p.get("n_origins")
    if not isinstance(n_origins, int) or isinstance(n_origins, bool) or n_origins < 1:
        errors.append("n_origins_not_positive_int")
        n_origins = None
    survivors = p.get("survivors")
    eliminated = p.get("eliminated")
    if not isinstance(survivors, list) or not all(isinstance(s, str) for s in survivors):
        errors.append("survivors_not_str_list")
        survivors = []
    if not isinstance(eliminated, Mapping):
        errors.append("eliminated_not_mapping")
        eliminated = {}
    else:
        overlap = set(survivors) & set(eliminated)
        if overlap:
            errors.append(f"survivor_and_eliminated_overlap:{sorted(overlap)}")
        for head, origin in eliminated.items():
            if not isinstance(head, str) or not isinstance(origin, int):
                errors.append("eliminated_entry_bad_shape")
                break
            if n_origins is not None and not (0 <= origin < n_origins):
                errors.append(f"eliminated_origin_out_of_range:{head}")
    if n_heads is not None and len(survivors) + len(eliminated) != n_heads:
        errors.append("survivor_eliminated_partition_mismatch")
    n_eliminated = p.get("n_eliminated")
    if n_eliminated != len(eliminated):
        errors.append("n_eliminated_mismatch")
    champion = p.get("champion")
    if champion is not None and champion not in survivors:
        errors.append("champion_not_in_survivors")
    if len(survivors) == 1 and champion != survivors[0]:
        errors.append("sole_survivor_not_champion")
    if len(survivors) != 1 and champion is not None:
        errors.append("champion_with_multiple_survivors")
    data_label = p.get("data_label")
    if not isinstance(data_label, str) or not data_label.strip():
        errors.append("data_label_not_nonempty_str")
    evidence = p.get("evidence")
    if not isinstance(evidence, list) or "anytime_valid" not in evidence:
        errors.append("evidence_missing_anytime_valid")
    return errors


def _serial_watch_errors(p: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if p.get("research_only") is not True:
        errors.append("research_only_not_true")
    if p.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    alpha = _num(p.get("alpha"))
    if alpha is None or not (0.0 < alpha < 1.0):
        errors.append("alpha_out_of_unit_interval")
    lam = _num(p.get("lam"))
    if lam is None or not (0.0 < lam < 1.0):
        errors.append("lam_out_of_unit_interval")
    n_lags = p.get("n_lags")
    if not isinstance(n_lags, int) or isinstance(n_lags, bool) or n_lags < 1:
        errors.append("n_lags_not_positive_int")
        n_lags = None
    n_origins = p.get("n_origins")
    if not isinstance(n_origins, int) or isinstance(n_origins, bool) or n_origins < 1:
        errors.append("n_origins_not_positive_int")
        n_origins = None
    per_lag = p.get("per_lag")
    if not isinstance(per_lag, Mapping) or not per_lag:
        errors.append("per_lag_not_mapping")
        per_lag = {}
    else:
        for key, entry in per_lag.items():
            try:
                k = int(key)
            except (TypeError, ValueError):
                errors.append(f"per_lag_key_not_int:{key}")
                continue
            if n_lags is not None and not (1 <= k <= n_lags):
                errors.append(f"per_lag_out_of_range:{key}")
            if not isinstance(entry, Mapping):
                errors.append(f"per_lag_entry_not_mapping:{key}")
                continue
            for side in ("pos", "neg"):
                v = _num(entry.get(side))
                if v is None or not (v > 0.0):
                    errors.append(f"per_lag_{side}_not_positive:{key}")
            if not isinstance(entry.get("alarmed"), bool):
                errors.append(f"per_lag_alarmed_not_bool:{key}")
    alarmed_lags = p.get("alarmed_lags")
    if not isinstance(alarmed_lags, list) or not all(
        isinstance(x, int) and not isinstance(x, bool) for x in alarmed_lags
    ):
        errors.append("alarmed_lags_not_int_list")
        alarmed_lags = []
    elif n_lags is not None:
        for k in alarmed_lags:
            if not (1 <= k <= n_lags):
                errors.append(f"alarmed_lag_out_of_range:{k}")
    alarm_origins = p.get("alarm_origins")
    if not isinstance(alarm_origins, Mapping):
        errors.append("alarm_origins_not_mapping")
    elif n_origins is not None:
        for key, origin in alarm_origins.items():
            if (
                not isinstance(key, str)
                or not key.startswith("lag")
                or not key.endswith(("_pos", "_neg"))
                or not isinstance(origin, int)
                or not (0 <= origin < n_origins)
            ):
                errors.append(f"alarm_origin_bad:{key}")
    for flag in ("any_lag_alarmed", "pooled_alarmed"):
        if not isinstance(p.get(flag), bool):
            errors.append(f"{flag}_not_bool")
    if (
        isinstance(alarmed_lags, list)
        and isinstance(p.get("any_lag_alarmed"), bool)
        and p["any_lag_alarmed"] != bool(alarmed_lags)
    ):
        errors.append("any_lag_alarmed_mismatch")
    pooled = _num(p.get("pooled_evalue"))
    if pooled is None or not (pooled > 0.0):
        errors.append("pooled_evalue_not_positive")
    elif (
        isinstance(p.get("pooled_alarmed"), bool)
        and p["pooled_alarmed"]
        and alpha is not None
        and pooled < 1.0 / alpha
    ):
        errors.append("pooled_alarmed_below_threshold")
    data_label = p.get("data_label")
    if not isinstance(data_label, str) or not data_label.strip():
        errors.append("data_label_not_nonempty_str")
    evidence = p.get("evidence")
    if not isinstance(evidence, list) or "anytime_valid" not in evidence:
        errors.append("evidence_missing_anytime_valid")
    return errors


def _emerge_drill_errors(p: Mapping[str, Any]) -> list[str]:
    """emerge_drill.v1: pooled-per-head e-values must re-derive the family claim."""
    errors: list[str] = []
    if p.get("research_only") is not True:
        errors.append("research_only_not_true")
    if p.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    if not isinstance(p.get("data_label"), str) or not p["data_label"].strip():
        errors.append("data_label_not_nonempty_str")
    pooled = p.get("pooled_per_head")
    if not isinstance(pooled, Mapping) or not pooled:
        errors.append("pooled_per_head_missing")
        return errors
    n_heads = p.get("n_heads")
    if not isinstance(n_heads, int) or isinstance(n_heads, bool) or n_heads != len(pooled):
        errors.append("n_heads_not_len_pooled")
    fam = _num(p.get("family_evalue"))
    if fam is None or fam <= 0:
        errors.append("family_evalue_not_positive")
    for head, row in pooled.items():
        if not isinstance(row, Mapping):
            errors.append(f"pooled_per_head:{head}:not_object")
            continue
        mean = _num(row.get("e_mean"))
        harm = _num(row.get("e_harmonic"))
        lanes = row.get("lanes")
        if mean is None or mean < 0:
            errors.append(f"pooled_per_head:{head}:e_mean_not_nonneg")
        if harm is None or harm < 0:
            errors.append(f"pooled_per_head:{head}:e_harmonic_not_nonneg")
        if not isinstance(lanes, Mapping) or not lanes:
            errors.append(f"pooled_per_head:{head}:lanes_missing")
        else:
            lane_vals = [_num(v) for v in lanes.values()]
            if any(v is None for v in lane_vals):
                errors.append(f"pooled_per_head:{head}:lane_evalue_not_numeric")
        # harmonic <= arithmetic mean (power-mean inequality)
        if mean is not None and harm is not None and harm - mean > 1e-9 * max(mean, 1.0):
            errors.append(f"pooled_per_head:{head}:harmonic_exceeds_mean")
    if (
        fam is not None
        and isinstance(n_heads, int)
        and not isinstance(n_heads, bool)
        and n_heads > 0
    ):
        # family e-value is the e-Bonferroni merger: n * min(e_mean)
        min_mean = min(
            (_num(r.get("e_mean")) or 0.0) for r in pooled.values() if isinstance(r, Mapping)
        )
        expected = min_mean * n_heads
        if abs(fam - expected) > 1e-6 * max(expected, 1.0):
            errors.append("family_evalue_not_bonferroni_min_mean")
    return errors


def _panel_audit_errors(p: Mapping[str, Any]) -> list[str]:
    """panel_audit.v1: per-head pooled e-values must re-derive the family verdict."""
    errors: list[str] = []
    if p.get("research_only") is not True:
        errors.append("research_only_not_true")
    if p.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    if not isinstance(p.get("data_label"), str) or not p["data_label"].strip():
        errors.append("data_label_not_nonempty_str")
    heads = p.get("heads")
    symbols = p.get("symbols")
    if not isinstance(heads, list) or not heads:
        errors.append("heads_missing")
    if not isinstance(symbols, list) or not symbols:
        errors.append("symbols_missing")
    n_heads, n_symbols = p.get("n_heads"), p.get("n_symbols")
    if isinstance(heads, list) and (
        not isinstance(n_heads, int) or isinstance(n_heads, bool) or n_heads != len(heads)
    ):
        errors.append("n_heads_not_len_heads")
    if isinstance(symbols, list) and (
        not isinstance(n_symbols, int) or isinstance(n_symbols, bool) or n_symbols != len(symbols)
    ):
        errors.append("n_symbols_not_len_symbols")
    cells_ok, cells_total = p.get("n_cells_ok"), p.get("n_cells_total")
    if not isinstance(cells_ok, int) or isinstance(cells_ok, bool) or cells_ok < 0:
        errors.append("n_cells_ok_not_nonneg_int")
    if not isinstance(cells_total, int) or isinstance(cells_total, bool) or cells_total < 0:
        errors.append("n_cells_total_not_nonneg_int")
    if (
        isinstance(cells_ok, int)
        and not isinstance(cells_ok, bool)
        and isinstance(cells_total, int)
        and not isinstance(cells_total, bool)
        and cells_ok > cells_total
    ):
        errors.append("n_cells_ok_exceeds_total")
    alpha = _num(p.get("alpha"))
    family = p.get("family")
    if alpha is None or not 0 < alpha < 1:
        errors.append("alpha_not_in_unit_interval")
    if not isinstance(family, Mapping):
        errors.append("family_missing")
    else:
        thr = _num(family.get("bonferroni_threshold"))
        mx = _num(family.get("max_head_pooled_evalue"))
        if thr is None:
            errors.append("bonferroni_threshold_missing")
        elif (
            alpha is not None
            and isinstance(n_heads, int)
            and not isinstance(n_heads, bool)
            and abs(thr - n_heads / alpha) > 1e-9 * max(thr, 1.0)
        ):
            errors.append("bonferroni_threshold_not_n_over_alpha")
        if isinstance(heads, list) and heads:
            pooled_vals = [
                _num(row.get("pooled_evalue")) for row in heads if isinstance(row, Mapping)
            ]
            if mx is None or any(v is None for v in pooled_vals):
                errors.append("max_head_pooled_evalue_missing")
            else:
                observed = max(v for v in pooled_vals if v is not None)
                if abs(mx - observed) > 1e-9 * max(mx, 1.0):
                    errors.append("max_head_pooled_not_max")
            alarmed = family.get("family_alarmed")
            if not isinstance(alarmed, bool):
                errors.append("family_alarmed_not_bool")
            elif isinstance(mx, float) and isinstance(thr, float) and alarmed != (mx >= thr):
                errors.append("family_alarmed_inconsistent")
    if not _is_sha256(p.get("inputs_sha256")):
        errors.append("inputs_sha256_invalid")
    return errors


def _asserted_honesty_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Reject asserted dishonesty in any evalue-family receipt.

    Early writers omitted the honesty stamps, so absence is tolerated (the
    committed corpus is immutable); an asserted bad value is not — a forged
    ``live_pnl_claim``/``research_only`` must fail even under a valid reseal.
    """
    errors: list[str] = []
    if receipt.get("live_pnl_claim") is True:
        errors.append("live_pnl_claim_true")
    if receipt.get("research_only") is False:
        errors.append("research_only_false")
    label = receipt.get("data_label")
    if label is not None and not (isinstance(label, str) and label.strip()):
        errors.append("data_label_not_nonempty_str")
    return errors


def evalue_family_contract_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Dispatch contract checks by ``kind``; empty list = structurally clean."""
    kind = receipt.get("kind") or receipt.get("schema")
    errors = _asserted_honesty_errors(receipt)
    if kind == "evalue_promotion.v1":
        return errors + _evalue_promotion_errors(receipt)
    if kind == "fleet_race.v1":
        return errors + _fleet_race_errors(receipt)
    if kind == "corpus_inference.v1":
        return errors + _corpus_inference_errors(receipt)
    if kind == "online_fdr.v1":
        return errors + _online_fdr_errors(receipt)
    if kind == "calibration_audit.v1":
        return errors + _calibration_audit_errors(receipt)
    if kind == "loss_cs.v1":
        return errors + _loss_cs_errors(receipt)
    if kind == "changepoint_localize.v1":
        return errors + _changepoint_localize_errors(receipt)
    if kind in ("coverage_audit", "coverage_audit.v1"):
        return errors + _coverage_audit_errors(receipt)
    if kind in ("coverage_cs", "coverage_cs.v1"):
        return errors + _coverage_cs_errors(receipt)
    if kind == "winner_curse.v1":
        return errors + _winner_curse_errors(receipt)
    if kind == "drift_alarm.v1":
        return errors + _drift_alarm_errors(receipt)
    if kind == "conformal_monitor.v1":
        return errors + _conformal_monitor_errors(receipt)
    if kind in ("tail_audit", "tail_audit.v1"):
        return errors + _tail_audit_errors(receipt)
    if kind in ("lane_power", "lane_power.v1"):
        return errors + _lane_power_errors(receipt)
    if kind == "honest_verdict.v1":
        return errors + _honest_verdict_errors(receipt)
    if kind in ("monitor_run", "monitor_run.v1"):
        return errors + _monitor_run_errors(receipt)
    if kind in ("suite_health", "suite_health.v1"):
        return errors + _suite_health_errors(receipt)
    if kind in ("mcs_seq", "mcs_seq.v1"):
        return errors + _mcs_seq_errors(receipt)
    if kind in ("serial_watch", "serial_watch.v1"):
        return errors + _serial_watch_errors(receipt)
    if kind == "emerge_drill.v1":
        return errors + _emerge_drill_errors(receipt)
    if kind == "panel_audit.v1":
        return errors + _panel_audit_errors(receipt)
    return errors
