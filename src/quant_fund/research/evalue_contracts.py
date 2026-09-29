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
    if p.get("data_label") != "SYNTHETIC":
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


def evalue_family_contract_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Dispatch contract checks by ``kind``; empty list = structurally clean."""
    kind = receipt.get("kind") or receipt.get("schema")
    if kind == "evalue_promotion.v1":
        return _evalue_promotion_errors(receipt)
    if kind == "fleet_race.v1":
        return _fleet_race_errors(receipt)
    if kind == "corpus_inference.v1":
        return _corpus_inference_errors(receipt)
    if kind == "online_fdr.v1":
        return _online_fdr_errors(receipt)
    if kind == "calibration_audit.v1":
        return _calibration_audit_errors(receipt)
    if kind == "loss_cs.v1":
        return _loss_cs_errors(receipt)
    if kind == "changepoint_localize.v1":
        return _changepoint_localize_errors(receipt)
    if kind == "coverage_audit.v1":
        return _coverage_audit_errors(receipt)
    if kind == "coverage_cs.v1":
        return _coverage_cs_errors(receipt)
    return []
