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
        "honest_verdict.v1",
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


_VERDICT_VALUES = frozenset(
    {"confirmed", "supported_with_caveats", "not_supported", "inconclusive"}
)
# Lanes whose absence forces ``inconclusive`` — honest_verdict only lets the
# core trio veto a claim; extension lanes may be unavailable without penalty.
_CORE_LANES = frozenset({"winner_curse", "promotion", "drift"})


def _honest_verdict_errors(p: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if p.get("data_label") != "SYNTHETIC":
        errors.append("data_label_not_synthetic")
    if p.get("research_only") is not True:
        errors.append("research_only_not_true")
    if p.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    if not _is_sha256(p.get("inputs_sha256")):
        errors.append("inputs_sha256_invalid")
    verdict = p.get("verdict")
    if verdict not in _VERDICT_VALUES:
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
    if not isinstance(unavailable, list):
        errors.append("unavailable_lanes_not_list")
        unavailable = []
    elif not all(isinstance(lane, str) for lane in unavailable):
        errors.append("unavailable_lanes_not_str_list")
    # unavailable lanes must be named components — a fabricated lane name
    # means the claim-evidence split itself is forged
    for lane in unavailable:
        if isinstance(components, Mapping) and lane not in components:
            errors.append(f"unavailable_lane_unknown:{lane}")
    # veto rule: a missing core lane forces inconclusive — nothing else may
    # mask an unverifiable claim as supported
    if any(lane in _CORE_LANES for lane in unavailable) and verdict != "inconclusive":
        errors.append("core_lane_missing_but_verdict_not_inconclusive")
    # promotion must have run for anything other than inconclusive
    if verdict != "inconclusive" and "promotion" in unavailable:
        errors.append("verdict_decisive_without_promotion")
    # confirmed requires a promoted flag actually recorded
    promotion_detail = components.get("promotion")
    if verdict == "confirmed" and (
        not isinstance(promotion_detail, Mapping) or promotion_detail.get("promoted") is not True
    ):
        errors.append("confirmed_without_promotion_flag")
    return errors


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
    if kind == "honest_verdict.v1":
        return _honest_verdict_errors(receipt)
    return []
