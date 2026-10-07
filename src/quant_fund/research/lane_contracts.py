"""Contract checks for lane-specific receipt schemas.

``verify-receipt`` seals prove a receipt was not tampered with after writing;
these checks go further — they re-derive claims *inside* the payload so a
receipt whose numbers were fabricated before sealing still fails:

- ``capacity_overlay.v1``: ``days_to_trade == max_participation /
  participation_cap`` and ``feasible == (max_participation <= cap)`` per row;
  impact and trading days monotone in AUM within each book;
  ``n_rows == len(results)``.
- ``cross_sectional_rankic.v1``: ``n_rows == len(results)``, error rows carry
  a non-empty ``error`` while ok rows carry none, ``|mean_*| <= 1``, and —
  the statistical check — ``p_*`` re-derived from ``t_*`` as the two-sided
  Student-t tail on ``n_dates - 1`` (the writer's ``mean_tstat`` convention).
- ``fast_replay_p42_conformance``: honesty flags, verdict/evidence/gaps
  structure, ``base_commit`` 40-hex, ``code_sha256`` a path→sha256 map whose
  referenced files exist.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from scipy.stats import t as _t_dist

__all__ = ["SIM_LIVE_KINDS", "lane_contract_errors", "sim_live_contract_errors"]

_CAPACITY_SCHEMA = "capacity_overlay.v1"
_RANKIC_SCHEMA = "cross_sectional_rankic.v1"
_P42_RECEIPT = "fast_replay_p42_conformance"
SIM_LIVE_KINDS = ("sim_live_receipt", "sim_live_bench_receipt")

_T_REL_TOL = 1e-6


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _capacity_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return ["results_missing"]
    if payload.get("n_rows") != len(results):
        errors.append("n_rows")
    books: dict[str, list[tuple[float, float, float]]] = {}
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            errors.append(f"results[{index}]_not_object")
            continue
        book = str(row.get("book", "?"))
        cap = row.get("participation_cap")
        max_p = row.get("max_participation")
        days = row.get("days_to_trade")
        feasible = row.get("feasible")
        if not all(_finite(v) for v in (cap, max_p, days, row.get("impact_bps"), row.get("aum"))):
            errors.append(f"results[{index}]_non_finite")
            continue
        cap_f, max_f, days_f = (
            float(cast(float, cap)),
            float(cast(float, max_p)),
            float(cast(float, days)),
        )
        if cap_f <= 0:
            errors.append(f"results[{index}].participation_cap")
            continue
        # participation_cap is a fraction of ADV; trading days to deploy the
        # book is exactly the max participation divided by the daily cap.
        if not math.isclose(days_f, max_f / cap_f, rel_tol=1e-9, abs_tol=1e-12):
            errors.append(f"results[{index}].days_to_trade")
        if feasible not in (0, 1, True, False):
            errors.append(f"results[{index}].feasible_not_bool")
        elif int(bool(feasible)) != int(max_f <= cap_f + 1e-12):
            errors.append(f"results[{index}].feasible")
        books.setdefault(book, []).append((float(row["aum"]), float(row["impact_bps"]), days_f))
    for book, rows in books.items():
        rows.sort(key=lambda r: r[0])
        for earlier, later in zip(rows, rows[1:], strict=False):
            if later[1] < earlier[1] - 1e-9:
                errors.append(f"{book}:impact_not_monotone_in_aum")
            if later[2] < earlier[2] - 1e-9:
                errors.append(f"{book}:days_not_monotone_in_aum")
    return errors


def _rankic_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return ["results_missing"]
    if payload.get("n_rows") != len(results):
        errors.append("n_rows")
    n_error_rows = sum(
        1 for row in results if isinstance(row, dict) and row.get("status") == "error"
    )
    if payload.get("n_error_rows") is not None and payload["n_error_rows"] != n_error_rows:
        errors.append("n_error_rows")
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            errors.append(f"results[{index}]_not_object")
            continue
        status = row.get("status")
        if status == "error":
            if not str(row.get("error", "")).strip():
                errors.append(f"results[{index}].error_empty")
            continue
        if status != "ok":
            errors.append(f"results[{index}].status")
            continue
        if str(row.get("error", "")).strip():
            errors.append(f"results[{index}].error_set_on_ok")
        n_dates = row.get("n_dates")
        if not isinstance(n_dates, int) or n_dates < 3:
            errors.append(f"results[{index}].n_dates")
            continue
        for name in ("mean_pearson", "mean_spearman"):
            value = row.get(name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or abs(value) > 1.0 + 1e-9
            ):
                errors.append(f"results[{index}].{name}")
        # Re-derive two-sided p from the HAC t-stat (writer: mean_tstat,
        # df = n - 1). A fabricated p fails at 1e-6 relative tolerance.
        for t_key, p_key in (("t_spearman", "p_spearman"), ("t_pearson", "p_pearson")):
            t_value, p_value = row.get(t_key), row.get(p_key)
            if t_value is None or p_value is None:
                continue
            if not (_finite(t_value) and _finite(p_value)):
                errors.append(f"results[{index}].{t_key}_non_finite")
                continue
            expected = 2.0 * float(_t_dist.sf(abs(float(t_value)), df=n_dates - 1))
            if not math.isclose(float(p_value), expected, rel_tol=_T_REL_TOL):
                errors.append(f"results[{index}].{p_key}")
    return errors


def _p42_conformance_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("research_only") is not True:
        errors.append("research_only")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim")
    if not str(payload.get("verdict", "")).strip():
        errors.append("verdict")
    if "SYNTHETIC" not in str(payload.get("disclaimer", "")):
        errors.append("disclaimer_synthetic")
    base_commit = payload.get("base_commit")
    if not (
        isinstance(base_commit, str)
        and len(base_commit) == 40
        and all(c in "0123456789abcdef" for c in base_commit)
    ):
        errors.append("base_commit")
    for name in ("evidence", "fixes", "gaps_refused"):
        value = payload.get(name)
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            errors.append(name)
    code_map = payload.get("code_sha256")
    if not isinstance(code_map, dict) or not code_map:
        errors.append("code_sha256")
    else:
        for path, digest in code_map.items():
            if not (
                isinstance(digest, str)
                and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest)
            ):
                errors.append(f"code_sha256:{path}_digest")
                continue
            # A sealed digest of a path that does not exist attests nothing.
            if not (Path(__file__).resolve().parents[3] / path).is_file():
                errors.append(f"code_sha256:{path}_missing_file")
    return errors


def sim_live_contract_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Fail-closed honesty-contract checks on a ``sim_live`` receipt body.

    Everything re-derivable from the sealed body alone: the kind tag plus the
    honesty flags — a receipt claiming live PnL or dropping the
    simulation-only markers fails verification even when the seal was
    recomputed honestly.
    """
    errors: list[str] = []
    if receipt.get("kind") not in SIM_LIVE_KINDS:
        errors.append("kind")
    if receipt.get("research_only") is not True:
        errors.append("research_only")
    if receipt.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim")
    if receipt.get("simulated_only") is not True:
        errors.append("simulated_only")
    return errors


#: Tape/bench measurement lanes commit receipts with a narrative ``claim``
#: string, a probe-map ``claim`` dict, or a named ``claims`` dict. The generic
#: contract re-derives every internal-consistency field the shape exposes and
#: pins the honesty envelope: revision, data label, research-only, no live
#: claim. (The seal + forbidden-metric scan live in receipt_v2 itself.)
_MEASURE_SCHEMAS = frozenset(
    {
        "abc_calibrate.v1",
        "cancel_cluster.v1",
        "deep_microprice.v1",
        "depth_consumption.v1",
        "event_burst.v1",
        "event_granger.v1",
        "event_matrix.v1",
        "exec_cost_real.v1",
        "exec_cost_split.v1",
        "forecast_pipeline_audit.v1",
        "glft_bench.v1",
        "hawkes_mv.v1",
        "hawkes_real.v1",
        "hidden_depth.v1",
        "impact_instant.v1",
        "intraday_shape.v1",
        "lob_exec.v1",
        "lob_resilience.v1",
        "marketable_limit.v1",
        "metaorder_detect.v1",
        "mid_jump.v1",
        "order_lifetime.v1",
        "order_revision.v1",
        "post_trade_drift.v1",
        "price_clustering.v1",
        "price_improvement.v1",
        "propagator_real.v1",
        "quote_place.v1",
        "round_lot.v1",
        "sign_autocorr_real.v1",
        "sign_predict.v1",
        "sigkernel_mmd.v1",
        "sim_real_ledger.v1",
        "split_flow.v1",
        "spread_dynamics.v1",
        "spread_response.v1",
        "stale_quote.v1",
        "streak_stats.v1",
        "tape_digest.v1",
        "tick_rule.v1",
        "vol_signature.v1",
        "vpin.v1",
    }
)

_HEX = frozenset("0123456789abcdef")


def _measurement_claim_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """Generic deep check for measurement-lane receipts.

    Re-derives whatever consistency the claim shape exposes: a dict claim's
    ``n_probes``/``n_passed``/``ok`` must match its ``results`` map, string
    claims must be non-empty, ``claims`` maps must be non-empty; and the
    honesty envelope (kind, 7-40-hex revision, data label enum,
    research_only, live_pnl_claim not true) is pinned for every shape.
    """
    errors: list[str] = []
    if not isinstance(payload.get("kind"), str) or not payload.get("kind"):
        errors.append("kind_missing")
    rev = payload.get("git_revision")
    if (
        not isinstance(rev, str)
        or not 7 <= len(rev) <= 40
        or any(ch not in _HEX for ch in rev.lower())
    ):
        errors.append("git_revision_not_hex")
    if payload.get("data_label") not in {"SYNTHETIC", "MIXED", "REAL"}:
        errors.append("data_label_invalid")
    if payload.get("research_only") is not True:
        errors.append("research_only_not_true")
    if payload.get("live_pnl_claim") is True:
        errors.append("live_pnl_claim_true")
    claim = payload.get("claim")
    if isinstance(claim, str):
        if not claim.strip():
            errors.append("claim_empty")
    elif isinstance(claim, dict):
        results = claim.get("results")
        if isinstance(results, dict) and all(isinstance(v, bool) for v in results.values()):
            if claim.get("n_probes") != len(results):
                errors.append("n_probes_mismatch")
            if claim.get("n_passed") != sum(1 for v in results.values() if v):
                errors.append("n_passed_mismatch")
            if claim.get("ok") is not None and claim.get("ok") != all(results.values()):
                errors.append("ok_mismatch")
    elif claim is not None:
        errors.append("claim_unexpected_type")
    claims = payload.get("claims")
    if claims is not None and not isinstance(claims, dict):
        errors.append("claims_not_mapping")
    elif isinstance(claims, dict) and not claims:
        errors.append("claims_empty")
    interp = payload.get("interpretation")
    if interp is not None and (not isinstance(interp, str) or not interp.strip()):
        errors.append("interpretation_empty")
    return errors


def lane_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """Deep-verify a committed lane receipt; ``[]`` when the schema is unknown."""
    schema = payload.get("schema")
    if schema == _CAPACITY_SCHEMA:
        return _capacity_contract_errors(payload)
    if schema == _RANKIC_SCHEMA:
        return _rankic_contract_errors(payload)
    if payload.get("receipt") == _P42_RECEIPT:
        return _p42_conformance_contract_errors(payload)
    if payload.get("kind") in SIM_LIVE_KINDS:
        return sim_live_contract_errors(payload)
    if schema == "corpus_epoch.v1" or payload.get("kind") == "corpus_epoch.v1":
        from quant_fund.research.corpus_epoch import epoch_contract_errors

        return epoch_contract_errors(payload)
    if schema == "corpus_proof.v1" or payload.get("kind") == "corpus_proof.v1":
        from quant_fund.research.epoch_merkle import corpus_proof_errors

        return corpus_proof_errors(payload)
    if schema == "corpus_absence.v1" or payload.get("kind") == "corpus_absence.v1":
        from quant_fund.research.epoch_merkle import corpus_absence_errors

        return corpus_absence_errors(payload)
    if schema == "corpus_history_absence.v1" or payload.get("kind") == "corpus_history_absence.v1":
        from quant_fund.research.epoch_merkle import history_absence_errors

        return history_absence_errors(payload)
    if schema == "repo_integrity.v1" or payload.get("kind") == "repo_integrity":
        from quant_fund.research.repo_integrity import repo_integrity_contract_errors

        return repo_integrity_contract_errors(payload)
    if schema == "epoch_consistency.v1" or payload.get("kind") == "epoch_consistency.v1":
        from quant_fund.research.epoch_consistency import consistency_contract_errors

        return consistency_contract_errors(payload)
    if schema == "epoch_delta.v1" or payload.get("kind") == "epoch_delta.v1":
        from quant_fund.research.epoch_delta import epoch_delta_errors

        return epoch_delta_errors(payload)
    if schema == "epoch_position.v1" or payload.get("kind") == "epoch_position.v1":
        from quant_fund.research.epoch_merkle import epoch_position_errors

        return epoch_position_errors(payload)
    if schema == "integrity_checkpoint.v1" or payload.get("kind") == "integrity_checkpoint":
        from quant_fund.research.integrity_checkpoint import checkpoint_contract_errors

        return checkpoint_contract_errors(payload)
    if schema == "integrity_witness.v1" or payload.get("kind") == "integrity_witness":
        from quant_fund.research.integrity_witness import witness_contract_errors

        return witness_contract_errors(payload)
    if schema == "auditor_bundle.v1" or payload.get("kind") == "auditor_bundle":
        from quant_fund.research.auditor_bundle import bundle_contract_errors

        return bundle_contract_errors(payload)
    if schema == "receipt_lattice.v1" or payload.get("kind") == "receipt_lattice.v1":
        from quant_fund.research.receipt_lattice import lattice_contract_errors

        return lattice_contract_errors(payload)
    if schema == "receipt_graph.v1" or payload.get("kind") == "receipt_graph.v1":
        from quant_fund.research.receipt_graph import graph_contract_errors

        return graph_contract_errors(payload)
    if payload.get("kind") in ("xwatch", "xwatch.v1"):
        from quant_fund.research.xwatch import xwatch_contract_errors

        return xwatch_contract_errors(payload)
    if schema == "tamper_drill.v1" or payload.get("kind") == "tamper_drill":
        from quant_fund.research.tamper_drill import drill_contract_errors

        return drill_contract_errors(payload)
    if schema == "checkpoint_chain.v1" or payload.get("kind") == "checkpoint_chain":
        from quant_fund.research.checkpoint_chain import chain_contract_errors

        return chain_contract_errors(payload)
    if schema in ("fuzz_drill.v1", "receipt_fuzz.v1") or payload.get("kind") in (
        "fuzz_drill",
        "receipt_fuzz",
    ):
        from quant_fund.research.fuzz_drill import fuzz_contract_errors

        return fuzz_contract_errors(payload)
    if schema == "receipt_tombstone.v1" or payload.get("kind") == "receipt_tombstone.v1":
        from quant_fund.research.receipt_tombstone import tombstone_contract_errors

        return tombstone_contract_errors(payload)
    if schema == "rough_vol.v1":
        from quant_fund.models.rbergomi import rough_vol_contract_errors

        return rough_vol_contract_errors(payload)
    if schema == "fbm_circulant.v1":
        from quant_fund.models.fbm import fbm_contract_errors

        return fbm_contract_errors(payload)
    if schema == "vol_of_vol.v1":
        from quant_fund.research.vol_of_vol import vol_of_vol_contract_errors

        return vol_of_vol_contract_errors(payload)
    if schema == "replay_coverage.v1" or payload.get("kind") in (
        "replay_coverage",
        "replay_coverage.v1",
    ):
        from quant_fund.research.replay_sweep import replay_coverage_contract_errors

        return replay_coverage_contract_errors(payload)
    if schema in _MEASURE_SCHEMAS:
        return _measurement_claim_contract_errors(payload)
    return []
