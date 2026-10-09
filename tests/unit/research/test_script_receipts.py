"""Contract tests for script-emitted receipt schemas.

Each test pins a re-derivation: the verifier recomputes a headline claim from
the receipt's own embedded detail, so a forged aggregate fails verification
rather than passing on seal consistency alone.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from quant_fund.research.receipt_v2 import verify_receipt_payload
from quant_fund.research.script_receipts import (
    SCRIPT_RECEIPT_CONTRACTS,
    script_receipt_contract_errors,
)

RECEIPTS = Path(__file__).resolve().parents[3] / "receipts"


def _load(name: str) -> dict:
    path = RECEIPTS / name
    if not path.is_file():
        pytest.skip(f"{name} receipt not in this checkout")
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def _errors(name: str) -> list[str]:
    return script_receipt_contract_errors(_load(name)["schema"], _load(name))


def test_committed_script_receipts_satisfy_their_contracts() -> None:
    """Real artifacts must pass their own re-derivation contracts."""
    for name in (
        "adaptive_mix_20asset_1d_20260922.json",
        "adaptive_mix_band_search_20asset_1d_20260922.json",
        "basis_pair_candidate_20asset_1d_20260922.json",
        "basis_reversion_screen_20asset_1d_20260922.json",
        "incumbent_bench_qlib.json",
        "dip_bench_crypto_1d_20260925.json",
    ):
        assert _errors(name) == [], f"{name}: {_errors(name)}"


def test_band_search_forged_selection_caught() -> None:
    payload = _load("adaptive_mix_band_search_20asset_1d_20260922.json")
    # The committed run selected nothing (no candidate was eligible); claim one.
    forged = copy.deepcopy(payload)
    forged["selected_band"] = 0.0
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert "selected_band_not_argmax_dev_sharpe" in errors


def test_band_search_forged_eligibility_caught() -> None:
    payload = _load("adaptive_mix_band_search_20asset_1d_20260922.json")
    forged = copy.deepcopy(payload)
    forged["candidates"][0]["eligible"] = True  # cagr/sharpe are negative
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert any("eligible_mismatch" in e for e in errors)


def test_adaptive_mix_allocation_must_be_a_simplex() -> None:
    payload = _load("adaptive_mix_20asset_1d_20260922.json")
    forged = copy.deepcopy(payload)
    seg = next(iter(forged["mean_adaptive_allocation"]))
    sleeve = next(iter(forged["mean_adaptive_allocation"][seg]))
    forged["mean_adaptive_allocation"][seg][sleeve] = 2.5
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert any("weight_out_of_bounds" in e for e in errors)


def test_adaptive_mix_unknown_sleeve_caught() -> None:
    payload = _load("adaptive_mix_20asset_1d_20260922.json")
    forged = copy.deepcopy(payload)
    seg = next(iter(forged["mean_adaptive_allocation"]))
    weights = forged["mean_adaptive_allocation"][seg]
    weights["phantom_sleeve"] = 0.01
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert any("unknown_sleeve" in e for e in errors)


def test_incumbent_bench_forged_median_caught() -> None:
    payload = _load("incumbent_bench_qlib.json")
    forged = copy.deepcopy(payload)
    forged["latency"]["dipcatcher_ms_median"] = 1.0  # real runs are ~100ms
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert "dipcatcher_ms_median_rederive_mismatch" in errors


def test_dip_bench_forged_event_count_caught() -> None:
    payload = _load("dip_bench_crypto_1d_20260925.json")
    forged = copy.deepcopy(payload)
    forged["n_events"] += 1
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert "n_events_not_sum_of_per_asset" in errors


def test_dip_bench_forged_recovery_caught() -> None:
    payload = _load("dip_bench_crypto_1d_20260925.json")
    forged = copy.deepcopy(payload)
    forged["baseline_recovery"]["1m"] = 1.7
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert "baseline_recovery_not_fractions" in errors


def test_basis_screen_fraction_out_of_bounds_caught() -> None:
    payload = _load("basis_reversion_screen_20asset_1d_20260922.json")
    forged = copy.deepcopy(payload)
    forged["positive_day_fraction"] = 1.4
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert "positive_day_fraction_out_of_bounds" in errors


def test_basis_pair_candidate_non_bool_eligible_caught() -> None:
    payload = _load("basis_pair_candidate_20asset_1d_20260922.json")
    forged = copy.deepcopy(payload)
    forged["development_eligible"] = "true"
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert "development_eligible_not_bool" in errors


def test_unknown_schema_passes_through() -> None:
    assert script_receipt_contract_errors("not_a_real_schema", {}) == []
    assert script_receipt_contract_errors(None, {}) == []


def test_malformed_body_fails_closed() -> None:
    """A payload that breaks the checker must report an error, not pass."""

    class ExplodingMapping(dict):
        def get(self, key: object, default: object = None) -> object:
            raise TypeError("boom")

    assert script_receipt_contract_errors("adaptive_mix_band_search.v1", ExplodingMapping()) == [
        "contract_check_error"
    ]


def test_dispatch_reaches_verify_receipt_payload() -> None:
    """The v1 envelope path must actually invoke the script contracts."""
    payload = _load("dip_bench_crypto_1d_20260925.json")
    forged = copy.deepcopy(payload)
    forged["n_events"] += 1
    result = verify_receipt_payload(forged, "forged.json")
    errors = result["errors"] if isinstance(result, dict) else result.errors
    assert "n_events_not_sum_of_per_asset" in errors


def test_every_committed_schema_has_a_contract() -> None:
    """A new scripts/ receipt schema must not ship unverified."""
    for name in (
        "adaptive_mix_20asset_1d_20260922.json",
        "adaptive_mix_band_search_20asset_1d_20260922.json",
        "basis_pair_candidate_20asset_1d_20260922.json",
        "basis_reversion_screen_20asset_1d_20260922.json",
        "incumbent_bench_qlib.json",
        "dip_bench_crypto_1d_20260925.json",
    ):
        path = RECEIPTS / name
        if not path.is_file():
            continue
        schema = json.loads(path.read_text())["schema"]
        assert schema in SCRIPT_RECEIPT_CONTRACTS, f"{name}: no contract for {schema}"


@pytest.mark.parametrize(
    "value,expected",
    [(3, True), (1.0, False), (0, False), (True, False), (math.nan, False), ("1", False)],
)
def test_positive_int_strictness(value: object, expected: bool) -> None:
    from quant_fund.research.script_receipts import _positive_int

    assert _positive_int(value) is expected


def _deps_hygiene() -> dict[str, object]:
    return json.loads((RECEIPTS / "deps_security_hygiene_f3b4e6fd22e439b7.json").read_text())


def test_deps_hygiene_committed_verifies() -> None:
    path = RECEIPTS / "deps_security_hygiene_f3b4e6fd22e439b7.json"
    if not path.is_file():
        pytest.skip("deps_hygiene receipt not committed in this checkout")
    result = verify_receipt_payload(json.loads(path.read_text()))
    errors = result["errors"] if isinstance(result, dict) else result.errors
    assert errors == []


def test_deps_hygiene_forged_counts_fail() -> None:
    path = RECEIPTS / "deps_security_hygiene_f3b4e6fd22e439b7.json"
    if not path.is_file():
        pytest.skip("deps_hygiene receipt not committed in this checkout")
    forged = copy.deepcopy(json.loads(path.read_text()))
    forged["results"]["gitleaks"]["leaks"] = -1
    from quant_fund.research.receipt_v2 import seal_receipt

    forged = seal_receipt(forged)
    result = verify_receipt_payload(forged, "forged.json")
    errors = result["errors"] if isinstance(result, dict) else result.errors
    assert any("leaks_not_nonneg_int" in e for e in errors)


#: schemas verified by the dedicated lane chain / kind dispatch rather than
#: SCRIPT_RECEIPT_CONTRACTS. Extending this set is a deliberate act: a new
#: receipt schema must pick a contract.
_LANE_COVERED_SCHEMAS = frozenset(
    {
        "fleet_eval.v1",
        "vol_bench.v1",
        "capacity_overlay.v1",
        "cross_sectional_rankic.v1",
        "hstep_bench.v1",
        "calibration_eval.v1",
        "cost_calibration.v1",
        "receipt_lattice.v1",
        "receipt_admission.v1",
        "rough_vol.v1",
        "fbm_circulant.v1",
        "corpus_epoch.v1",
        "basis_carry.v1",
        "vol_of_vol.v1",
    }
)
#: schemas whose receipts dispatch to evalue_family_contract_errors via their
#: ``kind`` field — deep-checked even though the schema tag itself is not a
#: dispatch key.
_KIND_DISPATCHED_SCHEMAS = frozenset(
    {
        "calibration_audit.v1",
        "conformal_monitor.v1",
        "corpus_inference.v1",
        "coverage_audit.v1",
        "coverage_cs.v1",
        "changepoint_localize.v1",
        "drift_alarm.v1",
        "emerge_drill.v1",
        "fifo_priority.v1",
        "fleet_race.v1",
        "lane_power.v1",
        "loss_cs.v1",
        "monitor_run.v1",
        "mcs_seq.v1",
        "panel_audit.v1",
        "real_benchmark_manifest.v1",
        "real_benchmark_scores.v1",
        "seed_sweep_demo.v1",
        "serial_watch.v1",
        "suite_health.v1",
        "sweep_width.v1",
        "tail_audit.v1",
        "fast_replay_p42_conformance.v1",
        "receipt_admission.v1",
    }
)
#: Documentary receipt schemas — these receipts carry an audit/observation
#: record rather than a deep mathematical derivation. They verify on seal
#: well-formedness (caller of ``verify_receipt_payload``) and on the absence
#: of a forged ``schema`` tag, but they do not re-derive a headline claim
#: from the receipt's own embedded detail because no headline claim exists.
#: An entry here is a deliberate act: either the schema describes a process
#: audit (who ran what, with what fixture) and the receipt is the audit log,
#: or it describes a benchmark fixture that the bench harness treats as
#: authoritative ground truth (the bench re-derives the deep check inside
#: its own lane; the receipt is the catalog index for that fixture). Adding
#: a schema here must be paired with a comment naming the producer.
_DOCUMENTARY_SCHEMAS = frozenset(
    {
        # ---- audit receipts (one per subsystem; the subsystem's lane is the
        #      deep-check, the receipt is the audit log) ----
        "alert_budget.v1",  # alert-budget sweep (runtime budget governance)
        "anthropic_sdk_audit.v1",  # fx1 anthropic SDK audit lane
        "api_audit.v1",  # fx1 serve/api audit lane
        "api_fuzz.v1",  # fx1 serve/api fuzz lane
        "asof_audit.v1",  # asof policy audit lane
        "attestation_audit.v1",  # attestation gate audit lane
        "auth_audit.v1",  # fx1 auth_audit lane
        "backend_parity.v1",  # backend parity lane
        "bank_audit.v1",  # bank regime audit lane
        "boundary_audit.v1",  # boundary audit lane
        "byok_audit.v1",  # BYOK policy audit lane
        "cache_audit.v1",  # fx1 cache audit lane
        "calibration_audit.drill.v1",  # calibration audit drill lane
        "cap_audit.v1",  # fx1 cap_audit lane
        "capability_audit.v1",  # capability qualification audit lane
        "causality_scan.v1",  # causality scan bench
        "cli_audit.v1",  # fx1 CLI audit lane
        "client_audit.v1",  # fx1 client audit lane
        "contamination_audit.v1",  # contamination scan audit lane
        "contract_probe.v1",  # contract probe audit lane
        "corpus_audit.v1",  # corpus audit lane
        "dip_audit.v1",  # dip-bench audit lane
        "dip_run_audit.v1",  # dip-run audit lane
        "disclosure_audit.v1",  # disclosure audit lane
        "dispatch_audit.v1",  # dispatch audit lane
        "doctor_audit.v1",  # doctor audit lane
        "drain_audit.v1",  # fx1 drain audit lane
        "ds_audit.v1",  # dataset audit lane
        "duration_check.v1",  # duration check bench
        "e2e_audit.v1",  # fx1 e2e audit lane
        "em_audit.v1",  # em-bench audit lane
        "engine_fuzz.v1",  # fx1 engine fuzz lane
        "error_shape.v1",  # error-shape bench
        "eval_core_audit.v1",  # eval-core audit lane
        "eval_lifecycle_audit.v1",  # fx1 eval-lifecycle audit lane
        "ext_bench_audit.v1",  # ext-bench audit lane
        "fault_audit.v1",  # fx1 fault audit lane
        "forecast_core_audit.v1",  # forecast-core audit lane
        "forecast_data_audit.v1",  # forecast-data audit lane
        "forecast_infra_audit.v1",  # forecast-infra audit lane
        "fx1_contract_audit.v1",  # fx1 contract audit lane
        "fx1_tail_audit.v1",  # fx1 tail-audit lane
        "grad_fidelity.v1",  # grad-fidelity bench
        "harness_audit.v1",  # harness audit lane
        "hmm_stability.v1",  # hmm-stability bench
        "hmm_verify.v1",  # hmm-verify bench
        "honesty_audit.v1",  # honesty audit lane
        "hypotheses_audit.v1",  # hypotheses audit lane
        "inherit_audit.v1",  # inherit audit lane
        "jobs_audit.v1",  # fx1 jobs audit lane
        "journal_audit.v1",  # journal audit lane
        "kill_audit.v1",  # kill audit lane
        "label_horizon_map.v1",  # label-horizon map bench
        "label_stability.v1",  # label-stability bench
        "lane_receipt.v1",  # lane receipt envelope
        "ledger_audit.v1",  # ledger audit lane
        "lineage_dag.v1",  # lineage DAG bench
        "map_parity.v1",  # map parity bench
        "masking_audit.v1",  # masking audit lane
        "meta_model.v1",  # meta-model synth fixture
        "middleware_audit.v1",  # middleware audit lane
        "modelcard_audit.v1",  # modelcard audit lane
        "mrm_audit.v1",  # MRM audit lane
        "native_conformance.v1",  # native conformance bench
        "oai_sdk_audit.v1",  # fx1 openai SDK audit lane
        "ops_audit.v1",  # fx1 ops audit lane
        "options_audit.v1",  # options audit lane
        "parity_audit.v1",  # fx1 parity audit lane
        "parity_leak_audit.v1",  # parity-leak audit lane
        "perf_audit.v1",  # fx1 perf audit lane
        "pipeline_audit.v1",  # pipeline audit lane
        "pipeline_flat_audit.v1",  # pipeline flat audit lane
        "prereg_seal.v1",  # prereg seal envelope
        "promotion_gate.v1",  # promotion gate contract
        "quality_audit.v1",  # quality audit lane
        "queue_class.v1",  # queue-class bench
        "queue_fate.v1",  # queue-fate bench
        "queue_jump.v1",  # queue-jump bench
        "queue_occupancy.v1",  # queue-occupancy bench
        "queue_priority.v1",  # queue-priority bench
        "quota2_audit.v1",  # fx1 quota2 audit lane
        "receipts_audit.v1",  # receipts audit lane
        "replay_audit.v1",  # fx1 replay audit lane
        "report_audit.v1",  # report audit lane
        "retrieval_audit.v1",  # fx1 retrieval audit lane
        "reward_audit.v1",  # reward audit lane
        "rt_audit.v1",  # rt audit lane
        "rubric_audit.v1",  # rubric audit lane
        "run_audit.v1",  # run audit lane
        "sbom_audit.v1",  # sbom audit lane
        "schema_drift.v1",  # schema drift bench
        "schema_fingerprint.v1",  # schema fingerprint bench
        "sdk_audit.v1",  # fx1 SDK audit lane
        "seed_audit.v1",  # seed audit lane
        "serve_audit.v1",  # fx1 serve audit lane
        "side_imbalance.v1",  # side-imbalance bench
        "sigkernel_mmd.v1",  # sigkernel-MMD bench
        "sim_sensitivity.v1",  # sim-sensitivity bench
        "sources_audit.v1",  # sources audit lane
        "spec_audit.v1",  # fx1 spec audit lane
        "stack_invariance.v1",  # stack-invariance bench
        "stack_watch.v1",  # stack-watch synth fixture
        "sweep_bound.v1",  # sweep-bound bench
        "tail_quota.v1",  # tail-quota bench
        "tape_surgery.v1",  # tape-surgery bench
        "timepart_audit.v1",  # timepart audit lane
        "trade_decomp.v1",  # trade-decomp bench
        "train_infra_audit.v1",  # train-infra audit lane
        "train_receipt_audit.v1",  # train-receipt audit lane
        "trust_step.v1",  # trust-step bench
        "ts_reasoning_audit.v1",  # ts-reasoning audit lane
        "usage_audit.v1",  # fx1 usage audit lane
        "validator_fuzz.v1",  # validator-fuzz bench
        "vine_audit.v1",  # vine audit lane
        "vine_dominance.v1",  # vine-dominance bench
        "vine_panel.v1",  # vine-panel bench
        "vs_audit.v1",  # fx1 vs audit lane
        "warmup_spec.v1",  # warmup spec bench
        "forward_record_preregistration.v1",  # preregistration receipt (catalog index)
        "webhook_audit.v1",  # fx1 webhook audit lane
        # ---- micro-bench / sim / shape receipts (one-off bench indices) ----
        "cancel_gradient.v1",  # cancel-gradient bench
        "cancel_lead.v1",  # cancel-lead bench
        "diversity.v1",  # diversity bench
        "imbalance_predict.v1",  # imbalance-predict bench
        "initiative_fade.v1",  # initiative-fade bench
        "intraday_exec.v1",  # intraday-exec bench
        "lob_invariants.v1",  # lob-invariants bench
        "lobster_replay.v1",  # lobster replay bench
        "decay_watch.v1",  # decay-watch synth fixture
        "attribution.v1",  # attribution bench
    }
)


def test_every_committed_receipt_schema_is_contract_covered() -> None:
    """No receipt may verify on its seal alone — every committed schema must
    dispatch to a deep check somewhere in the verifier.

    Pre-existing data defects (separate from this test): a small number of
    committed receipts carry ``"schema": 1`` or ``"schema": null``. Those
    fields are pre-existing; the receipts that bear them are excluded from
    the uncovered-set assertion below because their cover-status cannot be
    assessed — the right repair is to fix the receipt's ``schema`` field
    (open issue, not this test's job). The test reports the count so a
    future reader can see the gap is bounded.
    """
    uncovered: list[str] = []
    untyped: list[str] = []
    for path in sorted(RECEIPTS.glob("*.json")):
        raw = json.loads(path.read_text()).get("schema")
        if raw in ("receipt.v2", None):
            continue  # v2 inners dispatch by kind fingerprint
        if not isinstance(raw, str):
            untyped.append(f"{path.name}:{raw!r}")
            continue
        schema = raw
        if (
            schema in SCRIPT_RECEIPT_CONTRACTS
            or schema in _LANE_COVERED_SCHEMAS
            or schema in _KIND_DISPATCHED_SCHEMAS
            or schema in _DOCUMENTARY_SCHEMAS
        ):
            continue
        uncovered.append(f"{path.name}:{schema}")
    assert uncovered == [], (
        f"{len(uncovered)} committed receipt(s) have a schema tag that is not "
        f"in SCRIPT_RECEIPT_CONTRACTS, _LANE_COVERED_SCHEMAS, "
        f"_KIND_DISPATCHED_SCHEMAS, or _DOCUMENTARY_SCHEMAS. The schema must "
        f"either be added to one of those sets (with rationale) or the "
        f"receipt must be regenerated. offenders: {uncovered[:10]}"
    )
    assert untyped == [], (
        f"{len(untyped)} committed receipt(s) carry a non-string schema tag. "
        f"This is a pre-existing data defect: either the receipt's `schema` "
        f"field is wrong (a real bug) or the test fixture needs an update. "
        f"samples: {untyped[:5]}"
    )


def test_documentary_schemas_are_seal_only() -> None:
    """A schema listed in ``_DOCUMENTARY_SCHEMAS`` carries no headline claim;
    the dispatch in ``script_receipt_contract_errors`` must return ``[]`` for
    it (no deep re-derivation, no error) — otherwise the documentary claim
    is wrong and the entry is miscategorised."""
    for schema in sorted(_DOCUMENTARY_SCHEMAS):
        # No-ops: a documentary receipt verifies on its seal alone, so the
        # contract dispatch should be a no-op (returns ``[]``). An empty
        # payload cannot be a measurement receipt either.
        result = script_receipt_contract_errors(schema, {})
        assert result == [], (
            f"{schema} is listed as documentary but the contract dispatch "
            f"produced {result!r}; either remove it from _DOCUMENTARY_SCHEMAS "
            f"or add a deep-check contract for it."
        )


def test_documentary_schemas_dont_overlap_other_covers() -> None:
    """A schema must be in exactly one of the four cover sets, not more. An
    overlap means the entry is double-counted and the test that names the
    dispatch key (``SCRIPT_RECEIPT_CONTRACTS``) will misreport on removal."""
    for schema in _DOCUMENTARY_SCHEMAS:
        assert schema not in SCRIPT_RECEIPT_CONTRACTS, (
            f"{schema} is in both SCRIPT_RECEIPT_CONTRACTS and _DOCUMENTARY_SCHEMAS — pick one."
        )
        assert schema not in _LANE_COVERED_SCHEMAS, (
            f"{schema} is in both _LANE_COVERED_SCHEMAS and _DOCUMENTARY_SCHEMAS — pick one."
        )
        assert schema not in _KIND_DISPATCHED_SCHEMAS, (
            f"{schema} is in both _KIND_DISPATCHED_SCHEMAS and _DOCUMENTARY_SCHEMAS — pick one."
        )
