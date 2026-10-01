import json
from pathlib import Path

import pytest

from quant_fund.research.catalog import (
    BENCHMARK_CATALOG_VERSION,
    BENCHMARK_FAMILY_ORDER,
    DIST_CRPS_EPROCESS_REQUIRED_WHEN_DM,
    DIST_CRPS_SCALED_EPROCESS_REQUIRED_WHEN_DM,
    H1_HYPOTHESIS_ID,
    H2_HYPOTHESIS_ID,
    H3_HYPOTHESIS_ID,
    H4_HYPOTHESIS_ID,
    H4B_HYPOTHESIS_ID,
    H7_HYPOTHESIS_ID,
    H8_HYPOTHESIS_ID,
    H9_HYPOTHESIS_ID,
    H10_HYPOTHESIS_ID,
    H11_HYPOTHESIS_ID,
    H12_HYPOTHESIS_ID,
    H15_HYPOTHESIS_ID,
    H16_HYPOTHESIS_ID,
    H17_HYPOTHESIS_ID,
    H18_HYPOTHESIS_ID,
    H19_HYPOTHESIS_ID,
    H20_HYPOTHESIS_ID,
    H21_HYPOTHESIS_ID,
    H22_HYPOTHESIS_ID,
    RESEARCH_RECEIPT_SCHEMA_VERSION,
    TAIL_ES_BATTERY_REQUIRED_WHEN_ES_MARKERS,
    TAIL_VAR_BATTERY_REQUIRED_WHEN_KUPIEC,
    aci_has_finite_kupiec_p,
    conformal_aci_blob,
    conformal_mondrian_aci_blob,
    conformal_mondrian_aci_payload,
    conformal_rank_has_finite_fdr,
    coverage_guarantee_scope_consistency_errors,
    coverage_guarantee_scope_is_marginal,
    crc_has_finite_kupiec_p,
    cv_plus_has_finite_coverage_and_floor,
    dist_crps_eprocess_keys_present,
    dist_crps_eprocess_missing_keys,
    evalues_has_finite_e_sup,
    family_blob_executed,
    family_blob_forbidden_metrics_absent,
    family_blob_has_finite_observation,
    family_blob_nonempty,
    h1_hypothesis_consistency_errors,
    h2_hypothesis_consistency_errors,
    h3_hypothesis_consistency_errors,
    h4_hypothesis_consistency_errors,
    h4b_hypothesis_consistency_errors,
    h7_hypothesis_consistency_errors,
    h8_hypothesis_consistency_errors,
    h9_hypothesis_consistency_errors,
    h10_hypothesis_consistency_errors,
    h11_hypothesis_consistency_errors,
    h12_hypothesis_consistency_errors,
    h15_hypothesis_consistency_errors,
    h16_h18_panel_kupiec_consistency_errors,
    h19_hypothesis_consistency_errors,
    h20_hypothesis_consistency_errors,
    h21_hypothesis_consistency_errors,
    h22_hypothesis_consistency_errors,
    hypotheses_include_h1,
    hypotheses_include_h2,
    hypotheses_include_h3,
    hypotheses_include_h4,
    hypotheses_include_h4b,
    hypotheses_include_h7,
    hypotheses_include_h8,
    hypotheses_include_h9,
    hypotheses_include_h10,
    hypotheses_include_h11,
    hypotheses_include_h12,
    hypotheses_include_h15,
    hypotheses_include_h16,
    hypotheses_include_h17,
    hypotheses_include_h18,
    hypotheses_include_h19,
    hypotheses_include_h20,
    hypotheses_include_h21,
    hypotheses_include_h22,
    jackknife_plus_has_finite_coverage,
    jp_cv_blob_requires_marginal_coverage_scope,
    mondrian_aci_has_finite_high_x_kupiec_p,
    mondrian_has_finite_high_x_kupiec_p,
    northset_h23_h28_consistency_errors,
    northset_has_finite_book_uncrossed_rate,
    northset_has_finite_imbalance_p_ic,
    northset_has_finite_ohlc_identity_rate,
    oracle_has_finite_ls_p,
    oracle_has_finite_p_ic,
    panel_family_has_finite_kupiec_p,
    rankers_oracle_raw,
    tail_es_battery_keys_present,
    tail_es_battery_missing_keys,
    tail_has_finite_christoffersen_cc_p,
    tail_has_finite_kupiec_p,
    tail_var_battery_keys_present,
    tail_var_battery_missing_keys,
    volatility_has_finite_dm_p,
    weighted_conformal_has_finite_kupiec_p,
)
from quant_fund.research.catalog import (
    REQUIRED_BENCHMARK_FAMILIES as CATALOG_REQUIRED,
)
from quant_fund.research.receipt_schema import (
    migrate_research_receipt,
    unavailable_overfitting_block,
)
from quant_fund.research.verify import (
    REQUIRED_BENCHMARK_FAMILIES,
    _receipt_digest,
    verify_research_artifact,
)
from quant_fund.utils.hashing import hash_file


def test_benchmark_catalog_order_matches_required_set() -> None:
    assert len(BENCHMARK_FAMILY_ORDER) == 23
    assert len(set(BENCHMARK_FAMILY_ORDER)) == len(BENCHMARK_FAMILY_ORDER)
    assert set(BENCHMARK_FAMILY_ORDER) == REQUIRED_BENCHMARK_FAMILIES
    # verify re-export must stay identical to catalog source of truth
    assert REQUIRED_BENCHMARK_FAMILIES is CATALOG_REQUIRED
    assert frozenset(BENCHMARK_FAMILY_ORDER) == CATALOG_REQUIRED
    assert BENCHMARK_CATALOG_VERSION == 2
    assert "northset" in REQUIRED_BENCHMARK_FAMILIES


def _receipt(tmp_path: Path) -> Path:
    run_id = "a" * 64
    immutable = tmp_path / "immutable"
    immutable.mkdir()
    (immutable / f"{run_id}.md").write_text("# research")
    payload = {
        "schema_version": RESEARCH_RECEIPT_SCHEMA_VERSION,
        "firm": "Artificial Hedge",
        "product": "Dipcatcher",
        "version": "1.0.0",
        "generated_at": "2026-09-16T00:00:00+00:00",
        "data_source": "SYNTHETIC",
        "synthetic": True,
        "disclaimer": "research only",
        "ranking_target": "future_return_1",
        "claim": "research_only",
        "rankers": [],
        "hypotheses": [],
        "provenance": {
            "run_id": run_id,
            "git_revision": "HEAD",
            "git_worktree_sha256": "e" * 64,
            "config_sha256": "b" * 64,
            "dataset_sha256": "c" * 64,
            "dataset_content_sha256": "d" * 64,
            "northset_inputs_sha256": "e" * 64,
            "row_count": 10,
            "column_count": 3,
            "point_in_time": True,
            "execution_claim": "research_only",
            "benchmark_catalog_version": BENCHMARK_CATALOG_VERSION,
            "runtime": {
                "python": "3.12.0",
                "implementation": "CPython",
                "platform": "test",
                "machine": "test",
                "byteorder": "little",
                "packages": {
                    "numpy": "2.0.0",
                    "polars": "1.0.0",
                    "scipy": "1.0.0",
                    "scikit-learn": "1.0.0",
                },
            },
        },
        "scorecard": {
            name: {
                "executed": True,
                "nonempty": True,
                "finite_observation": True,
                "forbidden_metrics_absent": True,
                "claim": "research_metric_only",
            }
            for name in REQUIRED_BENCHMARK_FAMILIES
        },
        "families": {name: {"executed": True} for name in REQUIRED_BENCHMARK_FAMILIES},
        "backtest_overfitting": unavailable_overfitting_block(),
        "artifacts": {
            "immutable_json": str(immutable / f"{run_id}.json"),
            "immutable_markdown": str(immutable / f"{run_id}.md"),
            "immutable_markdown_sha256": hash_file(immutable / f"{run_id}.md"),
        },
    }
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path = tmp_path / "latest.json"
    path.write_text(json.dumps(payload))
    (immutable / f"{run_id}.json").write_text(json.dumps(payload))
    return path


def test_verify_research_artifact_accepts_complete_receipt(tmp_path: Path) -> None:
    result = verify_research_artifact(_receipt(tmp_path))
    assert result["valid"] is True
    assert result["errors"] == []


def test_verify_research_artifact_rejects_identity_tampering(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["product"] = "Not Dipcatcher"
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "invalid_notebook_product" in result["errors"]


def test_verify_research_artifact_rejects_source_label_mismatch(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["data_source"] = "parquet"
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "notebook_source_synthetic_mismatch" in result["errors"]


def test_verify_research_artifact_rejects_malformed_ranker_entries(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["rankers"] = [{"n_dates": -1}, "not-a-ranker"]
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "invalid_ranker_name:0" in result["errors"]
    assert "invalid_ranker_n_dates:0" in result["errors"]
    assert "invalid_ranker:1" in result["errors"]


def test_verify_research_artifact_rejects_duplicate_ranker_names(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["rankers"] = [{"name": "ridge", "n_dates": 10}, {"name": " ridge ", "n_folds": 10}]
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "duplicate_ranker_name:ridge" in result["errors"]


def test_verify_research_artifact_rejects_timestamp_without_timezone(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["generated_at"] = "2026-09-16T00:00:00"
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "invalid_notebook_generated_at" in result["errors"]


def test_verify_research_artifact_rejects_duplicate_hypotheses(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    hypothesis = {
        "id": "H1",
        "statement": "x",
        "test": "test",
        "statistic": 0.0,
        "p_value": 1.0,
        "reject_raw": False,
        "reject_fdr": False,
        "decision": "x",
        "family": "calibration",
    }
    payload["hypotheses"] = [hypothesis, dict(hypothesis)]
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "duplicate_hypothesis_id:H1" in result["errors"]


def test_verify_research_artifact_rejects_malformed_hypothesis_statistics(
    tmp_path: Path,
) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["hypotheses"] = [
        {
            "id": "H1",
            "statement": "x",
            "test": "test",
            "statistic": "not-a-number",
            "p_value": 1.5,
            "reject_raw": "yes",
            "reject_fdr": False,
            "decision": "x",
            "family": "calibration",
        }
    ]
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "invalid_hypothesis_statistic:0" in result["errors"]
    assert "invalid_hypothesis_p_value:0" in result["errors"]
    assert "invalid_hypothesis_reject_raw:0" in result["errors"]


def test_verify_research_artifact_accepts_zero_p_value_boundary(tmp_path: Path) -> None:
    """p_value=0.0 is a legal [0, 1] boundary value (mutation-testing find:
    the 0.0 <= p lower bound was previously untested — a LtE->Lt mutant of
    ``_p_value_valid`` survived the whole suite)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["hypotheses"] = [
        {
            "id": "H1",
            "statement": "x",
            "test": "test",
            "statistic": 0.0,
            "p_value": 0.0,
            "reject_raw": False,
            "reject_fdr": False,
            "decision": "x",
            "family": "calibration",
        }
    ]
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert "invalid_hypothesis_p_value:0" not in result["errors"]


def test_verify_research_artifact_rejects_negative_p_value(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["hypotheses"] = [
        {
            "id": "H1",
            "statement": "x",
            "test": "test",
            "statistic": 0.0,
            "p_value": -1e-12,
            "reject_raw": False,
            "reject_fdr": False,
            "decision": "x",
            "family": "calibration",
        }
    ]
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "invalid_hypothesis_p_value:0" in result["errors"]


def test_verify_research_artifact_accepts_unit_p_value_boundary(tmp_path: Path) -> None:
    """p_value=1.0 is the legal upper boundary (mutation-testing find: the
    ``numeric <= 1.0`` upper bound was untested for acceptance — a LtE->Lt
    mutant of the SECOND comparison on the line survived the suite)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["hypotheses"] = [
        {
            "id": "H1",
            "statement": "x",
            "test": "test",
            "statistic": 0.0,
            "p_value": 1.0,
            "reject_raw": False,
            "reject_fdr": False,
            "decision": "x",
            "family": "calibration",
        }
    ]
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert "invalid_hypothesis_p_value:0" not in result["errors"]


def test_verify_research_artifact_rejects_empty_families_dict(tmp_path: Path) -> None:
    """An empty families dict must flag families_missing (mutation find: the
    ``or not families`` arm of the isinstance-or-empty guard was untested —
    an Or->And mutant survived)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"] = {}
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "families_missing" in result["errors"]


def test_verify_research_artifact_accepts_integral_ranker_fields(tmp_path: Path) -> None:
    """Integer-valued ranker fields are valid (mutation find: the
    ``int(value) != float(value)`` integrality check had no ACCEPTING test —
    a NotEq->Eq mutant survived because the base receipt's rankers list is
    empty and the only int-field test used -1)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["rankers"] = [{"name": "ridge", "n_dates": 10, "n_folds": 5}]
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert not any(e.startswith("invalid_ranker") for e in result["errors"])


def test_verify_research_artifact_accepts_zero_ranker_count(tmp_path: Path) -> None:
    """n_dates=0 is non-negative and must pass the ``int(value) < 0`` guard
    (mutation find: an int 0->1 perturbation of the bound survived — no test
    exercised a zero-valued ranker count)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["rankers"] = [{"name": "ridge", "n_dates": 0}]
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert not any(e.startswith("invalid_ranker") for e in result["errors"])


def test_verify_research_artifact_rejects_non_string_package_version(tmp_path: Path) -> None:
    """A non-string package VERSION must flag runtime_packages_invalid
    (mutation find: the value-side ``or not value`` arm was untested — an
    Or->And mutant survived since no test used a truthy non-string value)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["provenance"]["runtime"]["packages"]["numpy"] = 42
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "runtime_packages_invalid" in result["errors"]


def test_verify_research_artifact_rejects_empty_package_name(tmp_path: Path) -> None:
    """An EMPTY package name must flag runtime_packages_invalid (mutation
    find: the name-side ``or not name`` arm — a non-string name is impossible
    from JSON, so the empty-string case is the only distinguisher for an
    Or->And mutant of that guard)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["provenance"]["runtime"]["packages"][""] = "1.0.0"
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "runtime_packages_invalid" in result["errors"]


def test_receipt_digest_is_insertion_order_invariant() -> None:
    """The canonical receipt digest must not depend on dict insertion order
    (full-210 mutation find: sort_keys=True->False survived — no test hashed
    same-content payloads built in different key orders)."""
    from quant_fund.research.verify import _receipt_digest

    a = {"z": 1, "a": {"y": 2, "x": [3, 4]}, "m": "s"}
    b = {"m": "s", "a": {"x": [3, 4], "y": 2}, "z": 1}
    assert _receipt_digest(a) == _receipt_digest(b)
    # The self-referential digest field is excluded from its own hash.
    c = dict(a, artifacts={"immutable_json_sha256": "f" * 64, "keep": 1})
    d = dict(a, artifacts={"keep": 1})
    assert _receipt_digest(c) == _receipt_digest(d)


def test_verify_research_artifact_accepts_zero_provenance_counts(tmp_path: Path) -> None:
    """row_count/column_count == 0 are non-negative and must pass the
    ``value < 0`` guard (full-210 mutation find: Lt->LtE and int 0->1
    mutants both survived — no test exercised zero-valued counts)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["provenance"]["row_count"] = 0
    payload["provenance"]["column_count"] = 0
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert "invalid_row_count" not in result["errors"]
    assert "invalid_column_count" not in result["errors"]


def test_verify_research_artifact_non_dict_scorecard_reports_zero_families(
    tmp_path: Path,
) -> None:
    """A non-dict scorecard must report scorecard_families == 0, not 1
    (full-210 mutation find: the ``else 0`` fallback int-perturbed to 1
    survived — no test asserted the field on a malformed scorecard)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["scorecard"] = ["not", "a", "dict"]
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert result["scorecard_families"] == 0


def test_verify_research_artifact_non_string_markdown_path_is_unverifiable(
    tmp_path: Path,
) -> None:
    """A non-string immutable_markdown must degrade to
    immutable_markdown_hash_unverifiable, never crash (full-210 mutation
    find: And->Or in the path ternary survived — the distinguisher is a
    non-str value, where the mutant evaluates Path(42) and raises TypeError
    while the original short-circuits to None)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["artifacts"]["immutable_markdown"] = 42
    payload["artifacts"]["immutable_markdown_sha256"] = "f" * 64
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "immutable_markdown_hash_unverifiable" in result["errors"]


def test_verify_research_artifact_allows_nan_unavailable_hypothesis_values(
    tmp_path: Path,
) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["hypotheses"] = [
        {
            "id": "H1",
            "statement": "unavailable test",
            "test": "test",
            "statistic": float("nan"),
            "p_value": float("nan"),
            "reject_raw": False,
            "reject_fdr": False,
            "decision": "unavailable",
            "family": "calibration",
        }
    ]
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    Path(payload["artifacts"]["immutable_json"]).write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True


def test_verify_research_artifact_rejects_rejection_without_p_value(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["hypotheses"] = [
        {
            "id": "H1",
            "statement": "unavailable test",
            "test": "test",
            "statistic": float("nan"),
            "p_value": None,
            "reject_raw": True,
            "reject_fdr": False,
            "decision": "rejected",
            "family": "calibration",
        }
    ]
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_rejection_without_p_value:0" in result["errors"]


def test_verify_research_artifact_allows_null_unavailable_hypothesis_p_value(
    tmp_path: Path,
) -> None:
    """Agent ``_jsonable`` maps nan→JSON null; bound hyps must still verify."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["hypotheses"] = [
        {
            "id": "H15_cv_plus_floor",
            "statement": "CV+ coverage meets its aggregation-specific finite-sample floor.",
            "test": "coverage − floor check; not FDR",
            "statistic": 0.05,
            "p_value": None,
            "reject_raw": False,
            "reject_fdr": False,
            "decision": "CV+ coverage is above its stated floor.",
            "family": "bound",
            "meets_floor": True,
        }
    ]
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    Path(payload["artifacts"]["immutable_json"]).write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []


def test_verify_research_artifact_rejects_tampered_immutable_markdown(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    Path(payload["artifacts"]["immutable_markdown"]).write_text("# tampered")
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "immutable_markdown_hash_mismatch" in result["errors"]


def test_verify_research_artifact_rejects_tampered_immutable_json(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable_payload = json.loads(immutable.read_text())
    immutable_payload["product"] = "Tampered"
    immutable.write_text(json.dumps(immutable_payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "immutable_json_hash_mismatch" in result["errors"]


def test_verify_research_artifact_rejects_incomplete_scorecard(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["scorecard"]["ranking"]["forbidden_metrics_absent"] = False
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "scorecard_invalid:ranking" in result["errors"]


def test_verify_research_artifact_rejects_forged_scorecard_claim(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["scorecard"]["ranking"]["claim"] = "live_performance"
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "scorecard_claim_invalid:ranking" in result["errors"]


def test_verify_research_artifact_rejects_missing_families(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    del payload["families"]
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "families_missing" in result["errors"]


def test_verify_research_artifact_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    path.write_text("[]")
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert result["errors"] == ["notebook_not_object"]


def test_verify_research_artifact_rejects_family_scorecard_mismatch(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"].pop("ranking")
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "families_missing:ranking" in result["errors"]


def test_verify_research_artifact_rejects_nonfinite_scorecard(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["scorecard"]["ranking"]["finite_observation"] = False
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "scorecard_invalid:ranking" in result["errors"]


def test_verify_research_artifact_rejects_mismatched_immutable_json(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps({"provenance": {"run_id": "b" * 64}}))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "immutable_provenance_mismatch" in result["errors"]


def test_verify_research_artifact_rejects_missing_runtime(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    del payload["provenance"]["runtime"]
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "runtime_missing" in result["errors"]


def test_verify_research_artifact_rejects_incomplete_runtime_fingerprint(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    del payload["provenance"]["runtime"]["packages"]["polars"]
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "runtime_packages_missing_required:polars" in result["errors"]


def test_verify_research_artifact_rejects_invalid_provenance_dimensions(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["provenance"]["git_revision"] = "  "
    payload["provenance"]["row_count"] = -1
    payload["provenance"]["column_count"] = True
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "invalid_git_revision" in result["errors"]
    assert "invalid_row_count" in result["errors"]
    assert "invalid_column_count" in result["errors"]


def test_verify_research_artifact_rejects_non_hex_digests(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["provenance"]["run_id"] = "g" * 64
    payload["provenance"]["config_sha256"] = "G" * 64
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "invalid_run_id" in result["errors"]
    assert "invalid_config_sha256" in result["errors"]


def test_verify_research_artifact_rejects_external_artifacts(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    external = tmp_path.parent / "external-research.json"
    external.write_text(json.dumps(payload))
    payload["artifacts"]["immutable_json"] = str(external)
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "artifact_outside_receipt_root:immutable_json" in result["errors"]


def test_verify_research_artifact_resolves_relative_artifacts_from_receipt_root(
    tmp_path: Path,
) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["artifacts"]["immutable_json"] = "immutable/" + ("a" * 64) + ".json"
    payload["artifacts"]["immutable_markdown"] = "immutable/" + ("a" * 64) + ".md"
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    (tmp_path / "immutable" / ("a" * 64 + ".json")).write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True


def test_verify_research_artifact_rejects_incomplete_benchmark_catalog(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["scorecard"].pop("cpcv")
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "scorecard_families_missing:cpcv" in result["errors"]


def test_verify_research_artifact_rejects_unknown_scorecard_family(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["scorecard"]["not_a_real_family"] = {
        "executed": True,
        "nonempty": True,
        "finite_observation": True,
        "forbidden_metrics_absent": True,
    }
    path.write_text(json.dumps(payload))
    # keep immutable in sync so unknown key is the fail reason
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert any(e.startswith("scorecard_families_unknown:") for e in result["errors"])
    assert "not_a_real_family" in ",".join(result["errors"])


def test_verify_research_artifact_rejects_wrong_catalog_version(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["provenance"]["benchmark_catalog_version"] = BENCHMARK_CATALOG_VERSION + 99
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "invalid_benchmark_catalog_version" in result["errors"]


def test_catalog_required_set_completeness_frozen() -> None:
    """Required bench set must stay complete — missing family is a catalog bug."""
    expected = {
        "ranking",
        "alpha",
        "volatility",
        "distribution",
        "regime",
        "tail",
        "drawdown",
        "liquidity",
        "reinforcement",
        "conformal",
        "evalues",
        "jackknife_plus",
        "crc",
        "weighted_conformal",
        "interval_risk",
        "quantile_bandit",
        "cv_plus",
        "cpcv",
        "localized_conformal",
        "conformal_rank",
        "online_crc",
        "portfolio_conformal",
        "northset",
    }
    assert frozenset(expected) == CATALOG_REQUIRED
    assert len(CATALOG_REQUIRED) == 23


def test_verify_research_artifact_rejects_family_forbidden_metrics(tmp_path: Path) -> None:
    """Fail closed when a family blob sneaks forbidden keys (even with forged flag)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"] = {
        "ranking": {"mean_ic": 0.1, "sharpe": 3.0},  # poisoned
    }
    # forge scorecard flag to True for ranking — verify must still catch families
    payload["scorecard"]["ranking"]["forbidden_metrics_absent"] = True
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "families_forbidden_metrics:ranking" in result["errors"]
    assert "scorecard_forbidden_flag_forged:ranking" in result["errors"]


def test_verify_research_artifact_scans_families_when_scorecard_is_invalid(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload.pop("scorecard")
    payload["families"] = {"ranking": {"nested_pnl": 1.0}}
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "families_forbidden_metrics:ranking" in result["errors"]


def test_verify_research_artifact_accepts_clean_families(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"] = {name: {"executed": True} for name in REQUIRED_BENCHMARK_FAMILIES}
    payload["families"]["ranking"] = {"mean_ic": 0.1, "coverage": 0.9}
    payload["families"]["cpcv"] = {
        "purge_embargo_valid": True,
        "claim": "validation_integrity_only",
    }
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []


def test_tail_var_battery_helper_empty_and_no_kupiec() -> None:
    """Empty {} and blobs without Kupiec markers skip the VaR-battery contract."""
    assert tail_var_battery_missing_keys({}) == []
    assert tail_var_battery_keys_present({}) is True
    assert tail_var_battery_missing_keys(None) == []  # type: ignore[arg-type]
    assert tail_var_battery_keys_present(None) is True  # type: ignore[arg-type]
    assert tail_var_battery_missing_keys({"executed": True}) == []
    assert tail_var_battery_keys_present({"executed": True}) is True
    assert TAIL_VAR_BATTERY_REQUIRED_WHEN_KUPIEC == (
        "christoffersen_cc_p",
        "christoffersen_cc_lr",
        "christoffersen_ind_p",
        "christoffersen_ind_lr",
    )


def test_tail_var_battery_helper_reports_missing_when_kupiec_present() -> None:
    missing = tail_var_battery_missing_keys({"kupiec_p": float("nan")})
    assert missing == list(TAIL_VAR_BATTERY_REQUIRED_WHEN_KUPIEC)
    assert tail_var_battery_keys_present({"kupiec_p": float("nan")}) is False
    # kupiec_lr alone also triggers (Day Wave 18 marker set).
    missing_lr = tail_var_battery_missing_keys({"kupiec_lr": 1.0})
    assert missing_lr == list(TAIL_VAR_BATTERY_REQUIRED_WHEN_KUPIEC)
    assert tail_var_battery_keys_present({"kupiec_lr": 1.0}) is False
    complete = {
        "kupiec_p": float("nan"),
        "kupiec_lr": float("nan"),
        "christoffersen_cc_p": float("nan"),
        "christoffersen_cc_lr": float("nan"),
        "christoffersen_ind_p": float("nan"),
        "christoffersen_ind_lr": float("nan"),
        "acerbi_szekely_z1": float("nan"),
        "acerbi_szekely_z2": float("nan"),
        "fissler_ziegel_mean": float("nan"),
        "es_hit_count": float("nan"),
    }
    assert tail_var_battery_missing_keys(complete) == []
    assert tail_var_battery_keys_present(complete) is True
    # Forbidden-metrics hygiene unchanged beside VaR-battery presence.
    assert family_blob_forbidden_metrics_absent(complete) is True
    dirty = dict(complete)
    dirty["flag_high_sharpe"] = 1.0
    assert family_blob_forbidden_metrics_absent(dirty) is False
    assert tail_var_battery_keys_present(dirty) is True  # presence still ok


def test_verify_rejects_tail_with_kupiec_but_missing_christoffersen_cc(tmp_path: Path) -> None:
    """Incomplete VaR battery fail-closed: kupiec_p without christoffersen_cc_*."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["tail"] = {"kupiec_p": 0.4, "kupiec_lr": 1.0}
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "tail_var_battery_incomplete:christoffersen_cc_p" in result["errors"]
    assert "tail_var_battery_incomplete:christoffersen_cc_lr" in result["errors"]
    assert "tail_var_battery_incomplete:christoffersen_ind_p" in result["errors"]
    assert "tail_var_battery_incomplete:christoffersen_ind_lr" in result["errors"]


def test_verify_accepts_tail_with_kupiec_and_christoffersen_keys_nan_ok(
    tmp_path: Path,
) -> None:
    """Complete key presence (NaN values ok) passes the VaR-battery contract."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["tail"] = {
        "executed": True,
        "kupiec_p": float("nan"),
        "kupiec_lr": float("nan"),
        "christoffersen_cc_p": float("nan"),
        "christoffersen_cc_lr": float("nan"),
        "christoffersen_ind_p": float("nan"),
        "christoffersen_ind_lr": float("nan"),
        "acerbi_szekely_z1": float("nan"),
        "acerbi_szekely_z2": float("nan"),
        "fissler_ziegel_mean": float("nan"),
        "es_hit_count": float("nan"),
    }
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert not any(e.startswith("tail_var_battery_incomplete:") for e in result["errors"])


def test_verify_rejects_tail_with_kupiec_lr_only_missing_cc(tmp_path: Path) -> None:
    """kupiec_lr without Christoffersen CC/ind fails the soft VaR-battery verify."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["tail"] = {"kupiec_lr": 0.9}
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "tail_var_battery_incomplete:christoffersen_cc_p" in result["errors"]
    assert "tail_var_battery_incomplete:christoffersen_cc_lr" in result["errors"]


def test_verify_accepts_empty_tail_family(tmp_path: Path) -> None:
    """Empty {} tail skips VaR-battery check (tiny panels may omit battery)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["tail"] = {"executed": True}
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert not any(e.startswith("tail_var_battery_incomplete:") for e in result["errors"])


def test_dist_crps_eprocess_helper_empty_and_no_dm() -> None:
    """Empty {} and blobs without dm_crps_* markers skip the e-process contract."""
    assert dist_crps_eprocess_missing_keys({}) == []
    assert dist_crps_eprocess_keys_present({}) is True
    assert dist_crps_eprocess_missing_keys(None) == []  # type: ignore[arg-type]
    assert dist_crps_eprocess_keys_present(None) is True  # type: ignore[arg-type]
    assert dist_crps_eprocess_missing_keys({"executed": True}) == []
    assert dist_crps_eprocess_keys_present({"executed": True}) is True
    assert DIST_CRPS_EPROCESS_REQUIRED_WHEN_DM == (
        "e_dm_crps_final",
        "e_dm_crps_reject",
        "e_dm_crps_n",
    )
    assert DIST_CRPS_SCALED_EPROCESS_REQUIRED_WHEN_DM == (
        "e_dm_crps_scaled_final",
        "e_dm_crps_scaled_reject",
        "e_dm_crps_scaled_n",
    )


def test_dist_crps_eprocess_helper_reports_missing_when_dm_present() -> None:
    missing = dist_crps_eprocess_missing_keys({"dm_crps_p": float("nan")})
    assert missing == list(DIST_CRPS_EPROCESS_REQUIRED_WHEN_DM)
    assert dist_crps_eprocess_keys_present({"dm_crps_p": float("nan")}) is False
    missing_s = dist_crps_eprocess_missing_keys({"dm_crps_scaled_p": 0.1})
    assert missing_s == list(DIST_CRPS_SCALED_EPROCESS_REQUIRED_WHEN_DM)
    assert dist_crps_eprocess_keys_present({"dm_crps_scaled_p": 0.1}) is False
    both = dist_crps_eprocess_missing_keys({"dm_crps_p": 0.2, "dm_crps_scaled_p": float("nan")})
    assert both == list(DIST_CRPS_EPROCESS_REQUIRED_WHEN_DM) + list(
        DIST_CRPS_SCALED_EPROCESS_REQUIRED_WHEN_DM
    )
    complete = {
        "dm_crps_p": float("nan"),
        "dm_crps_scaled_p": float("nan"),
        "e_dm_crps_final": float("nan"),
        "e_dm_crps_reject": False,
        "e_dm_crps_n": 0,
        "e_dm_crps_scaled_final": float("nan"),
        "e_dm_crps_scaled_reject": False,
        "e_dm_crps_scaled_n": 0,
    }
    assert dist_crps_eprocess_missing_keys(complete) == []
    assert dist_crps_eprocess_keys_present(complete) is True
    assert family_blob_forbidden_metrics_absent(complete) is True
    dirty = dict(complete)
    dirty["flag_high_sharpe"] = 1.0
    assert family_blob_forbidden_metrics_absent(dirty) is False
    assert dist_crps_eprocess_keys_present(dirty) is True


def test_verify_rejects_distribution_with_dm_but_missing_eprocess(tmp_path: Path) -> None:
    """Incomplete CRPS e-process battery: dm_crps_p without e_dm_crps_*."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["distribution"] = {"dm_crps_p": 0.3, "dm_crps_stat": 1.0}
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "dist_crps_eprocess_incomplete:e_dm_crps_final" in result["errors"]
    assert "dist_crps_eprocess_incomplete:e_dm_crps_reject" in result["errors"]
    assert "dist_crps_eprocess_incomplete:e_dm_crps_n" in result["errors"]


def test_verify_rejects_distribution_with_scaled_dm_but_missing_eprocess(
    tmp_path: Path,
) -> None:
    """dm_crps_scaled_p without e_dm_crps_scaled_* fails soft verify."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["distribution"] = {"dm_crps_scaled_p": 0.2}
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "dist_crps_eprocess_incomplete:e_dm_crps_scaled_final" in result["errors"]
    assert "dist_crps_eprocess_incomplete:e_dm_crps_scaled_reject" in result["errors"]
    assert "dist_crps_eprocess_incomplete:e_dm_crps_scaled_n" in result["errors"]


def test_verify_accepts_distribution_with_dm_and_eprocess_keys_nan_ok(
    tmp_path: Path,
) -> None:
    """Complete key presence (NaN/False/0 values ok) passes the e-process contract."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["distribution"] = {
        "executed": True,
        "dm_crps_p": float("nan"),
        "dm_crps_scaled_p": float("nan"),
        "e_dm_crps_final": float("nan"),
        "e_dm_crps_reject": False,
        "e_dm_crps_n": 0,
        "e_dm_crps_scaled_final": float("nan"),
        "e_dm_crps_scaled_reject": False,
        "e_dm_crps_scaled_n": 0,
    }
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert not any(e.startswith("dist_crps_eprocess_incomplete:") for e in result["errors"])


def test_verify_accepts_empty_distribution_family(tmp_path: Path) -> None:
    """Empty {} distribution skips CRPS e-process check."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["distribution"] = {"executed": True}
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert not any(e.startswith("dist_crps_eprocess_incomplete:") for e in result["errors"])


def test_tail_es_battery_helper_empty_and_no_es_markers() -> None:
    """Empty {} and blobs without es_95/realized_es/var_95 skip ES-battery."""
    assert tail_es_battery_missing_keys({}) == []
    assert tail_es_battery_keys_present({}) is True
    assert tail_es_battery_missing_keys(None) == []  # type: ignore[arg-type]
    assert tail_es_battery_keys_present(None) is True  # type: ignore[arg-type]
    assert tail_es_battery_missing_keys({"executed": True}) == []
    assert tail_es_battery_keys_present({"executed": True}) is True
    # Kupiec-only must NOT trigger ES-battery (VaR-battery owns Kupiec⇒Christoffersen).
    assert tail_es_battery_missing_keys({"kupiec_p": 0.4, "kupiec_lr": 1.0}) == []
    assert tail_es_battery_keys_present({"kupiec_p": 0.4}) is True
    assert TAIL_ES_BATTERY_REQUIRED_WHEN_ES_MARKERS == (
        "acerbi_szekely_z1",
        "acerbi_szekely_z2",
        "fissler_ziegel_mean",
        "es_hit_count",
    )


def test_tail_es_battery_helper_reports_missing_when_es_marker_present() -> None:
    missing = tail_es_battery_missing_keys({"es_95": float("nan")})
    assert missing == list(TAIL_ES_BATTERY_REQUIRED_WHEN_ES_MARKERS)
    assert tail_es_battery_keys_present({"es_95": float("nan")}) is False
    # realized_es alone also triggers.
    missing_re = tail_es_battery_missing_keys({"realized_es": 1.0})
    assert missing_re == list(TAIL_ES_BATTERY_REQUIRED_WHEN_ES_MARKERS)
    assert tail_es_battery_keys_present({"realized_es": 1.0}) is False
    # var_95 alone also triggers (even without es_95).
    missing_var = tail_es_battery_missing_keys({"var_95": 0.05})
    assert missing_var == list(TAIL_ES_BATTERY_REQUIRED_WHEN_ES_MARKERS)
    assert tail_es_battery_keys_present({"var_95": 0.05}) is False
    complete = {
        "es_95": float("nan"),
        "acerbi_szekely_z1": float("nan"),
        "acerbi_szekely_z2": float("nan"),
        "fissler_ziegel_mean": float("nan"),
        "es_hit_count": float("nan"),
    }
    assert tail_es_battery_missing_keys(complete) == []
    assert tail_es_battery_keys_present(complete) is True
    assert family_blob_forbidden_metrics_absent(complete) is True
    dirty = dict(complete)
    dirty["flag_high_sharpe"] = 1.0
    assert family_blob_forbidden_metrics_absent(dirty) is False
    assert tail_es_battery_keys_present(dirty) is True  # presence still ok


def test_verify_rejects_tail_with_es_marker_but_missing_acerbi(tmp_path: Path) -> None:
    """Incomplete ES battery fail-closed: es_95 without Acerbi/FZ keys."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["tail"] = {"es_95": 0.1, "var_95": 0.05, "realized_es": 0.12}
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "tail_es_battery_incomplete:acerbi_szekely_z1" in result["errors"]
    assert "tail_es_battery_incomplete:acerbi_szekely_z2" in result["errors"]
    assert "tail_es_battery_incomplete:fissler_ziegel_mean" in result["errors"]
    assert "tail_es_battery_incomplete:es_hit_count" in result["errors"]
    # No Kupiec ⇒ VaR-battery must stay quiet (orthogonal).
    assert not any(e.startswith("tail_var_battery_incomplete:") for e in result["errors"])


def test_verify_accepts_tail_with_es_keys_nan_ok(tmp_path: Path) -> None:
    """Complete ES key presence (NaN values ok) passes the ES-battery contract."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["tail"] = {
        "executed": True,
        "es_95": float("nan"),
        "var_95": float("nan"),
        "realized_es": float("nan"),
        "acerbi_szekely_z1": float("nan"),
        "acerbi_szekely_z2": float("nan"),
        "fissler_ziegel_mean": float("nan"),
        "es_hit_count": float("nan"),
    }
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert not any(e.startswith("tail_es_battery_incomplete:") for e in result["errors"])
    assert not any(e.startswith("tail_var_battery_incomplete:") for e in result["errors"])


def test_verify_rejects_tail_with_var_95_only_missing_es(tmp_path: Path) -> None:
    """var_95 without Acerbi/FZ fails the soft ES-battery verify."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["tail"] = {"var_95": 0.05}
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "tail_es_battery_incomplete:acerbi_szekely_z1" in result["errors"]
    assert "tail_es_battery_incomplete:fissler_ziegel_mean" in result["errors"]
    assert not any(e.startswith("tail_var_battery_incomplete:") for e in result["errors"])


def test_verify_accepts_empty_tail_skips_es_battery(tmp_path: Path) -> None:
    """Empty {} tail skips ES-battery check (tiny panels may omit battery)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["tail"] = {"executed": True}
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert not any(e.startswith("tail_es_battery_incomplete:") for e in result["errors"])
    assert not any(e.startswith("tail_var_battery_incomplete:") for e in result["errors"])


def test_verify_kupiec_only_does_not_fire_es_battery(tmp_path: Path) -> None:
    """Kupiec without ES markers → VaR-battery only; ES-battery stays quiet."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["tail"] = {"kupiec_p": 0.4, "kupiec_lr": 1.0}
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "tail_var_battery_incomplete:christoffersen_cc_p" in result["errors"]
    assert not any(e.startswith("tail_es_battery_incomplete:") for e in result["errors"])


def test_verify_both_batteries_fire_independently_when_both_markers(
    tmp_path: Path,
) -> None:
    """Kupiec + es_95 present but companion keys missing → both battery errors."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["tail"] = {
        "kupiec_p": 0.4,
        "kupiec_lr": 1.0,
        "es_95": 0.1,
    }
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "tail_var_battery_incomplete:christoffersen_cc_p" in result["errors"]
    assert "tail_es_battery_incomplete:acerbi_szekely_z1" in result["errors"]
    assert "tail_es_battery_incomplete:es_hit_count" in result["errors"]


def test_verify_both_batteries_complete_passes(tmp_path: Path) -> None:
    """Full VaR + ES battery key presence (NaN ok) passes together."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["tail"] = {
        "executed": True,
        "kupiec_p": float("nan"),
        "kupiec_lr": float("nan"),
        "christoffersen_cc_p": float("nan"),
        "christoffersen_cc_lr": float("nan"),
        "christoffersen_ind_p": float("nan"),
        "christoffersen_ind_lr": float("nan"),
        "es_95": float("nan"),
        "var_95": float("nan"),
        "realized_es": float("nan"),
        "acerbi_szekely_z1": float("nan"),
        "acerbi_szekely_z2": float("nan"),
        "fissler_ziegel_mean": float("nan"),
        "es_hit_count": float("nan"),
    }
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []


def test_verify_rejects_forged_tail_var_battery_ok_when_incomplete(tmp_path: Path) -> None:
    """Forge tail_var_battery_ok=True with Kupiec but missing Christoffersen → forge error."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["tail"] = {"kupiec_p": 0.4, "kupiec_lr": 1.0}
    payload["scorecard"]["tail"]["tail_var_battery_ok"] = True
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "scorecard_tail_var_battery_flag_forged:tail" in result["errors"]
    assert "tail_var_battery_incomplete:christoffersen_cc_p" in result["errors"]


def test_verify_rejects_forged_tail_es_battery_ok_when_incomplete(tmp_path: Path) -> None:
    """Forge tail_es_battery_ok=True with es_95 but missing Acerbi/FZ → forge error."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["tail"] = {"es_95": 0.1}
    payload["scorecard"]["tail"]["tail_es_battery_ok"] = True
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "scorecard_tail_es_battery_flag_forged:tail" in result["errors"]
    assert "tail_es_battery_incomplete:acerbi_szekely_z1" in result["errors"]


def test_verify_rejects_forged_dist_crps_eprocess_ok_when_incomplete(tmp_path: Path) -> None:
    """Forge dist_crps_eprocess_ok=True with dm_crps_p but missing e_dm_crps_* → forge."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["distribution"] = {"dm_crps_p": 0.3}
    payload["scorecard"]["distribution"]["dist_crps_eprocess_ok"] = True
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "scorecard_dist_crps_eprocess_flag_forged:distribution" in result["errors"]
    assert "dist_crps_eprocess_incomplete:e_dm_crps_final" in result["errors"]


def test_verify_honest_battery_ok_true_when_keys_present_no_forge(tmp_path: Path) -> None:
    """Honest True scorecard flags with complete keys → no forge errors (accept path)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["tail"] = {
        "executed": True,
        "kupiec_p": float("nan"),
        "kupiec_lr": float("nan"),
        "christoffersen_cc_p": float("nan"),
        "christoffersen_cc_lr": float("nan"),
        "christoffersen_ind_p": float("nan"),
        "christoffersen_ind_lr": float("nan"),
        "es_95": float("nan"),
        "acerbi_szekely_z1": float("nan"),
        "acerbi_szekely_z2": float("nan"),
        "fissler_ziegel_mean": float("nan"),
        "es_hit_count": float("nan"),
    }
    payload["families"]["distribution"] = {
        "executed": True,
        "dm_crps_p": float("nan"),
        "e_dm_crps_final": float("nan"),
        "e_dm_crps_reject": False,
        "e_dm_crps_n": 0,
    }
    payload["scorecard"]["tail"]["tail_var_battery_ok"] = True
    payload["scorecard"]["tail"]["tail_es_battery_ok"] = True
    payload["scorecard"]["distribution"]["dist_crps_eprocess_ok"] = True
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert not any("flag_forged" in e and "forbidden" not in e for e in result["errors"])
    assert "scorecard_tail_var_battery_flag_forged:tail" not in result["errors"]
    assert "scorecard_tail_es_battery_flag_forged:tail" not in result["errors"]
    assert "scorecard_dist_crps_eprocess_flag_forged:distribution" not in result["errors"]


def test_verify_absent_battery_ok_flags_no_forge_even_if_incomplete(tmp_path: Path) -> None:
    """Absent battery_ok flags → no forge errors; incomplete:* still fires."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    # Incomplete VaR + ES batteries and incomplete dist e-process; no *_ok flags set.
    payload["families"]["tail"] = {"kupiec_p": 0.4, "es_95": 0.1}
    payload["families"]["distribution"] = {"dm_crps_p": 0.3}
    for key in (
        "tail_var_battery_ok",
        "tail_es_battery_ok",
        "dist_crps_eprocess_ok",
    ):
        payload["scorecard"]["tail"].pop(key, None)
        payload["scorecard"]["distribution"].pop(key, None)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "scorecard_tail_var_battery_flag_forged:tail" not in result["errors"]
    assert "scorecard_tail_es_battery_flag_forged:tail" not in result["errors"]
    assert "scorecard_dist_crps_eprocess_flag_forged:distribution" not in result["errors"]
    assert not any("flag_forged" in e and "forbidden" not in e for e in result["errors"])
    assert "tail_var_battery_incomplete:christoffersen_cc_p" in result["errors"]
    assert "tail_es_battery_incomplete:acerbi_szekely_z1" in result["errors"]
    assert "dist_crps_eprocess_incomplete:e_dm_crps_final" in result["errors"]


def test_verify_false_battery_ok_flags_no_forge_even_if_incomplete(tmp_path: Path) -> None:
    """Explicit False battery_ok is not a forge; incomplete:* still fires."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["tail"] = {"kupiec_lr": 1.0}
    payload["scorecard"]["tail"]["tail_var_battery_ok"] = False
    payload["scorecard"]["tail"]["tail_es_battery_ok"] = False
    payload["families"]["distribution"] = {"dm_crps_scaled_p": 0.2}
    payload["scorecard"]["distribution"]["dist_crps_eprocess_ok"] = False
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "scorecard_tail_var_battery_flag_forged:tail" not in result["errors"]
    assert "scorecard_tail_es_battery_flag_forged:tail" not in result["errors"]
    assert "scorecard_dist_crps_eprocess_flag_forged:distribution" not in result["errors"]
    assert "tail_var_battery_incomplete:christoffersen_cc_p" in result["errors"]
    assert "dist_crps_eprocess_incomplete:e_dm_crps_scaled_final" in result["errors"]


def _h4b_hypothesis(*, family: str = "calibration", p_value: float = 0.35) -> dict:
    """Minimal valid H4b row matching REQUIRED_HYPOTHESIS_FIELDS."""
    return {
        "id": H4B_HYPOTHESIS_ID,
        "statement": "Christoffersen CC conditional coverage.",
        "test": "Christoffersen CC",
        "statistic": 1.2,
        "p_value": p_value,
        "reject_raw": False,
        "reject_fdr": False,
        "decision": "VaR hits consistent with independent 5% conditional coverage.",
        "family": family,
    }


def test_catalog_tail_has_finite_christoffersen_cc_p() -> None:
    assert tail_has_finite_christoffersen_cc_p({"christoffersen_cc_p": 0.4}) is True
    assert tail_has_finite_christoffersen_cc_p({"christoffersen_cc_p": float("nan")}) is False
    assert tail_has_finite_christoffersen_cc_p({"christoffersen_cc_p": float("inf")}) is False
    assert tail_has_finite_christoffersen_cc_p({}) is False
    assert tail_has_finite_christoffersen_cc_p(None) is False


def test_catalog_h4b_errors_skip_when_cc_p_nonfinite() -> None:
    assert h4b_hypothesis_consistency_errors({"christoffersen_cc_p": float("nan")}, []) == []
    assert h4b_hypothesis_consistency_errors({}, []) == []
    assert h4b_hypothesis_consistency_errors(None, []) == []


def test_verify_rejects_finite_christoffersen_cc_p_without_h4b(tmp_path: Path) -> None:
    """Finite tail.christoffersen_cc_p without H4b hyp → soft fail-closed."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    # Finite cc_p alone (no Kupiec) must still require H4b; avoid VaR-battery gate.
    payload["families"]["tail"] = {"christoffersen_cc_p": 0.42, "christoffersen_cc_lr": 1.1}
    payload["hypotheses"] = []
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h4b_missing_despite_finite_christoffersen_cc_p" in result["errors"]


def test_verify_accepts_finite_christoffersen_cc_p_with_h4b(tmp_path: Path) -> None:
    """Finite cc_p + calibration H4b hypothesis → no H4b consistency error."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["tail"] = {"christoffersen_cc_p": 0.42, "christoffersen_cc_lr": 1.1}
    payload["hypotheses"] = [_h4b_hypothesis()]
    assert hypotheses_include_h4b(payload["hypotheses"]) is True
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h4b_missing_despite_finite_christoffersen_cc_p" not in result["errors"]


def test_verify_nan_christoffersen_cc_p_without_h4b_ok(tmp_path: Path) -> None:
    """NaN christoffersen_cc_p does not require H4b (Day Wave 17 skip-on-nonfinite)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["tail"] = {
        "executed": True,
        "christoffersen_cc_p": float("nan"),
        "christoffersen_cc_lr": float("nan"),
    }
    payload["hypotheses"] = []
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h4b_missing_despite_finite_christoffersen_cc_p" not in result["errors"]


def test_verify_rejects_finite_cc_p_with_h4b_wrong_family(tmp_path: Path) -> None:
    """H4b id present but family != calibration still fails soft consistency."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["tail"] = {"christoffersen_cc_p": 0.2}
    payload["hypotheses"] = [_h4b_hypothesis(family="discovery")]
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h4b_missing_despite_finite_christoffersen_cc_p" in result["errors"]


def _h4_hypothesis(*, family: str = "calibration", p_value: float = 0.35) -> dict:
    """Minimal valid H4 row matching REQUIRED_HYPOTHESIS_FIELDS."""
    return {
        "id": H4_HYPOTHESIS_ID,
        "statement": "Kupiec unconditional coverage.",
        "test": "Kupiec POF",
        "statistic": 1.0,
        "p_value": p_value,
        "reject_raw": False,
        "reject_fdr": False,
        "decision": "VaR hits consistent with independent 5% unconditional coverage.",
        "family": family,
    }


def _tail_finite_kupiec_nan_cc() -> dict:
    """Finite kupiec_p + VaR-battery keys present; NaN cc_p so H4b soft-check skips."""
    return {
        "executed": True,
        "kupiec_p": 0.42,
        "kupiec_lr": 1.1,
        "christoffersen_cc_p": float("nan"),
        "christoffersen_cc_lr": float("nan"),
        "christoffersen_ind_p": float("nan"),
        "christoffersen_ind_lr": float("nan"),
    }


def test_catalog_tail_has_finite_kupiec_p() -> None:
    assert tail_has_finite_kupiec_p({"kupiec_p": 0.4}) is True
    assert tail_has_finite_kupiec_p({"kupiec_p": float("nan")}) is False
    assert tail_has_finite_kupiec_p({"kupiec_p": float("inf")}) is False
    assert tail_has_finite_kupiec_p({}) is False
    assert tail_has_finite_kupiec_p(None) is False


def test_catalog_h4_errors_skip_when_kupiec_p_nonfinite() -> None:
    assert h4_hypothesis_consistency_errors({"kupiec_p": float("nan")}, []) == []
    assert h4_hypothesis_consistency_errors({}, []) == []
    assert h4_hypothesis_consistency_errors(None, []) == []


def test_verify_rejects_finite_kupiec_p_without_h4(tmp_path: Path) -> None:
    """Finite tail.kupiec_p without H4 hyp → soft fail-closed."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    # Include VaR-battery keys (NaN cc) so only the H4 consistency gate fires.
    payload["families"]["tail"] = _tail_finite_kupiec_nan_cc()
    payload["hypotheses"] = []
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h4_missing_despite_finite_kupiec_p" in result["errors"]
    # H4b path unchanged / skipped when cc_p is non-finite.
    assert "hypothesis_h4b_missing_despite_finite_christoffersen_cc_p" not in result["errors"]


def test_verify_accepts_finite_kupiec_p_with_h4(tmp_path: Path) -> None:
    """Finite kupiec_p + calibration H4 hypothesis → no H4 consistency error."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["tail"] = _tail_finite_kupiec_nan_cc()
    payload["hypotheses"] = [_h4_hypothesis()]
    assert hypotheses_include_h4(payload["hypotheses"]) is True
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h4_missing_despite_finite_kupiec_p" not in result["errors"]


def test_verify_nan_kupiec_p_without_h4_ok(tmp_path: Path) -> None:
    """NaN kupiec_p does not require H4 (agent skip-on-nonfinite)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["tail"] = {
        "executed": True,
        "kupiec_p": float("nan"),
        "kupiec_lr": float("nan"),
        "christoffersen_cc_p": float("nan"),
        "christoffersen_cc_lr": float("nan"),
        "christoffersen_ind_p": float("nan"),
        "christoffersen_ind_lr": float("nan"),
    }
    payload["hypotheses"] = []
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h4_missing_despite_finite_kupiec_p" not in result["errors"]


def test_verify_rejects_finite_kupiec_p_with_h4_wrong_family(tmp_path: Path) -> None:
    """H4 id present but family != calibration still fails soft consistency."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["tail"] = _tail_finite_kupiec_nan_cc()
    payload["hypotheses"] = [_h4_hypothesis(family="discovery")]
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h4_missing_despite_finite_kupiec_p" in result["errors"]


def _h3_hypothesis(*, family: str = "discovery", p_value: float = 0.12) -> dict:
    """Minimal valid H3 row matching REQUIRED_HYPOTHESIS_FIELDS."""
    return {
        "id": H3_HYPOTHESIS_ID,
        "statement": "EWMA and rolling 20d vol forecasts have unequal MSE (Diebold–Mariano).",
        "test": "Diebold–Mariano",
        "statistic": 1.5,
        "p_value": p_value,
        "reject_raw": False,
        "reject_fdr": False,
        "decision": "Equal vol-forecast accuracy not rejected.",
        "family": family,
    }


def test_catalog_volatility_has_finite_dm_p() -> None:
    assert volatility_has_finite_dm_p({"dm_p": 0.04}) is True
    assert volatility_has_finite_dm_p({"dm_p": float("nan")}) is False
    assert volatility_has_finite_dm_p({"dm_p": float("inf")}) is False
    assert volatility_has_finite_dm_p({}) is False
    assert volatility_has_finite_dm_p(None) is False


def test_catalog_h3_errors_skip_when_dm_p_nonfinite() -> None:
    assert h3_hypothesis_consistency_errors({"dm_p": float("nan")}, []) == []
    assert h3_hypothesis_consistency_errors({}, []) == []
    assert h3_hypothesis_consistency_errors(None, []) == []


def test_verify_rejects_finite_dm_p_without_h3(tmp_path: Path) -> None:
    """Finite volatility.dm_p without H3 hyp → soft fail-closed."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["volatility"] = {"dm_p": 0.03, "dm_stat": 2.1, "dm_preferred": "ewma"}
    payload["hypotheses"] = []
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h3_missing_despite_finite_dm_p" in result["errors"]


def test_verify_accepts_finite_dm_p_with_h3(tmp_path: Path) -> None:
    """Finite dm_p + discovery H3 hypothesis → no H3 consistency error."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["volatility"] = {"dm_p": 0.03, "dm_stat": 2.1, "dm_preferred": "ewma"}
    payload["hypotheses"] = [_h3_hypothesis()]
    assert hypotheses_include_h3(payload["hypotheses"]) is True
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h3_missing_despite_finite_dm_p" not in result["errors"]


def test_verify_nan_dm_p_without_h3_ok(tmp_path: Path) -> None:
    """NaN dm_p does not require H3 (agent skip-on-nonfinite)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["volatility"] = {
        "executed": True,
        "dm_p": float("nan"),
        "dm_stat": float("nan"),
    }
    payload["hypotheses"] = []
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h3_missing_despite_finite_dm_p" not in result["errors"]


def test_verify_rejects_finite_dm_p_with_h3_wrong_family(tmp_path: Path) -> None:
    """H3 id present but family != discovery still fails soft consistency."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["volatility"] = {"dm_p": 0.03}
    payload["hypotheses"] = [_h3_hypothesis(family="calibration")]
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h3_missing_despite_finite_dm_p" in result["errors"]


def _h1_hypothesis(*, family: str = "discovery", p_value: float = 0.04) -> dict:
    """Minimal valid H1 row matching REQUIRED_HYPOTHESIS_FIELDS."""
    return {
        "id": H1_HYPOTHESIS_ID,
        "statement": "Labeled SYNTHETIC oracle has positive date-level ranking IC.",
        "test": "HAC t-stat of date IC",
        "statistic": 2.0,
        "p_value": p_value,
        "reject_raw": True,
        "reject_fdr": False,
        "decision": "Oracle ranking signal recovered.",
        "family": family,
    }


def _h2_hypothesis(*, family: str = "discovery", p_value: float = 0.05) -> dict:
    """Minimal valid H2 row matching REQUIRED_HYPOTHESIS_FIELDS."""
    return {
        "id": H2_HYPOTHESIS_ID,
        "statement": "Oracle decile long-short mean is positive (ranking science).",
        "test": "HAC t-stat of decile LS",
        "statistic": 1.8,
        "p_value": p_value,
        "reject_raw": True,
        "reject_fdr": False,
        "decision": "Deciles are informative.",
        "family": family,
    }


def _oracle_raw(*, p_ic: float = 0.04, ls_p: float = 0.05) -> dict:
    return {
        "name": "oracle_raw",
        "p_ic": p_ic,
        "t_ic": 2.0,
        "ls_p": ls_p,
        "ls_t": 1.8,
    }


def test_catalog_rankers_oracle_raw_skips_underscore() -> None:
    assert rankers_oracle_raw(None) is None
    assert rankers_oracle_raw([]) is None
    assert rankers_oracle_raw([{"name": "_oracle_raw", "p_ic": 0.01}]) is None
    row = {"name": "oracle_raw", "p_ic": 0.02}
    assert rankers_oracle_raw([{"name": "_meta"}, row]) is row


def test_catalog_oracle_has_finite_p_ic_and_ls_p() -> None:
    assert oracle_has_finite_p_ic({"p_ic": 0.04}) is True
    assert oracle_has_finite_p_ic({"p_ic": float("nan")}) is False
    assert oracle_has_finite_p_ic({"p_ic": float("inf")}) is False
    assert oracle_has_finite_p_ic({}) is False
    assert oracle_has_finite_p_ic(None) is False
    assert oracle_has_finite_ls_p({"ls_p": 0.05}) is True
    assert oracle_has_finite_ls_p({"ls_p": float("nan")}) is False
    assert oracle_has_finite_ls_p({}) is False
    assert oracle_has_finite_ls_p(None) is False


def test_catalog_h1_h2_errors_skip_when_nonfinite_or_missing() -> None:
    assert h1_hypothesis_consistency_errors([], []) == []
    assert (
        h1_hypothesis_consistency_errors([{"name": "oracle_raw", "p_ic": float("nan")}], []) == []
    )
    assert h1_hypothesis_consistency_errors([{"name": "oracle_raw"}], []) == []
    assert h1_hypothesis_consistency_errors(None, []) == []
    assert h2_hypothesis_consistency_errors([], []) == []
    assert (
        h2_hypothesis_consistency_errors([{"name": "oracle_raw", "ls_p": float("nan")}], []) == []
    )
    assert h2_hypothesis_consistency_errors([{"name": "oracle_raw"}], []) == []
    assert h2_hypothesis_consistency_errors(None, []) == []


def test_verify_rejects_finite_p_ic_without_h1(tmp_path: Path) -> None:
    """Finite oracle_raw.p_ic without H1 hyp → soft fail-closed."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["rankers"] = [_oracle_raw(p_ic=0.03, ls_p=float("nan"))]
    payload["hypotheses"] = []
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h1_missing_despite_finite_p_ic" in result["errors"]
    assert "hypothesis_h2_missing_despite_finite_ls_p" not in result["errors"]


def test_verify_accepts_finite_p_ic_with_h1(tmp_path: Path) -> None:
    """Finite p_ic + discovery H1 hypothesis → no H1 consistency error."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["rankers"] = [_oracle_raw(p_ic=0.03, ls_p=float("nan"))]
    payload["hypotheses"] = [_h1_hypothesis()]
    assert hypotheses_include_h1(payload["hypotheses"]) is True
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h1_missing_despite_finite_p_ic" not in result["errors"]


def test_verify_nan_p_ic_without_h1_ok(tmp_path: Path) -> None:
    """NaN p_ic does not require H1 (agent skip-on-nonfinite)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["rankers"] = [_oracle_raw(p_ic=float("nan"), ls_p=float("nan"))]
    payload["hypotheses"] = []
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h1_missing_despite_finite_p_ic" not in result["errors"]


def test_verify_rejects_finite_p_ic_with_h1_wrong_family(tmp_path: Path) -> None:
    """H1 id present but family != discovery still fails soft consistency."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["rankers"] = [_oracle_raw(p_ic=0.03, ls_p=float("nan"))]
    payload["hypotheses"] = [_h1_hypothesis(family="calibration")]
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h1_missing_despite_finite_p_ic" in result["errors"]


def test_verify_rejects_finite_ls_p_without_h2(tmp_path: Path) -> None:
    """Finite oracle_raw.ls_p without H2 hyp → soft fail-closed."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["rankers"] = [_oracle_raw(p_ic=float("nan"), ls_p=0.04)]
    payload["hypotheses"] = []
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h2_missing_despite_finite_ls_p" in result["errors"]
    assert "hypothesis_h1_missing_despite_finite_p_ic" not in result["errors"]


def test_verify_accepts_finite_ls_p_with_h2(tmp_path: Path) -> None:
    """Finite ls_p + discovery H2 hypothesis → no H2 consistency error."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["rankers"] = [_oracle_raw(p_ic=float("nan"), ls_p=0.04)]
    payload["hypotheses"] = [_h2_hypothesis()]
    assert hypotheses_include_h2(payload["hypotheses"]) is True
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h2_missing_despite_finite_ls_p" not in result["errors"]


def test_verify_nan_ls_p_without_h2_ok(tmp_path: Path) -> None:
    """NaN ls_p does not require H2 (agent skip-on-nonfinite)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["rankers"] = [_oracle_raw(p_ic=float("nan"), ls_p=float("nan"))]
    payload["hypotheses"] = []
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h2_missing_despite_finite_ls_p" not in result["errors"]


def test_verify_rejects_finite_ls_p_with_h2_wrong_family(tmp_path: Path) -> None:
    """H2 id present but family != discovery still fails soft consistency."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["rankers"] = [_oracle_raw(p_ic=float("nan"), ls_p=0.04)]
    payload["hypotheses"] = [_h2_hypothesis(family="calibration")]
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h2_missing_despite_finite_ls_p" in result["errors"]


def test_verify_h1_h2_gates_independent(tmp_path: Path) -> None:
    """Both finite without hyps → both errors; both with hyps → ok."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["rankers"] = [_oracle_raw(p_ic=0.02, ls_p=0.03)]
    payload["hypotheses"] = []
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h1_missing_despite_finite_p_ic" in result["errors"]
    assert "hypothesis_h2_missing_despite_finite_ls_p" in result["errors"]

    payload["hypotheses"] = [_h1_hypothesis(), _h2_hypothesis()]
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable.write_text(json.dumps(payload))
    result2 = verify_research_artifact(path)
    assert result2["valid"] is True
    assert result2["errors"] == []


def _h7_hypothesis(*, family: str = "calibration", p_value: float = 0.22) -> dict:
    """Minimal valid H7 row matching REQUIRED_HYPOTHESIS_FIELDS."""
    return {
        "id": H7_HYPOTHESIS_ID,
        "statement": "ACI 90% prediction-set miss rate matches nominal alpha=0.10.",
        "test": "Kupiec POF on ACI misses",
        "statistic": 1.1,
        "p_value": p_value,
        "reject_raw": False,
        "reject_fdr": False,
        "decision": "ACI coverage consistent with 90% nominal.",
        "family": family,
    }


def _conformal_aci(*, kupiec_p: float = 0.22, kupiec_lr: float = 1.1) -> dict:
    return {"executed": True, "aci": {"kupiec_p": kupiec_p, "kupiec_lr": kupiec_lr}}


def test_catalog_aci_has_finite_kupiec_p() -> None:
    assert aci_has_finite_kupiec_p({"kupiec_p": 0.22}) is True
    assert aci_has_finite_kupiec_p({"kupiec_p": float("nan")}) is False
    assert aci_has_finite_kupiec_p({"kupiec_p": float("inf")}) is False
    assert aci_has_finite_kupiec_p({}) is False
    assert aci_has_finite_kupiec_p(None) is False
    assert conformal_aci_blob(None) is None
    assert conformal_aci_blob({}) is None
    assert conformal_aci_blob({"aci": "nope"}) is None
    row = {"kupiec_p": 0.1}
    assert conformal_aci_blob({"aci": row}) is row


def test_catalog_h7_errors_skip_when_kupiec_p_nonfinite() -> None:
    assert h7_hypothesis_consistency_errors({"aci": {"kupiec_p": float("nan")}}, []) == []
    assert h7_hypothesis_consistency_errors({"aci": {}}, []) == []
    assert h7_hypothesis_consistency_errors({}, []) == []
    assert h7_hypothesis_consistency_errors(None, []) == []


def test_verify_rejects_finite_aci_kupiec_p_without_h7(tmp_path: Path) -> None:
    """Finite conformal.aci.kupiec_p without H7 hyp → soft fail-closed."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["conformal"] = _conformal_aci(kupiec_p=0.18)
    payload["hypotheses"] = []
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h7_missing_despite_finite_aci_kupiec_p" in result["errors"]


def test_verify_accepts_finite_aci_kupiec_p_with_h7(tmp_path: Path) -> None:
    """Finite ACI kupiec_p + calibration H7 → no H7 consistency error."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["conformal"] = _conformal_aci(kupiec_p=0.18)
    payload["hypotheses"] = [_h7_hypothesis()]
    assert hypotheses_include_h7(payload["hypotheses"]) is True
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h7_missing_despite_finite_aci_kupiec_p" not in result["errors"]


def test_verify_nan_aci_kupiec_p_without_h7_ok(tmp_path: Path) -> None:
    """NaN ACI kupiec_p does not require H7 (agent skip-on-nonfinite)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["conformal"] = _conformal_aci(kupiec_p=float("nan"))
    payload["hypotheses"] = []
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h7_missing_despite_finite_aci_kupiec_p" not in result["errors"]


def test_verify_rejects_finite_aci_kupiec_p_with_h7_wrong_family(tmp_path: Path) -> None:
    """H7 id present but family != calibration still fails soft consistency."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["conformal"] = _conformal_aci(kupiec_p=0.18)
    payload["hypotheses"] = [_h7_hypothesis(family="discovery")]
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h7_missing_despite_finite_aci_kupiec_p" in result["errors"]


def _h8_hypothesis(*, family: str = "calibration", p_value: float = 0.45) -> dict:
    return {
        "id": H8_HYPOTHESIS_ID,
        "statement": "Mondrian ACI high-vol slice miss rate matches nominal alpha=0.10.",
        "test": "Kupiec POF on Mondrian high-X misses",
        "statistic": 0.8,
        "p_value": p_value,
        "reject_raw": False,
        "reject_fdr": False,
        "decision": "High-vol Mondrian coverage consistent with 90% nominal.",
        "family": family,
    }


def _conformal_mondrian_aci(
    *, high_x_kupiec_p: float = 0.45, high_x_kupiec_lr: float = 0.8
) -> dict:
    return {
        "mondrian_aci": {
            "high_x_kupiec_p": high_x_kupiec_p,
            "high_x_kupiec_lr": high_x_kupiec_lr,
        }
    }


def test_catalog_mondrian_aci_has_finite_high_x_kupiec_p() -> None:
    assert mondrian_aci_has_finite_high_x_kupiec_p({"high_x_kupiec_p": 0.45}) is True
    assert mondrian_has_finite_high_x_kupiec_p({"high_x_kupiec_p": 0.45}) is True
    assert mondrian_aci_has_finite_high_x_kupiec_p({"high_x_kupiec_p": float("nan")}) is False
    assert mondrian_aci_has_finite_high_x_kupiec_p({"high_x_kupiec_p": float("inf")}) is False
    assert mondrian_aci_has_finite_high_x_kupiec_p({}) is False
    assert mondrian_aci_has_finite_high_x_kupiec_p(None) is False
    assert conformal_mondrian_aci_blob(None) is None
    assert conformal_mondrian_aci_blob({}) is None
    assert conformal_mondrian_aci_blob({"mondrian_aci": "nope"}) is None
    row = {"high_x_kupiec_p": 0.2}
    assert conformal_mondrian_aci_blob({"mondrian_aci": row}) is row
    assert conformal_mondrian_aci_payload({"mondrian_aci": row}) is row


def test_catalog_h8_errors_skip_when_high_x_kupiec_p_nonfinite() -> None:
    assert (
        h8_hypothesis_consistency_errors({"mondrian_aci": {"high_x_kupiec_p": float("nan")}}, [])
        == []
    )
    assert h8_hypothesis_consistency_errors({"mondrian_aci": {}}, []) == []
    assert h8_hypothesis_consistency_errors({}, []) == []
    assert h8_hypothesis_consistency_errors(None, []) == []


def test_verify_rejects_finite_high_x_kupiec_p_without_h8(tmp_path: Path) -> None:
    """Finite mondrian_aci.high_x_kupiec_p without H8 hyp → soft fail-closed."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["conformal"] = _conformal_mondrian_aci(high_x_kupiec_p=0.33)
    payload["hypotheses"] = []
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h8_missing_despite_finite_high_x_kupiec_p" in result["errors"]


def test_verify_accepts_finite_high_x_kupiec_p_with_h8(tmp_path: Path) -> None:
    """Finite high_x_kupiec_p + calibration H8 → no H8 consistency error."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["conformal"] = _conformal_mondrian_aci(high_x_kupiec_p=0.33)
    payload["hypotheses"] = [_h8_hypothesis()]
    assert hypotheses_include_h8(payload["hypotheses"]) is True
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h8_missing_despite_finite_high_x_kupiec_p" not in result["errors"]


def test_verify_nan_high_x_kupiec_p_without_h8_ok(tmp_path: Path) -> None:
    """NaN high_x_kupiec_p does not require H8 (agent skip-on-nonfinite)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["conformal"] = _conformal_mondrian_aci(high_x_kupiec_p=float("nan"))
    payload["hypotheses"] = []
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h8_missing_despite_finite_high_x_kupiec_p" not in result["errors"]


def test_verify_rejects_finite_high_x_kupiec_p_with_h8_wrong_family(tmp_path: Path) -> None:
    """H8 id present but family != calibration still fails soft consistency."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["conformal"] = _conformal_mondrian_aci(high_x_kupiec_p=0.33)
    payload["hypotheses"] = [_h8_hypothesis(family="discovery")]
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h8_missing_despite_finite_high_x_kupiec_p" in result["errors"]


def _h11_hypothesis(*, family: str = "calibration", p_value: float = 0.33) -> dict:
    """Minimal valid H11 row matching REQUIRED_HYPOTHESIS_FIELDS."""
    return {
        "id": H11_HYPOTHESIS_ID,
        "statement": "CRC 95% VaR-style hit rate matches the 5% nominal (Kupiec POF).",
        "test": "Kupiec POF on CRC hits",
        "statistic": 1.1,
        "p_value": p_value,
        "reject_raw": False,
        "reject_fdr": False,
        "decision": "CRC hits consistent with 5% nominal.",
        "family": family,
    }


def test_catalog_crc_has_finite_kupiec_p() -> None:
    assert crc_has_finite_kupiec_p({"kupiec_p": 0.33}) is True
    assert crc_has_finite_kupiec_p({"kupiec_p": float("nan")}) is False
    assert crc_has_finite_kupiec_p({"kupiec_p": float("inf")}) is False
    assert crc_has_finite_kupiec_p({}) is False
    assert crc_has_finite_kupiec_p(None) is False


def test_catalog_h11_crc_consistency_skips_nonfinite() -> None:
    assert h11_hypothesis_consistency_errors({"kupiec_p": float("nan")}, []) == []
    assert h11_hypothesis_consistency_errors({}, []) == []
    assert h11_hypothesis_consistency_errors(None, []) == []


def test_verify_rejects_finite_crc_kupiec_p_without_h11(tmp_path: Path) -> None:
    """Finite crc.kupiec_p without H11 hyp → soft fail-closed."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["crc"] = {"kupiec_p": 0.33}
    payload["hypotheses"] = []
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h11_missing_despite_finite_crc_kupiec_p" in result["errors"]


def test_verify_accepts_finite_crc_kupiec_p_with_h11(tmp_path: Path) -> None:
    """Finite CRC kupiec_p + calibration H11 → no H11 consistency error."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["crc"] = {"kupiec_p": 0.33}
    payload["hypotheses"] = [_h11_hypothesis()]
    assert hypotheses_include_h11(payload["hypotheses"]) is True
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h11_missing_despite_finite_crc_kupiec_p" not in result["errors"]


def test_verify_nan_crc_kupiec_p_without_h11_ok(tmp_path: Path) -> None:
    """NaN CRC kupiec_p does not require H11 (agent skip-on-nonfinite)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["crc"] = {"executed": True, "kupiec_p": float("nan")}
    payload["hypotheses"] = []
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h11_missing_despite_finite_crc_kupiec_p" not in result["errors"]


def test_verify_rejects_finite_crc_kupiec_p_with_h11_wrong_family(tmp_path: Path) -> None:
    """H11 id present but family != calibration still fails soft consistency."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["crc"] = {"kupiec_p": 0.33}
    payload["hypotheses"] = [_h11_hypothesis(family="discovery")]
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h11_missing_despite_finite_crc_kupiec_p" in result["errors"]


def _h12_hypothesis(*, family: str = "calibration", p_value: float = 0.41) -> dict:
    """Minimal valid H12 row matching REQUIRED_HYPOTHESIS_FIELDS."""
    return {
        "id": H12_HYPOTHESIS_ID,
        "statement": "Weighted split CQR miss rate matches nominal α=0.10 (Kupiec POF).",
        "test": "Kupiec POF on weighted CQR misses",
        "statistic": 1.2,
        "p_value": p_value,
        "reject_raw": False,
        "reject_fdr": False,
        "decision": "Weighted CQR coverage consistent with 90% nominal.",
        "family": family,
    }


def test_catalog_weighted_conformal_has_finite_kupiec_p() -> None:
    assert weighted_conformal_has_finite_kupiec_p({"kupiec_p": 0.41}) is True
    assert weighted_conformal_has_finite_kupiec_p({"kupiec_p": float("nan")}) is False
    assert weighted_conformal_has_finite_kupiec_p({"kupiec_p": float("inf")}) is False
    assert weighted_conformal_has_finite_kupiec_p({}) is False
    assert weighted_conformal_has_finite_kupiec_p(None) is False


def test_catalog_h12_wcqr_consistency_skips_nonfinite() -> None:
    assert h12_hypothesis_consistency_errors({"kupiec_p": float("nan")}, []) == []
    assert h12_hypothesis_consistency_errors({}, []) == []
    assert h12_hypothesis_consistency_errors(None, []) == []


def test_verify_rejects_finite_wcqr_kupiec_p_without_h12(tmp_path: Path) -> None:
    """Finite weighted_conformal.kupiec_p without H12 hyp → soft fail-closed."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["weighted_conformal"] = {"kupiec_p": 0.41}
    payload["hypotheses"] = []
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h12_missing_despite_finite_wcqr_kupiec_p" in result["errors"]


def test_verify_accepts_finite_wcqr_kupiec_p_with_h12(tmp_path: Path) -> None:
    """Finite weighted CQR kupiec_p + calibration H12 → no H12 consistency error."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["weighted_conformal"] = {"kupiec_p": 0.41}
    payload["hypotheses"] = [_h12_hypothesis()]
    assert hypotheses_include_h12(payload["hypotheses"]) is True
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h12_missing_despite_finite_wcqr_kupiec_p" not in result["errors"]


def test_verify_nan_wcqr_kupiec_p_without_h12_ok(tmp_path: Path) -> None:
    """NaN weighted CQR kupiec_p does not require H12 (agent skip-on-nonfinite)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["weighted_conformal"] = {
        "executed": True,
        "kupiec_p": float("nan"),
    }
    payload["hypotheses"] = []
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h12_missing_despite_finite_wcqr_kupiec_p" not in result["errors"]


def test_verify_rejects_finite_wcqr_kupiec_p_with_h12_wrong_family(tmp_path: Path) -> None:
    """H12 id present but family != calibration still fails soft consistency."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["weighted_conformal"] = {"kupiec_p": 0.41}
    payload["hypotheses"] = [_h12_hypothesis(family="discovery")]
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h12_missing_despite_finite_wcqr_kupiec_p" in result["errors"]


def _h9_hypothesis(*, family: str = "calibration", p_value: float = 0.25) -> dict:
    """Minimal valid H9 row matching REQUIRED_HYPOTHESIS_FIELDS."""
    return {
        "id": H9_HYPOTHESIS_ID,
        "statement": "ACI miss e-process is consistent with nominal α=0.10 (Ville, level 0.05).",
        "test": "Anytime-valid Bernoulli e-process (Ville)",
        "statistic": 2.0,
        "p_value": p_value,
        "reject_raw": False,
        "reject_fdr": False,
        "decision": "E-process stayed below 20; ACI misses consistent with α=0.10.",
        "family": family,
    }


def test_catalog_evalues_has_finite_e_sup() -> None:
    assert evalues_has_finite_e_sup({"e_sup": 2.0}) is True
    assert evalues_has_finite_e_sup({"e_sup": float("nan")}) is False
    assert evalues_has_finite_e_sup({"e_sup": float("inf")}) is False
    assert evalues_has_finite_e_sup({}) is False
    assert evalues_has_finite_e_sup(None) is False


def test_catalog_h9_eprocess_consistency_skips_nonfinite() -> None:
    assert h9_hypothesis_consistency_errors({"e_sup": float("nan")}, []) == []
    assert h9_hypothesis_consistency_errors({}, []) == []
    assert h9_hypothesis_consistency_errors(None, []) == []


def test_verify_rejects_finite_e_sup_without_h9(tmp_path: Path) -> None:
    """Finite evalues.e_sup without H9 hyp → soft fail-closed."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["evalues"] = {"e_sup": 2.0}
    payload["hypotheses"] = []
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h9_missing_despite_finite_e_sup" in result["errors"]


def test_verify_accepts_finite_e_sup_with_h9(tmp_path: Path) -> None:
    """Finite e_sup + calibration H9 → no H9 consistency error."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["evalues"] = {"e_sup": 2.0}
    payload["hypotheses"] = [_h9_hypothesis()]
    assert hypotheses_include_h9(payload["hypotheses"]) is True
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h9_missing_despite_finite_e_sup" not in result["errors"]


def test_verify_nan_e_sup_without_h9_ok(tmp_path: Path) -> None:
    """NaN e_sup does not require H9 (agent skip-on-nonfinite)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["evalues"] = {"executed": True, "e_sup": float("nan")}
    payload["hypotheses"] = []
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h9_missing_despite_finite_e_sup" not in result["errors"]


def test_verify_rejects_finite_e_sup_with_h9_wrong_family(tmp_path: Path) -> None:
    """H9 id present but family != calibration still fails soft consistency."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["evalues"] = {"e_sup": 2.0}
    payload["hypotheses"] = [_h9_hypothesis(family="discovery")]
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h9_missing_despite_finite_e_sup" in result["errors"]


def _h10_hypothesis(*, family: str = "bound", p_value: float = float("nan")) -> dict:
    """Minimal valid H10 row matching REQUIRED_HYPOTHESIS_FIELDS (family=bound)."""
    return {
        "id": H10_HYPOTHESIS_ID,
        "statement": "Jackknife+ coverage meets the 1-2α finite-sample floor.",
        "test": "coverage − (1−2α) floor check; not a Kupiec null, not FDR",
        "statistic": 0.05,
        "p_value": p_value,
        "reject_raw": False,
        "reject_fdr": False,
        "decision": "Jackknife+ coverage is above the 1-2α floor.",
        "family": family,
    }


def test_catalog_jackknife_plus_has_finite_coverage() -> None:
    assert jackknife_plus_has_finite_coverage({"coverage": 0.85}) is True
    assert jackknife_plus_has_finite_coverage({"coverage": float("nan")}) is False
    assert jackknife_plus_has_finite_coverage({"coverage": float("inf")}) is False
    assert jackknife_plus_has_finite_coverage({}) is False
    assert jackknife_plus_has_finite_coverage(None) is False


def test_catalog_h10_jp_consistency_skips_nonfinite() -> None:
    assert h10_hypothesis_consistency_errors({"coverage": float("nan")}, []) == []
    assert h10_hypothesis_consistency_errors({}, []) == []
    assert h10_hypothesis_consistency_errors(None, []) == []


def test_verify_rejects_finite_jp_coverage_without_h10(tmp_path: Path) -> None:
    """Finite jackknife_plus.coverage without H10 hyp → soft fail-closed."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["jackknife_plus"] = {
        "coverage": 0.85,
        "coverage_guarantee_scope": "marginal_exchangeable",
    }
    payload["hypotheses"] = []
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h10_missing_despite_finite_coverage" in result["errors"]


def test_verify_accepts_finite_jp_coverage_with_h10(tmp_path: Path) -> None:
    """Finite coverage + bound H10 → no H10 consistency error."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["jackknife_plus"] = {
        "coverage": 0.85,
        "coverage_guarantee_scope": "marginal_exchangeable",
    }
    payload["hypotheses"] = [_h10_hypothesis()]
    assert hypotheses_include_h10(payload["hypotheses"]) is True
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h10_missing_despite_finite_coverage" not in result["errors"]


def test_verify_nan_jp_coverage_without_h10_ok(tmp_path: Path) -> None:
    """NaN coverage does not require H10 (agent skip-on-nonfinite)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["jackknife_plus"] = {
        "coverage": float("nan"),
        "coverage_guarantee_scope": "marginal_exchangeable",
    }
    payload["hypotheses"] = []
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h10_missing_despite_finite_coverage" not in result["errors"]


def test_verify_rejects_finite_jp_coverage_with_h10_wrong_family(tmp_path: Path) -> None:
    """H10 id present but family != bound still fails soft consistency."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["jackknife_plus"] = {
        "coverage": 0.85,
        "coverage_guarantee_scope": "marginal_exchangeable",
    }
    payload["hypotheses"] = [_h10_hypothesis(family="calibration")]
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h10_missing_despite_finite_coverage" in result["errors"]


def _h15_hypothesis(*, family: str = "bound", p_value: float = float("nan")) -> dict:
    """Minimal valid H15 row matching REQUIRED_HYPOTHESIS_FIELDS (family=bound)."""
    return {
        "id": H15_HYPOTHESIS_ID,
        "statement": "CV+ coverage meets its aggregation-specific finite-sample floor.",
        "test": "coverage − floor check; not FDR",
        "statistic": 0.05,
        "p_value": p_value,
        "reject_raw": False,
        "reject_fdr": False,
        "decision": "CV+ coverage is above its stated floor.",
        "family": family,
    }


def test_catalog_cv_plus_has_finite_coverage_and_floor() -> None:
    assert cv_plus_has_finite_coverage_and_floor({"coverage": 0.90, "coverage_floor": 0.80}) is True
    assert (
        cv_plus_has_finite_coverage_and_floor({"coverage": 0.90, "coverage_floor": float("nan")})
        is False
    )
    assert (
        cv_plus_has_finite_coverage_and_floor({"coverage": float("nan"), "coverage_floor": 0.80})
        is False
    )
    assert cv_plus_has_finite_coverage_and_floor({"coverage": 0.90}) is False
    assert cv_plus_has_finite_coverage_and_floor({}) is False
    assert cv_plus_has_finite_coverage_and_floor(None) is False


def test_catalog_h15_cv_plus_consistency_skips_nonfinite() -> None:
    assert (
        h15_hypothesis_consistency_errors({"coverage": float("nan"), "coverage_floor": 0.8}, [])
        == []
    )
    assert (
        h15_hypothesis_consistency_errors({"coverage": 0.9, "coverage_floor": float("nan")}, [])
        == []
    )
    assert h15_hypothesis_consistency_errors({"coverage": 0.9}, []) == []
    assert h15_hypothesis_consistency_errors({}, []) == []
    assert h15_hypothesis_consistency_errors(None, []) == []


def test_verify_rejects_finite_cv_plus_coverage_without_h15(tmp_path: Path) -> None:
    """Finite cv_plus coverage+floor without H15 hyp → soft fail-closed."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["cv_plus"] = {
        "coverage": 0.90,
        "coverage_floor": 0.80,
        "coverage_guarantee_scope": "marginal_exchangeable",
    }
    payload["hypotheses"] = []
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h15_missing_despite_finite_coverage_and_floor" in result["errors"]


def test_verify_accepts_finite_cv_plus_coverage_with_h15(tmp_path: Path) -> None:
    """Finite coverage+floor + bound H15 → no H15 consistency error."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["cv_plus"] = {
        "coverage": 0.90,
        "coverage_floor": 0.80,
        "coverage_guarantee_scope": "marginal_exchangeable",
    }
    payload["hypotheses"] = [_h15_hypothesis()]
    assert hypotheses_include_h15(payload["hypotheses"]) is True
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h15_missing_despite_finite_coverage_and_floor" not in result["errors"]


def test_verify_nan_cv_plus_coverage_without_h15_ok(tmp_path: Path) -> None:
    """NaN coverage does not require H15 (agent skip when either nonfinite)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["cv_plus"] = {
        "coverage": float("nan"),
        "coverage_floor": 0.80,
        "coverage_guarantee_scope": "marginal_exchangeable",
    }
    payload["hypotheses"] = []
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h15_missing_despite_finite_coverage_and_floor" not in result["errors"]


def test_verify_nan_cv_plus_floor_without_h15_ok(tmp_path: Path) -> None:
    """NaN coverage_floor does not require H15 (agent requires both finite)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["cv_plus"] = {
        "coverage": 0.90,
        "coverage_floor": float("nan"),
        "coverage_guarantee_scope": "marginal_exchangeable",
    }
    payload["hypotheses"] = []
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h15_missing_despite_finite_coverage_and_floor" not in result["errors"]


def test_verify_rejects_finite_cv_plus_with_h15_wrong_family(tmp_path: Path) -> None:
    """H15 id present but family != bound still fails soft consistency."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["cv_plus"] = {
        "coverage": 0.90,
        "coverage_floor": 0.80,
        "coverage_guarantee_scope": "marginal_exchangeable",
    }
    payload["hypotheses"] = [_h15_hypothesis(family="calibration")]
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h15_missing_despite_finite_coverage_and_floor" in result["errors"]


def _h16_hypothesis(*, family: str = "calibration", p_value: float = 0.48) -> dict:
    """Minimal valid H16 row matching REQUIRED_HYPOTHESIS_FIELDS."""
    return {
        "id": H16_HYPOTHESIS_ID,
        "statement": "Localized CQR miss rate matches nominal alpha on the lab panel.",
        "test": "Kupiec POF vs nominal alpha (panel)",
        "statistic": 0.5,
        "p_value": p_value,
        "reject_raw": False,
        "reject_fdr": False,
        "decision": "localized_conformal coverage consistent with nominal alpha.",
        "family": family,
    }


def _h17_hypothesis(*, family: str = "calibration", p_value: float = 0.65) -> dict:
    """Minimal valid H17 row matching REQUIRED_HYPOTHESIS_FIELDS."""
    return {
        "id": H17_HYPOTHESIS_ID,
        "statement": "Online CRC hit risk matches nominal alpha on the lab panel.",
        "test": "Kupiec POF vs nominal alpha (panel)",
        "statistic": 0.2,
        "p_value": p_value,
        "reject_raw": False,
        "reject_fdr": False,
        "decision": "online_crc coverage consistent with nominal alpha.",
        "family": family,
    }


def _h18_hypothesis(*, family: str = "calibration", p_value: float = 0.75) -> dict:
    """Minimal valid H18 row matching REQUIRED_HYPOTHESIS_FIELDS."""
    return {
        "id": H18_HYPOTHESIS_ID,
        "statement": "Book-level conformal coverage matches nominal alpha on the lab panel.",
        "test": "Kupiec POF vs nominal alpha (panel)",
        "statistic": 0.1,
        "p_value": p_value,
        "reject_raw": False,
        "reject_fdr": False,
        "decision": "portfolio_conformal coverage consistent with nominal alpha.",
        "family": family,
    }


def test_catalog_panel_family_has_finite_kupiec_p() -> None:
    assert panel_family_has_finite_kupiec_p({"kupiec_p": 0.48, "dgp": "panel"}) is True
    assert panel_family_has_finite_kupiec_p({"kupiec_p": 0.48}) is True  # missing dgp ok
    assert panel_family_has_finite_kupiec_p({"kupiec_p": 0.48, "dgp": "fixture"}) is False
    assert panel_family_has_finite_kupiec_p({"kupiec_p": float("nan")}) is False
    assert panel_family_has_finite_kupiec_p({"kupiec_p": float("inf")}) is False
    assert panel_family_has_finite_kupiec_p({}) is False
    assert panel_family_has_finite_kupiec_p(None) is False
    assert panel_family_has_finite_kupiec_p({"date_clustered_p": 0.48, "dgp": "panel"}) is True
    assert (
        panel_family_has_finite_kupiec_p(
            {"date_clustered_p": float("nan"), "kupiec_p": float("nan"), "dgp": "panel"}
        )
        is False
    )


def test_catalog_h16_h18_consistency_skips_nonfinite_and_fixture() -> None:
    assert (
        h16_h18_panel_kupiec_consistency_errors(
            {"localized_conformal": {"kupiec_p": float("nan")}}, []
        )
        == []
    )
    assert (
        h16_h18_panel_kupiec_consistency_errors(
            {"localized_conformal": {"kupiec_p": 0.5, "dgp": "fixture"}}, []
        )
        == []
    )
    assert h16_h18_panel_kupiec_consistency_errors({}, []) == []
    assert h16_h18_panel_kupiec_consistency_errors(None, []) == []


def test_catalog_h16_h18_requires_hypothesis_for_clustered_p() -> None:
    errors = h16_h18_panel_kupiec_consistency_errors(
        {"localized_conformal": {"date_clustered_p": 0.5, "dgp": "panel"}}, []
    )
    assert errors == ["hypothesis_h16_missing_despite_finite_kupiec_p"]


def test_verify_rejects_finite_panel_kupiec_p_without_h16_h18(tmp_path: Path) -> None:
    """Finite panel kupiec_p without H16/H17/H18 → soft fail-closed (batch)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["localized_conformal"] = {"kupiec_p": 0.48, "dgp": "panel"}
    payload["families"]["online_crc"] = {"kupiec_p": 0.65, "dgp": "panel"}
    payload["families"]["portfolio_conformal"] = {"kupiec_p": 0.75, "dgp": "panel"}
    payload["hypotheses"] = []
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h16_missing_despite_finite_kupiec_p" in result["errors"]
    assert "hypothesis_h17_missing_despite_finite_kupiec_p" in result["errors"]
    assert "hypothesis_h18_missing_despite_finite_kupiec_p" in result["errors"]


def test_verify_accepts_finite_panel_kupiec_p_with_h16_h18(tmp_path: Path) -> None:
    """Finite panel kupiec_p + calibration H16/H17/H18 → no panel consistency errors."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["localized_conformal"] = {"kupiec_p": 0.48, "dgp": "panel"}
    payload["families"]["online_crc"] = {"kupiec_p": 0.65, "dgp": "panel"}
    payload["families"]["portfolio_conformal"] = {"kupiec_p": 0.75, "dgp": "panel"}
    payload["hypotheses"] = [_h16_hypothesis(), _h17_hypothesis(), _h18_hypothesis()]
    assert hypotheses_include_h16(payload["hypotheses"]) is True
    assert hypotheses_include_h17(payload["hypotheses"]) is True
    assert hypotheses_include_h18(payload["hypotheses"]) is True
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h16_missing_despite_finite_kupiec_p" not in result["errors"]
    assert "hypothesis_h17_missing_despite_finite_kupiec_p" not in result["errors"]
    assert "hypothesis_h18_missing_despite_finite_kupiec_p" not in result["errors"]


def test_verify_rejects_latest_json_pointer_mismatch(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["artifacts"]["json"] = "other-latest.json"
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "latest_json_artifact_pointer_mismatch" in result["errors"]


def test_verify_rejects_malformed_present_artifact_fields(tmp_path: Path) -> None:
    receipt_path = _receipt(tmp_path)
    payload = json.loads(receipt_path.read_text())
    payload["artifacts"]["json"] = {"path": "latest.json"}
    payload["artifacts"]["immutable_markdown_sha256"] = "not-a-sha256"
    receipt_path.write_text(json.dumps(payload))

    result = verify_research_artifact(receipt_path)

    assert result["valid"] is False
    assert "invalid_artifact_pointer:json" in result["errors"]
    assert "invalid_artifact_hash:immutable_markdown_sha256" in result["errors"]


@pytest.mark.parametrize("field", ["schema_version", "benchmark_catalog_version"])
def test_verify_rejects_boolean_version_markers(tmp_path: Path, field: str) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    if field == "schema_version":
        payload[field] = True
    else:
        payload["provenance"][field] = True
    path.write_text(json.dumps(payload))

    result = verify_research_artifact(path)

    assert result["valid"] is False
    expected = (
        "invalid_research_receipt_schema_version"
        if field == "schema_version"
        else "invalid_benchmark_catalog_version"
    )
    assert expected in result["errors"]


def test_verify_rejects_whitespace_padded_synthetic_source_mismatch(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["data_source"] = " SYNTHETIC "
    payload["synthetic"] = False
    path.write_text(json.dumps(payload))

    result = verify_research_artifact(path)

    assert result["valid"] is False
    assert "notebook_source_synthetic_mismatch" in result["errors"]


def test_verify_fixture_dgp_panel_kupiec_without_h16_h18_ok(tmp_path: Path) -> None:
    """Fixture DGP skips H16–H18 consistency (agent never mints panel H-table)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["localized_conformal"] = {"kupiec_p": 0.48, "dgp": "fixture"}
    payload["families"]["online_crc"] = {"kupiec_p": 0.65, "dgp": "fixture"}
    payload["families"]["portfolio_conformal"] = {"kupiec_p": 0.75, "dgp": "fixture"}
    payload["hypotheses"] = []
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []


def test_verify_nan_panel_kupiec_p_without_h16_h18_ok(tmp_path: Path) -> None:
    """NaN panel kupiec_p does not require H16–H18 (agent skip-on-nonfinite)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["localized_conformal"] = {"kupiec_p": float("nan"), "dgp": "panel"}
    payload["hypotheses"] = []
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h16_missing_despite_finite_kupiec_p" not in result["errors"]


def test_verify_rejects_finite_panel_kupiec_p_with_h16_wrong_family(tmp_path: Path) -> None:
    """H16 id present but family != calibration still fails soft consistency."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["localized_conformal"] = {"kupiec_p": 0.48, "dgp": "panel"}
    payload["hypotheses"] = [_h16_hypothesis(family="discovery")]
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h16_missing_despite_finite_kupiec_p" in result["errors"]


def _h19_hypothesis(*, family: str = "bound", p_value: float = float("nan")) -> dict:
    """Minimal valid H19 row matching REQUIRED_HYPOTHESIS_FIELDS (family=bound)."""
    return {
        "id": H19_HYPOTHESIS_ID,
        "statement": "Conformal top-k date-grouped FDR stays at or below alpha on the lab panel.",
        "test": "FDR ≤ alpha bound check; not FDR-discovery",
        "statistic": -0.05,
        "p_value": p_value,
        "reject_raw": False,
        "reject_fdr": False,
        "decision": "Top-k FDR at or below alpha.",
        "family": family,
    }


def test_catalog_conformal_rank_has_finite_fdr() -> None:
    assert conformal_rank_has_finite_fdr({"fdr": 0.12, "dgp": "panel"}) is True
    assert conformal_rank_has_finite_fdr({"fdr": 0.12}) is True  # missing dgp ok
    assert conformal_rank_has_finite_fdr({"fdr": 0.12, "dgp": "fixture"}) is False
    assert conformal_rank_has_finite_fdr({"fdr": float("nan")}) is False
    assert conformal_rank_has_finite_fdr({"fdr": float("inf")}) is False
    assert conformal_rank_has_finite_fdr({}) is False
    assert conformal_rank_has_finite_fdr(None) is False


def test_catalog_h19_conformal_rank_consistency_skips_nonfinite_and_fixture() -> None:
    assert h19_hypothesis_consistency_errors({"fdr": float("nan")}, []) == []
    assert h19_hypothesis_consistency_errors({"fdr": 0.12, "dgp": "fixture"}, []) == []
    assert h19_hypothesis_consistency_errors({}, []) == []
    assert h19_hypothesis_consistency_errors(None, []) == []


def test_verify_rejects_finite_conformal_rank_fdr_without_h19(tmp_path: Path) -> None:
    """Finite conformal_rank.fdr (non-fixture) without H19 hyp → soft fail-closed."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["conformal_rank"] = {"fdr": 0.12, "dgp": "panel", "alpha": 0.20}
    payload["hypotheses"] = []
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h19_missing_despite_finite_fdr" in result["errors"]


def test_verify_accepts_finite_conformal_rank_fdr_with_h19(tmp_path: Path) -> None:
    """Finite fdr + bound H19 → no H19 consistency error."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["conformal_rank"] = {"fdr": 0.12, "dgp": "panel", "alpha": 0.20}
    payload["hypotheses"] = [_h19_hypothesis()]
    assert hypotheses_include_h19(payload["hypotheses"]) is True
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h19_missing_despite_finite_fdr" not in result["errors"]


def test_verify_nan_conformal_rank_fdr_without_h19_ok(tmp_path: Path) -> None:
    """NaN fdr does not require H19 (agent skip-on-nonfinite)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["conformal_rank"] = {"fdr": float("nan"), "dgp": "panel"}
    payload["hypotheses"] = []
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h19_missing_despite_finite_fdr" not in result["errors"]


def test_verify_fixture_dgp_conformal_rank_without_h19_ok(tmp_path: Path) -> None:
    """Fixture DGP never shares H19 H-table — skip even with finite fdr."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["conformal_rank"] = {"fdr": 0.12, "dgp": "fixture", "alpha": 0.20}
    payload["hypotheses"] = []
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "hypothesis_h19_missing_despite_finite_fdr" not in result["errors"]


def test_verify_rejects_finite_conformal_rank_fdr_with_h19_wrong_family(tmp_path: Path) -> None:
    """H19 id present but family != bound still fails soft consistency."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["conformal_rank"] = {"fdr": 0.12, "dgp": "panel", "alpha": 0.20}
    payload["hypotheses"] = [_h19_hypothesis(family="calibration")]
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h19_missing_despite_finite_fdr" in result["errors"]


# --- Day Wave 42: coverage_guarantee_scope receipt honesty ---


def test_catalog_finite_observation_is_recursive_and_fail_closed() -> None:
    assert family_blob_has_finite_observation({"nested": [{"metric": 1.25}]}) is True
    assert family_blob_has_finite_observation({"flag": False}) is True
    assert family_blob_has_finite_observation({"label": "  observed  "}) is True
    assert family_blob_has_finite_observation({"label": "nan"}) is False
    assert family_blob_has_finite_observation({"label": "none"}) is False
    assert family_blob_has_finite_observation({"metric": float("nan")}) is False
    assert family_blob_has_finite_observation({"metric": float("inf")}) is False
    assert family_blob_has_finite_observation({}) is False
    assert family_blob_has_finite_observation(None) is False


def test_catalog_jp_cv_blob_requires_marginal_coverage_scope() -> None:
    assert jp_cv_blob_requires_marginal_coverage_scope({"coverage": 0.9}) is True
    assert jp_cv_blob_requires_marginal_coverage_scope({"coverage_floor": 0.8}) is True
    assert jp_cv_blob_requires_marginal_coverage_scope({"coverage": float("nan")}) is True
    assert jp_cv_blob_requires_marginal_coverage_scope({"coverage_floor": float("nan")}) is True
    assert jp_cv_blob_requires_marginal_coverage_scope({"executed": True}) is False
    assert jp_cv_blob_requires_marginal_coverage_scope({}) is False
    assert jp_cv_blob_requires_marginal_coverage_scope(None) is False


def test_catalog_coverage_guarantee_scope_is_marginal() -> None:
    assert coverage_guarantee_scope_is_marginal(
        {"coverage_guarantee_scope": "marginal_exchangeable"}
    )
    assert (
        coverage_guarantee_scope_is_marginal({"coverage_guarantee_scope": "training_conditional"})
        is False
    )
    assert coverage_guarantee_scope_is_marginal({}) is False
    assert coverage_guarantee_scope_is_marginal(None) is False


def test_catalog_coverage_guarantee_scope_consistency_skips_empty() -> None:
    assert coverage_guarantee_scope_consistency_errors({}) == []
    assert (
        coverage_guarantee_scope_consistency_errors(
            {"jackknife_plus": {}, "cv_plus": {"executed": True}}
        )
        == []
    )
    assert coverage_guarantee_scope_consistency_errors(None) == []


def test_verify_accepts_jp_coverage_with_marginal_scope(tmp_path: Path) -> None:
    """Complete jp coverage + scope + bound H10 → ok (Wave42)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["jackknife_plus"] = {
        "coverage": 0.85,
        "coverage_guarantee_scope": "marginal_exchangeable",
    }
    payload["hypotheses"] = [_h10_hypothesis()]
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert not any(e.startswith("coverage_guarantee_scope_") for e in result["errors"])


def test_verify_rejects_jp_coverage_without_scope(tmp_path: Path) -> None:
    """Coverage key without coverage_guarantee_scope → missing token."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["jackknife_plus"] = {"coverage": 0.85}
    payload["hypotheses"] = [_h10_hypothesis()]
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "coverage_guarantee_scope_missing:jackknife_plus" in result["errors"]


def test_verify_rejects_cv_plus_wrong_scope(tmp_path: Path) -> None:
    """Wrong coverage_guarantee_scope value → invalid token."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["cv_plus"] = {
        "coverage": 0.90,
        "coverage_floor": 0.80,
        "coverage_guarantee_scope": "training_conditional",
    }
    payload["hypotheses"] = [_h15_hypothesis()]
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "coverage_guarantee_scope_invalid:cv_plus" in result["errors"]


def test_verify_empty_jp_cv_blobs_skip_scope(tmp_path: Path) -> None:
    """Empty {} / no coverage keys → Wave42 skip (default receipt already ok)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["jackknife_plus"] = {"executed": True}
    payload["families"]["cv_plus"] = {"executed": True}
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []
    assert not any(e.startswith("coverage_guarantee_scope_") for e in result["errors"])


def test_verify_nan_jp_coverage_still_requires_scope(tmp_path: Path) -> None:
    """NaN coverage still requires Wave41 scope key (presence-only gate)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["jackknife_plus"] = {"coverage": float("nan")}
    payload["hypotheses"] = []
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "coverage_guarantee_scope_missing:jackknife_plus" in result["errors"]
    # H10 still skips on nonfinite coverage
    assert "hypothesis_h10_missing_despite_finite_coverage" not in result["errors"]


def test_verify_scope_ok_forbidden_hygiene_unchanged(tmp_path: Path) -> None:
    """Wave42 scope ok must not weaken forbidden-metrics scorecard hygiene."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["jackknife_plus"] = {
        "coverage": 0.85,
        "coverage_guarantee_scope": "marginal_exchangeable",
        "sharpe": 1.2,
    }
    payload["hypotheses"] = [_h10_hypothesis()]
    payload["scorecard"]["jackknife_plus"]["forbidden_metrics_absent"] = True
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "scorecard_forbidden_flag_forged:jackknife_plus" in result["errors"]
    assert not any(
        e.startswith("coverage_guarantee_scope_missing:")
        or e.startswith("coverage_guarantee_scope_invalid:")
        for e in result["errors"]
    )


# --- Day Wave 44: scorecard executed/nonempty/finite_observation forge honesty (parked/pre-landed) ---


def test_catalog_family_blob_executed_nonempty() -> None:
    assert family_blob_executed({"coverage": 0.9}) is True
    assert family_blob_nonempty({"coverage": 0.9}) is True
    assert family_blob_executed({}) is False
    assert family_blob_nonempty({}) is False
    assert family_blob_executed(None) is False
    assert family_blob_nonempty(None) is False


def test_catalog_family_blob_has_finite_observation() -> None:
    assert family_blob_has_finite_observation({"coverage": 0.9}) is True
    assert family_blob_has_finite_observation({"ok": True}) is True
    assert family_blob_has_finite_observation({"n": 3}) is True
    assert family_blob_has_finite_observation({"label": "aci"}) is True
    assert family_blob_has_finite_observation({}) is False
    assert family_blob_has_finite_observation({"coverage": float("nan")}) is False
    assert family_blob_has_finite_observation({"a": float("nan"), "b": float("nan")}) is False
    assert family_blob_has_finite_observation({"nested": {"x": float("nan")}}) is False
    assert family_blob_has_finite_observation({"nested": {"x": 1.0}}) is True
    assert family_blob_has_finite_observation({"label": "nan"}) is False
    assert family_blob_has_finite_observation({"label": "none"}) is False
    assert family_blob_has_finite_observation(None) is False
    assert family_blob_has_finite_observation([float("nan"), 2.0]) is True
    assert family_blob_has_finite_observation([float("nan")]) is False


def test_verify_rejects_forged_executed_nonempty_on_empty_blob(tmp_path: Path) -> None:
    """Hand-edited scorecard True on empty family blob → executed/nonempty forge."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["ranking"] = {}
    payload["scorecard"]["ranking"]["executed"] = True
    payload["scorecard"]["ranking"]["nonempty"] = True
    # Keep other required flags True so scorecard_invalid does not mask forges.
    payload["scorecard"]["ranking"]["finite_observation"] = False
    payload["scorecard"]["ranking"]["forbidden_metrics_absent"] = True
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "scorecard_executed_flag_forged:ranking" in result["errors"]
    assert "scorecard_nonempty_flag_forged:ranking" in result["errors"]
    # finite_observation False → not a forge
    assert "scorecard_finite_observation_flag_forged:ranking" not in result["errors"]
    # valid-receipt gate still sees invalid scorecard (finite_observation False)
    assert "scorecard_invalid:ranking" in result["errors"]


def test_verify_rejects_forged_finite_observation_on_all_nan_blob(tmp_path: Path) -> None:
    """All-NaN blob with finite_observation=True → forge (executed/nonempty honest)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["ranking"] = {"mean_ic": float("nan"), "n": float("nan")}
    payload["scorecard"]["ranking"]["executed"] = True
    payload["scorecard"]["ranking"]["nonempty"] = True
    payload["scorecard"]["ranking"]["finite_observation"] = True
    payload["scorecard"]["ranking"]["forbidden_metrics_absent"] = True
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "scorecard_finite_observation_flag_forged:ranking" in result["errors"]
    assert "scorecard_executed_flag_forged:ranking" not in result["errors"]
    assert "scorecard_nonempty_flag_forged:ranking" not in result["errors"]


def test_verify_rejects_forged_all_three_flags_on_empty_blob(tmp_path: Path) -> None:
    """Empty {} with all three flags True → three forge tokens (parallel battery forge)."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["volatility"] = {}
    sc = payload["scorecard"]["volatility"]
    sc["executed"] = True
    sc["nonempty"] = True
    sc["finite_observation"] = True
    sc["forbidden_metrics_absent"] = True
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "scorecard_executed_flag_forged:volatility" in result["errors"]
    assert "scorecard_nonempty_flag_forged:volatility" in result["errors"]
    assert "scorecard_finite_observation_flag_forged:volatility" in result["errors"]


def test_verify_false_observation_flags_no_forge_even_if_empty(tmp_path: Path) -> None:
    """Explicit False executed/nonempty/finite_observation is not a forge."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["alpha"] = {}
    sc = payload["scorecard"]["alpha"]
    sc["executed"] = False
    sc["nonempty"] = False
    sc["finite_observation"] = False
    sc["forbidden_metrics_absent"] = True
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "scorecard_executed_flag_forged:alpha" not in result["errors"]
    assert "scorecard_nonempty_flag_forged:alpha" not in result["errors"]
    assert "scorecard_finite_observation_flag_forged:alpha" not in result["errors"]
    assert "scorecard_invalid:alpha" in result["errors"]


def test_verify_absent_observation_flags_no_forge_even_if_empty(tmp_path: Path) -> None:
    """Absent observation flags → no forge; scorecard_invalid still fires."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["regime"] = {}
    sc = payload["scorecard"]["regime"]
    for key in ("executed", "nonempty", "finite_observation"):
        sc.pop(key, None)
    sc["forbidden_metrics_absent"] = True
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "scorecard_executed_flag_forged:regime" not in result["errors"]
    assert "scorecard_nonempty_flag_forged:regime" not in result["errors"]
    assert "scorecard_finite_observation_flag_forged:regime" not in result["errors"]
    assert "scorecard_invalid:regime" in result["errors"]


def test_verify_honest_finite_observation_true_no_forge(tmp_path: Path) -> None:
    """Honest True flags with finite observation → no observation forge errors."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["ranking"] = {"mean_ic": 0.12, "n": 10}
    sc = payload["scorecard"]["ranking"]
    sc["executed"] = True
    sc["nonempty"] = True
    sc["finite_observation"] = True
    sc["forbidden_metrics_absent"] = True
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    # May still fail on other soft gates if ranking-only families incomplete — assert forges absent
    assert "scorecard_executed_flag_forged:ranking" not in result["errors"]
    assert "scorecard_nonempty_flag_forged:ranking" not in result["errors"]
    assert "scorecard_finite_observation_flag_forged:ranking" not in result["errors"]


def _h20_hypothesis(*, family: str = "bound") -> dict:
    return {
        "id": H20_HYPOTHESIS_ID,
        "statement": "Daily candlestick rows satisfy OHLC identities (high/low envelope).",
        "test": "ohlc_identity_rate ≥ 1 bound check; not FDR",
        "statistic": 0.0,
        "p_value": float("nan"),
        "reject_raw": False,
        "reject_fdr": False,
        "decision": "OHLC identities hold.",
        "family": family,
        "meets_floor": True,
    }


def _h21_hypothesis(*, family: str = "bound") -> dict:
    return {
        "id": H21_HYPOTHESIS_ID,
        "statement": "L2 snapshots are uncrossed (best bid strictly below best ask).",
        "test": "book_uncrossed_rate ≥ 1 bound check; not FDR",
        "statistic": 0.0,
        "p_value": float("nan"),
        "reject_raw": False,
        "reject_fdr": False,
        "decision": "Books are uncrossed.",
        "family": family,
        "meets_floor": True,
    }


def _h22_hypothesis(*, family: str = "discovery", p_value: float = 0.04) -> dict:
    return {
        "id": H22_HYPOTHESIS_ID,
        "statement": "Top-of-book depth imbalance has nonzero date-level IC vs next-bar return.",
        "test": "HAC t-stat of date-level imbalance IC",
        "statistic": 2.0,
        "p_value": p_value,
        "reject_raw": True,
        "reject_fdr": False,
        "decision": "Imbalance IC is distinguishable from 0.",
        "family": family,
    }


def test_catalog_northset_finite_helpers() -> None:
    assert northset_has_finite_ohlc_identity_rate({"ohlc_identity_rate": 1.0}) is True
    assert northset_has_finite_ohlc_identity_rate({"ohlc_identity_rate": float("nan")}) is False
    assert northset_has_finite_book_uncrossed_rate({"book_uncrossed_rate": 1.0}) is True
    assert northset_has_finite_imbalance_p_ic({"imbalance_top_p_ic": 0.02}) is True
    assert northset_has_finite_imbalance_p_ic({}) is False


def test_catalog_h20_h22_consistency_skips_nonfinite() -> None:
    assert h20_hypothesis_consistency_errors({"ohlc_identity_rate": float("nan")}, []) == []
    assert h21_hypothesis_consistency_errors({}, []) == []
    assert h22_hypothesis_consistency_errors(None, []) == []


def test_catalog_h20_h22_consistency_requires_matching_family() -> None:
    blob = {
        "ohlc_identity_rate": 1.0,
        "book_uncrossed_rate": 1.0,
        "imbalance_top_p_ic": 0.03,
    }
    assert h20_hypothesis_consistency_errors(blob, []) == [
        "hypothesis_h20_missing_despite_finite_ohlc_identity_rate"
    ]
    assert h21_hypothesis_consistency_errors(blob, []) == [
        "hypothesis_h21_missing_despite_finite_book_uncrossed_rate"
    ]
    assert h22_hypothesis_consistency_errors(blob, []) == [
        "hypothesis_h22_missing_despite_finite_imbalance_p_ic"
    ]
    hyps = [_h20_hypothesis(), _h21_hypothesis(), _h22_hypothesis()]
    assert h20_hypothesis_consistency_errors(blob, hyps) == []
    assert h21_hypothesis_consistency_errors(blob, hyps) == []
    assert h22_hypothesis_consistency_errors(blob, hyps) == []
    assert hypotheses_include_h20(hyps) is True
    assert hypotheses_include_h21(hyps) is True
    assert hypotheses_include_h22(hyps) is True
    assert h22_hypothesis_consistency_errors(blob, [_h22_hypothesis(family="bound")]) == [
        "hypothesis_h22_missing_despite_finite_imbalance_p_ic"
    ]


def test_verify_research_artifact_requires_h20_when_ohlc_rate_finite(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["families"]["northset"] = {"ohlc_identity_rate": 1.0}
    payload["hypotheses"] = []
    path.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "hypothesis_h20_missing_despite_finite_ohlc_identity_rate" in result["errors"]


def test_catalog_h23_h42_consistency_requires_rows() -> None:
    blob = {
        "session_reconstructs_daily_rate": 1.0,
        "session_volume_conservation_rate": 1.0,
        "microprice_p_ic": 0.04,
        "wick_skew_p_ic": 0.03,
        "ofi_p_ic": 0.02,
        "dm_gk_vs_park_p": 0.01,
        "session_chain_rate": 1.0,
        "clv_p_ic": 0.05,
        "dm_split_vs_park_p": 0.02,
        "vpin_p_ic": 0.04,
        "sweep_reject_signed_p_ic": 0.03,
        "sweep_follow_signed_p_ic": 0.06,
        "sweep_reject_event_p": 0.02,
        "sweep_follow_event_p": 0.03,
        "sweep_reject_placebo_p": 0.04,
        "sweep_follow_placebo_p": 0.05,
        "sweep_reject_cost_adjusted_mean_bps": -1.0,
        "sweep_follow_cost_adjusted_mean_bps": 2.0,
        "sweep_reject_fold_positive_fraction": 0.5,
        "sweep_follow_fold_positive_fraction": 0.75,
        "sweep_reject_control_diff_p": 0.03,
        "sweep_follow_control_diff_p": 0.01,
        "sweep_reject_liq_control_diff_p": 0.04,
        "sweep_follow_liq_control_diff_p": 0.02,
        "sweep_follow_oot_holdout_mean_bps": 3.0,
        "sweep_follow_name_cluster_p": 0.01,
        "sweep_follow_two_way_cluster_p": 0.02,
        "sweep_follow_overnight_gap_p": 0.04,
    }
    errors = northset_h23_h28_consistency_errors(blob, [])
    assert "hypothesis_h23_missing_despite_finite_session_reconstructs_daily_rate" in errors
    assert "hypothesis_h24_missing_despite_finite_session_volume_conservation_rate" in errors
    assert "hypothesis_h25_missing_despite_finite_microprice_p_ic" in errors
    assert "hypothesis_h26_missing_despite_finite_wick_skew_p_ic" in errors
    assert "hypothesis_h27_missing_despite_finite_ofi_p_ic" in errors
    assert "hypothesis_h28_missing_despite_finite_dm_gk_vs_park_p" in errors
    assert "hypothesis_h29_missing_despite_finite_session_chain_rate" in errors
    assert "hypothesis_h30_missing_despite_finite_clv_p_ic" in errors
    assert "hypothesis_h31_missing_despite_finite_dm_split_vs_park_p" in errors
    assert "hypothesis_h32_missing_despite_finite_vpin_p_ic" in errors
    assert "hypothesis_h33_missing_despite_finite_sweep_reject_signed_p_ic" in errors
    assert "hypothesis_h34_missing_despite_finite_sweep_follow_signed_p_ic" in errors
    assert "hypothesis_h35_missing_despite_finite_sweep_reject_event_p" in errors
    assert "hypothesis_h36_missing_despite_finite_sweep_follow_event_p" in errors
    assert "hypothesis_h37_missing_despite_finite_sweep_reject_placebo_p" in errors
    assert "hypothesis_h38_missing_despite_finite_sweep_follow_placebo_p" in errors
    assert "hypothesis_h39_missing_despite_finite_sweep_reject_cost_adjusted_mean_bps" in errors
    assert "hypothesis_h40_missing_despite_finite_sweep_follow_cost_adjusted_mean_bps" in errors
    assert "hypothesis_h41_missing_despite_finite_sweep_reject_fold_positive_fraction" in errors
    assert "hypothesis_h42_missing_despite_finite_sweep_follow_fold_positive_fraction" in errors
    assert "hypothesis_h44_missing_despite_finite_sweep_reject_control_diff_p" in errors
    assert "hypothesis_h45_missing_despite_finite_sweep_follow_control_diff_p" in errors
    assert "hypothesis_h46_missing_despite_finite_sweep_reject_liq_control_diff_p" in errors
    assert "hypothesis_h47_missing_despite_finite_sweep_follow_liq_control_diff_p" in errors
    assert "hypothesis_h48_missing_despite_finite_sweep_follow_oot_holdout_mean_bps" in errors
    assert "hypothesis_h49_missing_despite_finite_sweep_follow_name_cluster_p" in errors
    assert "hypothesis_h50_missing_despite_finite_sweep_follow_two_way_cluster_p" in errors
    assert "hypothesis_h51_missing_despite_finite_sweep_follow_overnight_gap_p" in errors
    assert (
        northset_h23_h28_consistency_errors({"session_reconstructs_daily_rate": float("nan")}, [])
        == []
    )


def test_verify_accepts_schema_v1_without_overfitting_block(tmp_path: Path) -> None:
    """Legacy notebooks predate the block and stay valid. Their numbers are untouched."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["schema_version"] = 1
    payload.pop("backtest_overfitting")
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    text = json.dumps(payload)
    path.write_text(text)
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(text)
    result = verify_research_artifact(path)
    assert result["valid"] is True
    assert result["errors"] == []


def test_verify_rejects_schema_v2_without_overfitting_block(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload.pop("backtest_overfitting")
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "backtest_overfitting_missing" in result["errors"]


def test_verify_rejects_unknown_research_schema(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["schema_version"] = 3
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "invalid_research_receipt_schema_version" in result["errors"]


def test_verify_rejects_overfitting_block_with_forbidden_key(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["backtest_overfitting"] = {**unavailable_overfitting_block(), "sharpe": 1.0}
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "backtest_overfitting_forbidden_metrics" in result["errors"]


def test_verify_rejects_dsr_above_psr(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["backtest_overfitting"] = {
        **unavailable_overfitting_block(),
        "n_trials": 4,
        "n_trials_effective": 2,
        "metrics_status": "computed",
        "pbo": 0.4,
        "psr": 0.5,
        "dsr": 0.9,
        "dsr_counted_trials": 0.4,
        "min_trl": 12.0,
    }
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "backtest_overfitting_dsr_above_psr" in result["errors"]


def test_migrate_schema_v1_keeps_existing_fields_and_does_not_invent_metrics(
    tmp_path: Path,
) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["schema_version"] = 1
    payload.pop("backtest_overfitting")
    payload["rankers"] = [
        {"name": "ridge_public", "mean_ic": 0.01},
        {"name": "ridge_oracle", "mean_ic": 0.02},
        {"name": "_pairwise_dm_summary"},
    ]
    families = payload["families"]
    migrated = migrate_research_receipt(payload)
    assert payload["schema_version"] == 1
    assert "backtest_overfitting" not in payload
    assert migrated["schema_version"] == RESEARCH_RECEIPT_SCHEMA_VERSION
    assert migrated["families"] == families
    assert migrated["rankers"] == payload["rankers"]
    block = migrated["backtest_overfitting"]
    assert block["metrics_status"] == "legacy_uncomputed"
    assert block["migrated_from_schema"] == 1
    assert block["n_trials"] == 2
    assert block["pbo"] is None
    assert block["dsr"] is None
    assert migrate_research_receipt(migrated)["backtest_overfitting"] == block


def test_timestamp_utcoffset_redundant_arm_is_documented_equivalent() -> None:
    """Mutation-campaign note (full 210-mutant run, 209 killed, 99.52%):
    the sole survivor is the And->Or mutant of ``_timestamp_valid``'s
    ``parsed.tzinfo is not None and parsed.utcoffset() is not None``.
    It is a PROVABLY EQUIVALENT mutant: ``datetime.fromisoformat`` only
    ever produces fixed-offset timezones, whose ``utcoffset()`` is never
    None when ``tzinfo`` is not None — no string input can distinguish
    the two arms.  This test pins that redundancy claim so a future
    survivor is recognized as the known equivalent, not a new gap.

    NOTE (full-campaign re-run): the "99.52% / 209 killed" figure above was
    measured against the harness DEFAULT 50-mutant sample, not all 210. The
    genuinely-full 210-mutant run found 15 survivors; M0032 is the only
    provably-equivalent one and the other 14 are closed by the tests appended
    below. See INFLIGHT 2026-10-01."""
    from datetime import datetime

    samples = [
        "2026-09-16T00:00:00+00:00",
        "2026-09-16T00:00:00Z",
        "2026-09-16T12:30:00-05:00",
        "2026-09-16T12:30:00+05:30",
    ]
    for text in samples:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() is not None  # never None given tzinfo


# ---------------------------------------------------------------------------
# Full 210-mutant campaign kill tests (2026-10-01)
#
# The prior close-out reported 99.52% (209/210) but the harness defaults to
# --max-mutants 50, so that figure came from a 50-mutant seed-7 sample, not
# the full sweep. Re-running all 210 surfaced 15 survivors; M0032 is provably
# equivalent (pinned above) and the other 14 are closed here. Each test
# docstring names the exact mutant + line it kills.
# ---------------------------------------------------------------------------


def test_verify_rejects_non_string_generated_at(tmp_path: Path) -> None:
    """Kills M0031 (Or->And, line 195) and M0059 (False->True, line 196).

    ``_timestamp_valid``'s guard is ``not isinstance(value, str) or not
    value.strip()``. With a non-string the ``or`` short-circuits to the
    isinstance arm and returns False cleanly; the And mutant instead
    evaluates ``value.strip()`` on the non-string and raises AttributeError.
    Only strings had ever been passed, so both mutants survived.
    """
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["generated_at"] = 1757980800
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "invalid_notebook_generated_at" in result["errors"]


def test_verify_rejects_unparseable_generated_at(tmp_path: Path) -> None:
    """Kills M0106 (False->True, line 200) — the ``except ValueError`` arm.

    A syntactically invalid timestamp string raises ValueError inside
    ``fromisoformat``. The original returns False (fail-closed); the mutant
    returns True and then dereferences the unbound ``parsed`` name. Every
    prior timestamp test used a *parseable* naive string, which reaches line
    201 instead and never exercised this handler.
    """
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["generated_at"] = "not-a-timestamp"
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "invalid_notebook_generated_at" in result["errors"]


@pytest.mark.parametrize("key", ["version", "disclaimer", "ranking_target"])
def test_verify_rejects_non_string_notebook_identity_field(tmp_path: Path, key: str) -> None:
    """Kills M0064 (Or->And, line 249) on the notebook identity loop.

    Same ``or``/``and`` shape as the timestamp guard, applied to
    version / data_source / disclaimer / ranking_target. ``data_source``
    is excluded from the parametrization: a non-string there also trips the
    isinstance check on line 256, so it cannot isolate this mutant.
    """
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload[key] = 1
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert f"invalid_notebook_{key}" in result["errors"]


def test_verify_rejects_non_string_hypothesis_id(tmp_path: Path) -> None:
    """Kills M0113 (Or->And, line 313) on the hypothesis id guard."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["hypotheses"] = [
        {
            "id": 7,
            "statement": "x",
            "test": "test",
            "statistic": 0.0,
            "p_value": 1.0,
            "reject_raw": False,
            "reject_fdr": False,
            "decision": "x",
            "family": "calibration",
        }
    ]
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "invalid_hypothesis_id:0" in result["errors"]


@pytest.mark.parametrize("field", ["statement", "test", "decision"])
def test_verify_rejects_non_string_hypothesis_text_field(tmp_path: Path, field: str) -> None:
    """Kills M0159 (Or->And, line 323) on the statement/test/decision guard."""
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    hypothesis = {
        "id": "H1",
        "statement": "x",
        "test": "test",
        "statistic": 0.0,
        "p_value": 1.0,
        "reject_raw": False,
        "reject_fdr": False,
        "decision": "x",
        "family": "calibration",
    }
    hypothesis[field] = 3.5
    payload["hypotheses"] = [hypothesis]
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert f"invalid_hypothesis_{field}:0" in result["errors"]


def test_verify_rejects_non_string_runtime_fingerprint_field(tmp_path: Path) -> None:
    """Kills M0122 (Or->And, line 386) on the runtime fingerprint guard.

    The guard is ``not isinstance(runtime.get(field), str) or not
    runtime[field]``. A non-string short-circuits on the isinstance arm; the
    And mutant would index and truth-test the non-string instead. Prior
    runtime tests only ever removed fields or set a package *version* to a
    non-string, never a fingerprint field.
    """
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["provenance"]["runtime"]["python"] = 3.12
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "runtime_field_invalid:python" in result["errors"]


def test_verify_rejects_empty_packages_dict(tmp_path: Path) -> None:
    """Kills M0079 (Or->And, line 389) on the packages guard.

    ``not isinstance(packages, dict) or not packages`` distinguishes an empty
    dict (``not packages`` is True) from a populated one. The And mutant lets
    ``{}`` fall through to the elif branch and reports missing *required*
    packages instead of ``runtime_packages_missing``, so the error token
    differs and the mutant is observable.
    """
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["provenance"]["runtime"]["packages"] = {}
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "runtime_packages_missing" in result["errors"]


def test_verify_rejects_non_numeric_hypothesis_p_value(tmp_path: Path) -> None:
    """Kills M0053 (False->True, line 179) in ``_p_value_valid``.

    The non-numeric arm returns False. Flipping it to True lets a string
    p_value reach ``math.isnan`` and raise TypeError. Prior p_value tests
    covered the numeric range boundaries (0.0, 1.0, 1.5, -1e-12) and the
    None/NaN unavailable paths, never a non-numeric type.
    """
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["hypotheses"] = [
        {
            "id": "H1",
            "statement": "x",
            "test": "test",
            "statistic": 0.0,
            "p_value": "0.05",
            "reject_raw": False,
            "reject_fdr": False,
            "decision": "x",
            "family": "calibration",
        }
    ]
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert "invalid_hypothesis_p_value:0" in result["errors"]


def test_verify_rejects_unreadable_receipt_json(tmp_path: Path) -> None:
    """Kills M0186 (False->True, line 902) — the unreadable-receipt branch.

    A file that is not valid JSON takes the ``except (OSError,
    json.JSONDecodeError)`` path, which returns a ``valid: False`` stub. The
    mutant reports the unreadable receipt as valid. No prior test fed
    malformed JSON (they all round-tripped a dict), so the flag was never
    observed.
    """
    path = tmp_path / "corrupt.json"
    path.write_text("{not json at all")
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert any(error.startswith("unreadable:") for error in result["errors"])


def test_verify_rejects_non_string_immutable_json_pointer(tmp_path: Path) -> None:
    """Kills M0098 (negation, line 848), M0100 (And->Or, line 856),
    M0142 (And->Or, line 849) on the immutable-artifact path ternary.

    ``immutable_json`` set to a non-string drives the ternary to its
    ``else ... else None`` arm, so ``immutable_path`` is None and the
    hash-verification block is skipped. The mutants respectively make the
    relative-path test invert, make the ``is not None and is_file()`` guard
    an ``or`` (evaluating ``None.is_file()``), and turn the relative-path
    conjunction into a disjunction — all observable as a crash or a changed
    error set. The existing non-string test only covered
    ``immutable_markdown``'s *pointer*, which reaches a different arm.
    """
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["artifacts"]["immutable_json"] = 42
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert any(error.startswith("artifact_") for error in result["errors"])


def test_verify_treats_unresolvable_artifact_path_as_outside_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kills M0200 (False->True, line 841) — the ``except OSError`` arm.

    ``Path.resolve()`` can raise OSError (symlink loops, vanished parents,
    permission faults). The handler fails closed by setting
    ``inside_receipt = False``; the mutant would set it True and accept an
    artifact whose real location cannot be established. The branch is
    unreachable with ordinary fixture paths, so ``Path.resolve`` is patched
    to raise for the artifact check specifically — patching the whole method
    would also break line 805's ``receipt_root`` computation.
    """
    path = _receipt(tmp_path)
    original_resolve = Path.resolve
    receipt_root = path.resolve().parent
    artifact_name = f"{'a' * 64}.json"

    def flaky_resolve(self: Path, *args: object, **kwargs: object) -> Path:
        # Let receipt_root (the parent) resolve normally; raise only when the
        # immutable artifact itself is being checked.
        if self.name == artifact_name:
            raise OSError("simulated unresolvable path")
        return original_resolve(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "resolve", flaky_resolve)
    result = verify_research_artifact(path)
    monkeypatch.undo()

    assert receipt_root.is_dir()
    assert result["valid"] is False
    assert "artifact_outside_receipt_root:immutable_json" in result["errors"]
