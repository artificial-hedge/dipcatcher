"""Retired optional benchmark families — append-only acceptance history.

``OPTIONAL_BENCHMARK_FAMILIES`` is the historical ACCEPTED set and must never
shrink: an archived receipt naming a since-retired family has to keep
verifying (``docs/BENCHMARK_FAMILY_LIFECYCLE.md``, RETIRED state). Retirement
is recorded separately — ``RETIRED_BENCHMARK_FAMILIES`` maps each retired name
to its reason and the last receipt schema version that could have carried it,
derived from the canon qualification audit
(``quality/canon_qualification_summary.json``) — and
``LIVE_OPTIONAL_BENCHMARK_FAMILIES`` is the live scorecard
(``OPTIONAL - RETIRED``) consumed by ``research/agent.py`` for runtime
emission.

These tests pin exactly that contract: the set algebra, the evidence fields,
audit<->registry agreement (no drift), old receipts still verifying, and
fail-closed behavior for unknown families.
"""

from __future__ import annotations

import json
from pathlib import Path

from quant_fund.research.catalog.registry import (
    BENCHMARK_CATALOG_VERSION,
    LIVE_OPTIONAL_BENCHMARK_FAMILIES,
    OPTIONAL_BENCHMARK_FAMILIES,
    REQUIRED_BENCHMARK_FAMILIES,
    RESEARCH_RECEIPT_SCHEMA_VERSION,
    RESEARCH_RECEIPT_SCHEMA_VERSIONS_ACCEPTED,
    RETIRED_BENCHMARK_FAMILIES,
)
from quant_fund.research.receipt_schema import unavailable_overfitting_block
from quant_fund.research.verify import _receipt_digest, verify_research_artifact
from quant_fund.utils.hashing import hash_file

REPO_ROOT = Path(__file__).resolve().parents[3]
SUMMARY_PATH = REPO_ROOT / "quality" / "canon_qualification_summary.json"

#: Wave-1649 template-backed family, retired in the 2026-10-07 audit.
RETIRED_SAMPLE = "aswang_qa_studies"


def _receipt_payload(schema_version: int, families: dict[str, dict[str, object]]) -> dict:
    """Minimal valid research receipt carrying *families* in families+scorecard."""
    run_id = "a" * 64
    return {
        "schema_version": schema_version,
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
            for name in families
        },
        "families": dict(families),
        "artifacts": {
            "immutable_json": "immutable_placeholder",
            "immutable_markdown": "immutable_placeholder",
            "immutable_markdown_sha256": "f" * 64,
        },
    }


def _write_receipt(tmp_path: Path, schema_version: int, extra_family: str) -> Path:
    """Write a receipt carrying the REQUIRED set plus *extra_family*."""
    immutable = tmp_path / "immutable"
    immutable.mkdir(exist_ok=True)
    run_id = "a" * 64
    (immutable / f"{run_id}.md").write_text("# research")
    families: dict[str, dict[str, object]] = {
        name: {"executed": True} for name in sorted(REQUIRED_BENCHMARK_FAMILIES)
    }
    families[extra_family] = {"executed": True}
    payload = _receipt_payload(schema_version, families)
    if schema_version >= 2:
        payload["backtest_overfitting"] = unavailable_overfitting_block()
    payload["artifacts"] = {
        "immutable_json": str(immutable / f"{run_id}.json"),
        "immutable_markdown": str(immutable / f"{run_id}.md"),
        "immutable_markdown_sha256": hash_file(immutable / f"{run_id}.md"),
    }
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path = tmp_path / f"receipt_v{schema_version}.json"
    path.write_text(json.dumps(payload))
    (immutable / f"{run_id}.json").write_text(json.dumps(payload))
    return path


def test_live_is_optional_minus_retired() -> None:
    assert (
        frozenset(OPTIONAL_BENCHMARK_FAMILIES) - frozenset(RETIRED_BENCHMARK_FAMILIES)
        == LIVE_OPTIONAL_BENCHMARK_FAMILIES
    )


def test_live_and_retired_are_disjoint() -> None:
    assert not (set(LIVE_OPTIONAL_BENCHMARK_FAMILIES) & set(RETIRED_BENCHMARK_FAMILIES))


def test_retired_stays_inside_the_accepted_set() -> None:
    # OPTIONAL is append-only history: retirement never erases acceptance.
    assert set(RETIRED_BENCHMARK_FAMILIES) <= set(OPTIONAL_BENCHMARK_FAMILIES)
    # REQUIRED is frozen and permanent — nothing REQUIRED is ever retired.
    assert not (set(RETIRED_BENCHMARK_FAMILIES) & set(REQUIRED_BENCHMARK_FAMILIES))


def test_every_retired_entry_has_specific_reason_and_schema_version() -> None:
    for name, entry in RETIRED_BENCHMARK_FAMILIES.items():
        reason = entry["reason"]
        assert isinstance(reason, str)
        assert len(reason) >= 30, f"{name}: retirement reason too vague"
        assert "template" in reason.lower(), f"{name}: reason must state the template evidence"
        schema = entry["last_receipt_schema_version"]
        assert schema in RESEARCH_RECEIPT_SCHEMA_VERSIONS_ACCEPTED, name


def test_retired_set_matches_qualification_audit() -> None:
    # The registry derives from the audit; drift between them must fail here.
    assert SUMMARY_PATH.is_file(), f"missing audit summary at {SUMMARY_PATH}"
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    audited_retired = set(summary["retired_families"])
    assert set(RETIRED_BENCHMARK_FAMILIES) == audited_retired
    assert summary["ruleset_version"] == 1


def test_archived_receipt_with_retired_family_still_verifies(tmp_path: Path) -> None:
    for schema_version in sorted(RESEARCH_RECEIPT_SCHEMA_VERSIONS_ACCEPTED):
        path = _write_receipt(tmp_path, schema_version, RETIRED_SAMPLE)
        result = verify_research_artifact(path)
        assert result["valid"] is True, f"schema {schema_version}: {result['errors']}"
        assert result["errors"] == []
    assert RETIRED_SAMPLE in RETIRED_BENCHMARK_FAMILIES


def test_unknown_family_still_fails_closed(tmp_path: Path) -> None:
    path = _write_receipt(tmp_path, RESEARCH_RECEIPT_SCHEMA_VERSION, "no_such_family_zzz")
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert any("unknown" in error for error in result["errors"])
