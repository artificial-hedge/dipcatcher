"""Tests for tamper_drill — the mechanized fail-closed proof."""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_fund.research.tamper_drill import (
    DRILL_SCHEMA,
    drill_contract_errors,
    tamper_drill,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
QUALITY = REPO_ROOT / "quality"
requires_tree = pytest.mark.skipif(
    not (QUALITY / "crown_jewels.json").is_file(),
    reason="committed integrity state not present",
)


@requires_tree
@pytest.mark.slow
def test_tamper_drill_fail_closed_on_committed_tree() -> None:
    """Every probe over the real integrity state must be caught."""
    result = tamper_drill(REPO_ROOT)
    assert result["schema"] == DRILL_SCHEMA
    assert result["baseline_errors"] == []
    assert result["n_probes"] >= 10, "drill needs real surface to be meaningful"
    assert result["n_caught"] == result["n_probes"], [
        p["probe"] for p in result["probes"] if not p.get("caught")
    ]
    assert result["verdict"] == "fail_closed"
    assert drill_contract_errors(result) == []


def test_drill_contract_clean_and_strict() -> None:
    good = {
        "schema": "tamper_drill.v1",
        "research_only": True,
        "live_pnl_claim": False,
        "probes": [
            {"probe": "flip_x", "caught": True, "errors": ["gate:x"]},
            {"probe": "delete_y", "caught": True, "errors": []},
        ],
        "n_probes": 2,
        "n_caught": 2,
        "verdict": "fail_closed",
    }
    assert drill_contract_errors(good) == []

    assert drill_contract_errors({"schema": "other"}) == ["schema_mismatch"]
    bad = {**good, "n_caught": 1}
    assert "n_caught_mismatch" in drill_contract_errors(bad)
    escaped = {**good, "verdict": "fail_closed", "n_caught": 1, "n_probes": 2}
    assert "verdict_fail_closed_but_escaped" in drill_contract_errors(escaped)
