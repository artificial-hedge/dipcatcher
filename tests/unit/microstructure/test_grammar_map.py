"""Tests for grammar_map."""

import json
from pathlib import Path

from quant_fund.microstructure.grammar_map import grammar_map


def test_grammar_map_shape(tmp_path: Path) -> None:
    # Write two sealed member receipts so presence/lookup paths run.
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

    for name, claims in (
        ("zone_card.json", {"life_on_tape_scale": False}),
        ("zone_ttl.json", {"ttl_reaches_life_scale": True}),
    ):
        body = {"claims": claims, "kind": "microstructure_bench"}
        body["receipt_sha256"] = hash_bytes(canonical_json_bytes(body))
        (tmp_path / name).write_text(json.dumps(body))

    out = grammar_map(receipts_dir=tmp_path)
    assert out["schema"] == "grammar_map.v1"
    assert out["research_only"] is True
    assert out["data_label"] == "MIXED"
    assert out["n_lanes"] == 7
    assert len(out["lanes"]) == 7
    assert set(out["claims"]) == {
        "all_member_receipts_sealed",
        "all_lanes_present",
        "grammar_gap_was_real",
        "ttl_reaches_scale",
        "churn_closes",
        "churn_not_mechanism",
        "residual_is_underfill",
        "latency_closes",
        "joint_frontier_exists",
    }
    # Missing members -> lanes_present False, sealed False.
    assert out["claims"]["all_lanes_present"] is False
    assert out["claims"]["all_member_receipts_sealed"] is False
    assert len(out["receipt_sha256"]) == 64
