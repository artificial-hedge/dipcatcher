from __future__ import annotations

import json
from pathlib import Path

from quant_fund.microstructure.tape_digest import LANE_RECEIPTS, tape_digest


def _write_receipt(dir_: Path, fname: str, divergences: list[str]) -> None:
    (dir_ / fname).write_text(json.dumps({"receipt_sha256": "x" * 64, "divergences": divergences}))


def test_digest_coverage_matrix(tmp_path: Path) -> None:
    _write_receipt(tmp_path, LANE_RECEIPTS["propagator"], ["iid_lag1_gap"])
    _write_receipt(tmp_path, LANE_RECEIPTS["vpin"], [])
    out = tape_digest(tmp_path)
    assert out["n_lanes_present"] == 2
    assert out["lanes"]["propagator"]["arm_status"] == {
        "iid": "diverged",
        "regime": "covered",
        "split": "covered",
    }
    assert out["lanes"]["vpin"]["arm_status"]["iid"] == "covered"
    assert out["best_arm"] in ("regime", "split")


def test_digest_missing_lanes_skipped(tmp_path: Path) -> None:
    out = tape_digest(tmp_path)
    assert out["n_lanes_present"] == 0
    assert out["lanes"]["hawkes"]["present"] is False


def test_uncovered_lanes_listed(tmp_path: Path) -> None:
    _write_receipt(tmp_path, LANE_RECEIPTS["hawkes"], ["iid_x", "regime_x", "split_x"])
    out = tape_digest(tmp_path)
    assert "hawkes" in out["uncovered_lanes"]


def test_sim_mechanism_gap_diverges_all(tmp_path: Path) -> None:
    _write_receipt(tmp_path, LANE_RECEIPTS["hidden_depth"], ["sim_has_no_x"])
    out = tape_digest(tmp_path)
    assert all(s == "diverged" for s in out["lanes"]["hidden_depth"]["arm_status"].values())


def test_unscored_when_no_divergence_field(tmp_path: Path) -> None:
    (tmp_path / LANE_RECEIPTS["vpin"]).write_text(json.dumps({"receipt_sha256": "x" * 64}))
    out = tape_digest(tmp_path)
    assert out["lanes"]["vpin"]["arm_status"]["iid"] == "unscored"


def test_digest_is_sealed(tmp_path: Path) -> None:
    out = tape_digest(tmp_path)
    assert len(out["receipt_sha256"]) == 64
    assert out["data_label"] == "MIXED"


def test_arm_discovery_from_sim_arms(tmp_path: Path) -> None:
    (tmp_path / LANE_RECEIPTS["vpin"]).write_text(
        json.dumps(
            {
                "receipt_sha256": "x" * 64,
                "table": {"sim_arms": {"iid": {}, "deep_split": {}}},
                "divergences": ["deep_split_gap_1.0_vs_0.5"],
            }
        )
    )
    out = tape_digest(tmp_path)
    status = out["lanes"]["vpin"]["arm_status"]
    assert status == {"iid": "covered", "deep_split": "diverged"}
    assert "deep_split" in out["arm_universe"]


def test_markov_regime_aliases_to_regime(tmp_path: Path) -> None:
    (tmp_path / LANE_RECEIPTS["vpin"]).write_text(
        json.dumps(
            {
                "receipt_sha256": "x" * 64,
                "table": {"sim_arms": {"iid": {}, "markov_regime": {}}},
                "divergences": ["markov_regime_gap"],
            }
        )
    )
    out = tape_digest(tmp_path)
    assert out["lanes"]["vpin"]["arm_status"] == {"iid": "covered", "regime": "diverged"}


def test_dict_divergences_attribute(tmp_path: Path) -> None:
    (tmp_path / LANE_RECEIPTS["vpin"]).write_text(
        json.dumps(
            {
                "receipt_sha256": "x" * 64,
                "table": {"sim_arms": {"iid": {}, "regime": {}}},
                "divergences": [
                    {"arm": "iid", "diverges": True},
                    {"arm": "regime", "diverges": False},
                ],
            }
        )
    )
    out = tape_digest(tmp_path)
    assert out["lanes"]["vpin"]["arm_status"] == {"iid": "diverged", "regime": "covered"}


def test_unknown_prefix_does_not_blame_named_arm(tmp_path: Path) -> None:
    _write_receipt(tmp_path, LANE_RECEIPTS["vpin"], ["frobnicate_gap"])
    out = tape_digest(tmp_path)
    assert out["lanes"]["vpin"]["arm_status"]["iid"] == "covered"
