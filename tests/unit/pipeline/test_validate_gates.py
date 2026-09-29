"""Fail-closed quant validate gates."""

from pathlib import Path

import polars as pl
import pytest

from quant_fund.config.loader import load_config
from quant_fund.validation.gates import (
    _multi_fold_stability,
    _positive_integral_count,
    _walk_forward_evidence,
    validate_candidate,
)


@pytest.mark.parametrize("value", [True, False, 1.5, float("nan"), float("inf"), "3", "3.5"])
def test_count_evidence_rejects_coercion_shaped_values(value: object) -> None:
    assert _positive_integral_count(value) == 0
    assert _walk_forward_evidence({"n_folds": value}, None) is False
    stable, reason = _multi_fold_stability({"n_folds": value}, None, min_folds=2)
    assert stable is False
    assert reason is None


def test_count_evidence_accepts_only_positive_integral_counts() -> None:
    assert _positive_integral_count(3) == 3
    assert _positive_integral_count(3.0) == 3
    assert _walk_forward_evidence({"n_folds": 3.0}, None) is True
    assert _multi_fold_stability({"n_folds": 3.0}, None, min_folds=2) == (True, None)


def test_multi_fold_marker_without_count_fails_closed() -> None:
    assert _multi_fold_stability({"walk_forward_complete": True}, None, min_folds=2) == (
        False,
        None,
    )


def test_ranker_date_count_cannot_masquerade_as_fold_count() -> None:
    notebook = {"rankers": [{"name": "ridge", "n_dates": 100, "mean_ic": 0.1}]}
    assert _multi_fold_stability({}, notebook, min_folds=2) == (False, None)


def test_walk_forward_evidence_rejects_malformed_ranker_rows() -> None:
    assert _walk_forward_evidence({}, {"rankers": [{}], "hypotheses": []}) is False
    assert (
        _walk_forward_evidence({}, {"rankers": [{"name": "ridge", "n_dates": 0}], "hypotheses": []})
        is False
    )
    assert (
        _walk_forward_evidence(
            {},
            {"rankers": [{"name": "ridge", "n_dates": 12, "mean_ic": 0.1}], "hypotheses": []},
        )
        is True
    )
    assert (
        _walk_forward_evidence(
            {}, {"rankers": [{"name": "ridge", "n_dates": 12}], "hypotheses": []}
        )
        is False
    )
    assert _walk_forward_evidence({}, {"rankers": [], "hypotheses": [{"id": "H1"}]}) is False


def test_validate_fails_without_evidence(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    result = validate_candidate("model-x", cfg, metrics={})
    assert result["ok"] is False
    assert result["promote"] is False
    assert result["data_label"] == "SYNTHETIC"
    assert "missing_causal_panel_or_walk_forward_evidence" in result["reasons"]


def test_validate_rejects_truthy_string_evidence_flags(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    result = validate_candidate(
        "model-x",
        cfg,
        metrics={
            "evidence_complete": "false",
            "walk_forward_complete": "false",
            "causal_panel": "false",
        },
    )
    assert result["ok"] is False
    assert result["promote"] is False
    assert result["gates"]["causal_or_walk_forward"] is False


def test_validate_rejects_forged_causal_panel_marker(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    result = validate_candidate(
        "model-x",
        cfg,
        metrics={"causal_panel": True, "evidence_complete": True},
    )
    assert result["ok"] is False
    assert result["gates"]["causal_or_walk_forward"] is False
    assert "missing_causal_panel_or_walk_forward_evidence" in result["reasons"]


def test_validate_rejects_forged_walk_forward_marker(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    result = validate_candidate(
        "model-x",
        cfg,
        metrics={"walk_forward_complete": True, "evidence_complete": True},
    )
    assert result["ok"] is False
    assert result["gates"]["causal_or_walk_forward"] is False
    assert "missing_causal_panel_or_walk_forward_evidence" in result["reasons"]


def test_validate_rejects_forged_evidence_complete_marker(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    result = validate_candidate(
        "model-x",
        cfg,
        metrics={
            "evidence_complete": True,
            "n_folds": 2,
            "walk_forward_complete": True,
        },
    )
    assert result["ok"] is False
    assert "research_notebook_missing" in result["reasons"]


def test_validate_rejects_synthetic_as_live(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    # Plant a fake causal panel so the missing-panel gate is not the only failure.
    gold = tmp_path / "gold"
    gold.mkdir(parents=True)
    pl.DataFrame(
        {
            "event_time": ["2024-01-01"],
            "security_id": ["A"],
            "target_weight": [0.01],
        }
    ).write_parquet(gold / "target_weights.parquet")
    result = validate_candidate(
        "model-x",
        cfg,
        metrics={
            "evidence_complete": True,
            "mean_ic": 0.1,
            "net_spread": 0.01,
            "turnover": 0.2,
            "data_source": "SYNTHETIC",
        },
        claim_live=True,
    )
    assert result["ok"] is False
    assert result["promote"] is False
    assert "synthetic_claimed_as_live" in result["reasons"]
    assert "synthetic_evidence_not_promotable" in result["reasons"]


def test_validate_research_ok_with_causal_panel_synthetic(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    gold = tmp_path / "gold"
    gold.mkdir(parents=True)
    pl.DataFrame(
        {
            "event_time": ["2024-01-01"],
            "security_id": ["A"],
            "target_weight": [0.01],
        }
    ).write_parquet(gold / "target_weights.parquet")
    result = validate_candidate(
        "model-x",
        cfg,
        metrics={
            "evidence_complete": True,
            "mean_ic": 0.1,
            "net_spread": 0.01,
            "turnover": 0.2,
            "n_folds": 2,
            "walk_forward_complete": True,
        },
        claim_live=False,
    )
    assert result["ok"] is True
    assert result["promote"] is False
    assert result["data_label"] == "SYNTHETIC"


def test_validate_rejects_invalid_metrics_json(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    path = tmp_path / "metrics.json"
    path.write_text("{not-json")
    result = validate_candidate("model-x", cfg, metrics_path=path)
    assert result["ok"] is False
    assert result["promote"] is False
    assert str(result["reasons"][0]).startswith("metrics_file_invalid:")


def test_validate_rejects_tampered_research_receipt(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.source = "file"  # type: ignore[assignment]
    receipt_dir = tmp_path / "metadata" / "research"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "latest.json").write_text('{"synthetic": false, "claim": "research_only"}')
    result = validate_candidate(
        "model-x",
        cfg,
        metrics={
            "evidence_complete": True,
            "mean_ic": 0.99,
            "net_spread": 0.99,
            "turnover": 0.01,
            "n_folds": 10,
            "walk_forward_complete": True,
        },
    )
    assert result["gates"]["research_notebook_receipt_valid"] is False
    assert result["promote"] is False
    assert "research_notebook_invalid" in result["reasons"]


def test_validate_never_promotes_without_research_receipt(tmp_path: Path) -> None:
    """Complete non-synthetic metrics cannot bypass the immutable receipt gate."""
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.source = "file"  # type: ignore[assignment]
    gold = tmp_path / "gold"
    gold.mkdir(parents=True)
    pl.DataFrame(
        {
            "event_time": ["2024-01-01", "2024-01-02"],
            "security_id": ["A", "A"],
            "target_weight": [0.01, 0.02],
        }
    ).write_parquet(gold / "target_weights.parquet")
    result = validate_candidate(
        "model-x",
        cfg,
        metrics={
            "evidence_complete": True,
            "mean_ic": 0.5,
            "net_spread": 0.2,
            "turnover": 0.1,
            "n_folds": 5,
            "walk_forward_complete": True,
            "data_source": "file",
            "run_id": "run-x",
        },
    )
    assert result["gates"]["research_notebook_receipt_valid"] is False
    assert result["promote"] is False
    assert result["promotion"]["research_receipt_valid"] is False
    assert "research_notebook_missing" in result["reasons"]


def test_validate_rejects_receipt_run_id_mismatch(tmp_path: Path, monkeypatch) -> None:
    """A valid receipt cannot be replayed for a different candidate run."""
    import quant_fund.validation.gates as gates

    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.source = "file"  # type: ignore[assignment]
    gold = tmp_path / "gold"
    gold.mkdir(parents=True)
    pl.DataFrame(
        {
            "event_time": ["2024-01-01", "2024-01-02"],
            "security_id": ["A", "A"],
            "target_weight": [0.01, 0.02],
        }
    ).write_parquet(gold / "target_weights.parquet")
    receipt_run_id = "a" * 64
    monkeypatch.setattr(
        gates,
        "_load_research_notebook",
        lambda _config: {"provenance": {"run_id": receipt_run_id}, "hypotheses": []},
    )
    monkeypatch.setattr(gates, "verify_research_artifact", lambda _path: {"valid": True})
    result = gates.validate_candidate(
        "model-x",
        cfg,
        metrics={
            "evidence_complete": True,
            "mean_ic": 0.5,
            "net_spread": 0.2,
            "turnover": 0.1,
            "n_folds": 5,
            "walk_forward_complete": True,
            "data_source": "file",
            "run_id": "b" * 64,
        },
    )
    assert result["gates"]["research_receipt_run_id_bound"] is False
    assert result["promote"] is False
    assert "research_receipt_run_id_unbound" in result["reasons"]


def test_validate_rejects_receipt_from_stale_worktree(tmp_path: Path, monkeypatch) -> None:
    """A receipt minted before checkout changes cannot authorize promotion."""
    import quant_fund.validation.gates as gates

    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.source = "file"  # type: ignore[assignment]
    monkeypatch.setattr(
        gates,
        "_load_research_notebook",
        lambda _config: {
            "provenance": {
                "run_id": "a" * 64,
                "git_worktree_sha256": "b" * 64,
            },
            "hypotheses": [],
        },
    )
    monkeypatch.setattr(gates, "verify_research_artifact", lambda _path: {"valid": True})
    monkeypatch.setattr(gates, "_current_worktree_sha256", lambda: "c" * 64)
    result = gates.validate_candidate(
        "model-x",
        cfg,
        metrics={"run_id": "a" * 64, "evidence_complete": True},
    )
    assert result["gates"]["research_receipt_worktree_bound"] is False
    assert result["promote"] is False
    assert "research_receipt_worktree_unbound" in result["reasons"]


def test_validate_rejects_missing_metrics_file(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    result = validate_candidate("model-x", cfg, metrics_path=tmp_path / "missing.json")
    assert result["ok"] is False
    assert result["promote"] is False
    assert str(result["reasons"][0]).startswith("metrics_file_missing:")


def test_validate_rejects_nonfinite_causal_panel(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    gold = tmp_path / "gold"
    gold.mkdir(parents=True)
    pl.DataFrame(
        {
            "event_time": ["2024-01-01"],
            "security_id": ["A"],
            "target_weight": [float("nan")],
        }
    ).write_parquet(gold / "target_weights.parquet")
    result = validate_candidate(
        "model-x",
        cfg,
        metrics={"evidence_complete": True, "mean_ic": 0.1, "net_spread": 0.01, "turnover": 0.2},
    )
    assert result["gates"]["causal_or_walk_forward"] is False
    assert result["ok"] is False


@pytest.mark.parametrize(
    "rows",
    [
        {"event_time": [None], "security_id": ["A"], "target_weight": [0.01]},
        {"event_time": ["2024-01-01"], "security_id": [""], "target_weight": [0.01]},
        {
            "event_time": ["2024-01-01", "2024-01-01"],
            "security_id": ["A", "A"],
            "target_weight": [0.01, 0.01],
        },
    ],
)
def test_validate_rejects_ambiguous_causal_panel(
    tmp_path: Path, rows: dict[str, list[object]]
) -> None:
    """Null/empty keys and duplicate as-of rows are not causal evidence."""
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    gold = tmp_path / "gold"
    gold.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(gold / "target_weights.parquet")
    result = validate_candidate(
        "model-x",
        cfg,
        metrics={"evidence_complete": True, "mean_ic": 0.1, "net_spread": 0.01, "turnover": 0.2},
    )
    assert result["gates"]["causal_or_walk_forward"] is False
    assert result["ok"] is False


def test_validate_synthetic_never_promotes_even_with_perfect_metrics(tmp_path: Path) -> None:
    """SYNTHETIC + perfect evidence still promote=false (research_ok may be true)."""
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    gold = tmp_path / "gold"
    gold.mkdir(parents=True)
    pl.DataFrame(
        {
            "event_time": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "security_id": ["A", "A", "A"],
            "target_weight": [0.01, 0.02, 0.015],
        }
    ).write_parquet(gold / "target_weights.parquet")
    result = validate_candidate(
        "model-perfect-synth",
        cfg,
        metrics={
            "evidence_complete": True,
            "mean_ic": 0.99,
            "net_spread": 0.99,
            "turnover": 0.01,
            "n_folds": 10,
            "fold_ic_stability": 1.0,
            "walk_forward_complete": True,
            "data_source": "SYNTHETIC",
        },
        claim_live=False,
    )
    assert result["ok"] is True
    assert result["promote"] is False
    assert result["data_label"] == "SYNTHETIC"
    assert "synthetic_evidence_not_promotable" in result["reasons"]
    assert result["gates"]["promotion"] is False


def test_validate_claim_live_fails_closed_on_config_synthetic(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    assert str(cfg.data.source).lower() == "synthetic"
    cfg.data.root = tmp_path
    gold = tmp_path / "gold"
    gold.mkdir(parents=True)
    pl.DataFrame(
        {
            "event_time": ["2024-01-01"],
            "security_id": ["A"],
            "target_weight": [0.01],
        }
    ).write_parquet(gold / "target_weights.parquet")
    result = validate_candidate(
        "model-x",
        cfg,
        metrics={
            "evidence_complete": True,
            "mean_ic": 0.5,
            "net_spread": 0.1,
            "turnover": 0.1,
            "n_folds": 5,
            "walk_forward_complete": True,
        },
        claim_live=True,
    )
    assert result["ok"] is False
    assert result["promote"] is False
    assert "synthetic_claimed_as_live" in result["reasons"]
    assert result["gates"]["synthetic_not_claimed_live"] is False


def test_validate_insufficient_folds_fails_research_ok(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    gold = tmp_path / "gold"
    gold.mkdir(parents=True)
    pl.DataFrame(
        {
            "event_time": ["2024-01-01"],
            "security_id": ["A"],
            "target_weight": [0.01],
        }
    ).write_parquet(gold / "target_weights.parquet")
    result = validate_candidate(
        "model-x",
        cfg,
        metrics={
            "evidence_complete": True,
            "mean_ic": 0.2,
            "net_spread": 0.05,
            "turnover": 0.1,
            "n_folds": 1,  # min_folds default 2
        },
        claim_live=False,
    )
    assert result["gates"]["multi_fold_stability"] is False
    assert result["ok"] is False
    assert result["promote"] is False
    assert "insufficient_multi_fold_stability" in result["reasons"]


def test_validate_metrics_synthetic_flag_blocks_promote(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    # Force non-synthetic config source to isolate metrics.synthetic flag
    cfg.data.source = "file"  # type: ignore[assignment]
    gold = tmp_path / "gold"
    gold.mkdir(parents=True)
    pl.DataFrame(
        {
            "event_time": ["2024-01-01"],
            "security_id": ["A"],
            "target_weight": [0.01],
        }
    ).write_parquet(gold / "target_weights.parquet")
    result = validate_candidate(
        "model-x",
        cfg,
        metrics={
            "evidence_complete": True,
            "mean_ic": 0.5,
            "net_spread": 0.2,
            "turnover": 0.1,
            "n_folds": 5,
            "synthetic": True,
            "walk_forward_complete": True,
        },
        claim_live=False,
    )
    assert result["synthetic"] is True
    assert result["promote"] is False
    assert "synthetic_evidence_not_promotable" in result["reasons"]


@pytest.mark.parametrize("count", [0, -1, 1.5, "not-a-count"])
def test_walk_forward_count_metadata_must_be_positive_integer(
    tmp_path: Path, count: object
) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    result = validate_candidate(
        "model-x",
        cfg,
        metrics={"n_ic_dates": count},
        claim_live=False,
    )
    assert result["gates"]["causal_or_walk_forward"] is False
    assert "missing_causal_panel_or_walk_forward_evidence" in result["reasons"]


# ---------------------------------------------------------------------------
# hypothesis_family_split: computed AND enforced (SOTA wave 28)
#
# A notebook that registers hypotheses without pre-registered families cannot
# separate "we predicted this" from "we found this". The gate used to be
# computed into `gates` + `reasons` only, leaving `ok=True`/`promote=True` —
# a computed-but-unenforced gate. These lock the enforcement.
# ---------------------------------------------------------------------------

_COMPLETE_NON_SYNTHETIC = {
    "evidence_complete": True,
    "mean_ic": 0.5,
    "net_spread": 0.2,
    "turnover": 0.1,
    "n_folds": 5,
    "fold_ic_stability": 1.0,
    "walk_forward_complete": True,
    "data_source": "file",
    "run_id": "a" * 64,
}


def _bound_candidate(tmp_path: Path, monkeypatch, notebook: dict) -> dict:
    """A receipt-valid, run-id/worktree-bound, non-synthetic candidate.

    Everything except the hypothesis family split is green, so the family-split
    gate is the only thing that can move the verdict.
    """
    import quant_fund.validation.gates as gates

    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.source = "file"  # type: ignore[assignment]
    monkeypatch.setattr(gates, "_load_research_notebook", lambda _config: notebook)
    monkeypatch.setattr(gates, "verify_research_artifact", lambda _path: {"valid": True})
    monkeypatch.setattr(gates, "_current_worktree_sha256", lambda: "b" * 64)
    return gates.validate_candidate("model-x", cfg, metrics=dict(_COMPLETE_NON_SYNTHETIC))


@pytest.mark.parametrize("family", ["calibration", "discovery", "bound"])
def test_family_split_present_still_validates(tmp_path: Path, monkeypatch, family: str) -> None:
    """Each pre-registered family on its own satisfies the split gate."""
    notebook = {
        "provenance": {"run_id": "a" * 64, "git_worktree_sha256": "b" * 64},
        "hypotheses": [{"id": "H1", "family": family, "p_value": 0.01}],
    }
    result = _bound_candidate(tmp_path, monkeypatch, notebook)
    assert result["gates"]["hypothesis_family_split"] is True
    assert result["ok"] is True
    assert result["promote"] is True
    assert result["gates"]["promotion"] is True
    assert "hypothesis_family_split_missing" not in result["reasons"]


@pytest.mark.parametrize(
    "hypotheses",
    [
        # families stripped entirely (legacy / migrated / hand-authored receipt)
        [{"id": "H1", "p_value": 0.01}, {"id": "H2", "p_value": 0.02}],
        # a family that is not one of the three pre-registered ones
        [{"id": "H1", "family": "post_hoc", "p_value": 0.01}],
        # family present but None — cannot be coerced into a valid split
        [{"id": "H1", "family": None, "p_value": 0.01}],
        # non-dict rows carrying no family attribute
        ["H1", "H2"],
    ],
)
def test_hypotheses_without_family_split_are_rejected(
    tmp_path: Path, monkeypatch, hypotheses: list[object]
) -> None:
    """PERFECT IC + bound receipt still REJECTED when the family split is missing.

    This is the regression lock for the computed-but-unenforced gate: before the
    fix this candidate returned ``ok=True`` and ``promote=True`` while the
    reason ``hypothesis_family_split_missing`` sat unread in the ledger.
    """
    notebook = {
        "provenance": {"run_id": "a" * 64, "git_worktree_sha256": "b" * 64},
        "hypotheses": hypotheses,
    }
    result = _bound_candidate(tmp_path, monkeypatch, notebook)
    assert result["gates"]["hypothesis_family_split"] is False
    assert result["ok"] is False
    assert result["promote"] is False
    assert result["gates"]["promotion"] is False
    assert result["promotion"]["promote"] is False
    assert "hypothesis_family_split_missing" in result["reasons"]
    # The reason must not be the only signal: the gate dict must agree with it.
    assert result["gates"]["promotion"] is False


def test_no_hypotheses_does_not_trip_the_family_split_gate(tmp_path: Path, monkeypatch) -> None:
    """An empty hypothesis list is not a family-split violation (fail-closed elsewhere)."""
    notebook = {
        "provenance": {"run_id": "a" * 64, "git_worktree_sha256": "b" * 64},
        "hypotheses": [],
    }
    result = _bound_candidate(tmp_path, monkeypatch, notebook)
    assert result["gates"]["hypothesis_family_split"] is False
    assert "hypothesis_family_split_missing" not in result["reasons"]
    assert result["ok"] is True


def test_family_split_gate_blocks_even_when_other_gates_all_pass(
    tmp_path: Path, monkeypatch
) -> None:
    """Mixed families where NONE is pre-registered: gate computed, verdict flipped."""
    notebook = {
        "provenance": {"run_id": "a" * 64, "git_worktree_sha256": "b" * 64},
        "hypotheses": [
            {"id": "H1", "family": "oracle", "p_value": 0.001},
            {"id": "H2", "family": "snooping", "p_value": 0.001},
            {"id": "H3", "family": "vol_tail", "p_value": 0.001},
        ],
    }
    result = _bound_candidate(tmp_path, monkeypatch, notebook)
    # Every other gate is green.
    assert result["gates"]["causal_or_walk_forward"] is True
    assert result["gates"]["synthetic_not_claimed_live"] is True
    assert result["gates"]["multi_fold_stability"] is True
    assert result["gates"]["research_notebook_receipt_valid"] is True
    assert result["gates"]["research_receipt_run_id_bound"] is True
    assert result["gates"]["research_receipt_worktree_bound"] is True
    # ...and the single computed gate is enough to reject.
    assert result["gates"]["hypothesis_family_split"] is False
    assert result["ok"] is False
    assert result["promote"] is False
