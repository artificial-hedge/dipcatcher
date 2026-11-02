"""Fail-closed validation inputs, evidence decoding, and optimizer timeouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quant_fund.config.loader import load_config
from quant_fund.validation.agent_referee import (
    RefereeLedger,
    RefereeVerdict,
    frozen_referee_evalue,
    referee_fdr_control,
)
from quant_fund.validation.gates import _load_metrics, _load_research_notebook
from quant_fund.validation.optuna_guard import tune_on_train


def _objective_factory(x: np.ndarray, y: np.ndarray) -> Any:
    return lambda trial: 0.0


@pytest.mark.parametrize("n_trials", [1.5, True, float("nan"), float("inf")])
def test_optuna_rejects_noninteger_trial_counts(n_trials: Any) -> None:
    with pytest.raises(ValueError, match="n_trials must be an integer"):
        tune_on_train(_objective_factory, np.ones((2, 1)), np.ones(2), n_trials, 0)


@pytest.mark.parametrize("timeout_s", [0.0, -1.0, float("nan"), float("inf")])
def test_optuna_rejects_invalid_timeouts(timeout_s: float) -> None:
    with pytest.raises(ValueError, match="timeout_s must be finite"):
        tune_on_train(
            _objective_factory,
            np.ones((2, 1)),
            np.ones(2),
            1,
            0,
            timeout_s=timeout_s,
        )


def test_optuna_forwards_bounded_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class Study:
        trials = [object()]
        best_params = {"score": 1.0}
        best_value = 1.0

        def optimize(self, objective: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

    study = Study()
    monkeypatch.setattr(
        "quant_fund.validation.optuna_guard.optuna.create_study", lambda **kwargs: study
    )
    result = tune_on_train(
        _objective_factory,
        np.ones((2, 1)),
        np.ones(2),
        np.int64(2),
        7,
        timeout_s=1.25,
    )
    assert result["n_trials"] == 2
    assert captured == {"n_trials": 2, "show_progress_bar": False, "timeout": 1.25}


@pytest.mark.parametrize(
    "options",
    [
        {"null_mean": float("nan")},
        {"null_mean": float("inf")},
        {"null_std": float("nan")},
        {"null_std": float("inf")},
        {"bet_fraction": float("nan")},
        {"bet_fraction": float("inf")},
    ],
)
def test_referee_evalue_rejects_nonfinite_parameters(options: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="finite"):
        frozen_referee_evalue(np.array([1.0]), **options)


def test_referee_ledger_rejects_nonfinite_configuration() -> None:
    with pytest.raises(ValueError, match="null_std must be finite"):
        RefereeLedger(null_std=float("nan"))
    with pytest.raises(ValueError, match="alpha must be finite"):
        RefereeLedger(alpha=float("inf"))


@pytest.mark.parametrize("e_value", [float("nan"), float("inf"), -1.0])
def test_referee_fdr_rejects_invalid_evalues(e_value: float) -> None:
    with pytest.raises(ValueError, match="e-values must be finite"):
        referee_fdr_control([RefereeVerdict("candidate", e_value, 1)])


def test_referee_fdr_rejects_nonfinite_alpha() -> None:
    with pytest.raises(ValueError, match="alpha must be finite"):
        referee_fdr_control([], alpha=float("nan"))


def test_metrics_invalid_utf8_is_invalid_evidence(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_bytes(b"\xff")
    assert _load_metrics(path) == {"_metrics_file_invalid": True, "path": str(path)}


def test_notebook_invalid_utf8_is_invalid_evidence(tmp_path: Path) -> None:
    config = load_config("configs/research.yaml")
    config.data.root = tmp_path
    path = tmp_path / "metadata" / "research" / "latest.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xff")
    assert _load_research_notebook(config) == {"_notebook_invalid": True}
