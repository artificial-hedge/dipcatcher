"""Optuna may only see training data. Holdout is not passed in."""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Integral
from typing import Any

import numpy as np
import optuna


def tune_on_train(
    objective_factory: Callable[[np.ndarray, np.ndarray], Callable[[optuna.Trial], float]],
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int,
    seed: int,
    *,
    x_holdout: np.ndarray | None = None,
    y_holdout: np.ndarray | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Run Optuna on train only; fail closed on holdout leak / empty / bad space."""
    if x_holdout is not None or y_holdout is not None:
        raise ValueError("Holdout data must not be passed to tune_on_train")
    if isinstance(n_trials, bool) or not isinstance(n_trials, Integral) or n_trials < 1:
        raise ValueError("n_trials must be an integer >= 1")
    if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s <= 0):
        raise ValueError("timeout_s must be finite and > 0")
    trials = int(n_trials)
    x = np.asarray(x_train)
    y = np.asarray(y_train).reshape(-1)
    if x.size == 0 or y.size == 0:
        raise ValueError("empty train arrays are not allowed for Optuna")
    if x.ndim < 1:
        raise ValueError("x_train must be at least 1-d")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x_train and y_train length mismatch")
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective_factory(x, y), n_trials=trials, show_progress_bar=False, timeout=timeout_s
    )
    if len(study.trials) == 0:
        raise ValueError("empty Optuna study after optimize")
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": trials,
        "seed": seed,
    }
