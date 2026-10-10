import numpy as np
import pytest

from quant_fund.combination.diversity_metrics import (
    ambiguity_decomposition,
    diversity_bonus,
    effective_ensemble_size,
)

pytestmark = pytest.mark.synthetic


def test_ambiguity_decomposition_identity() -> None:
    rng = np.random.default_rng(70)
    n = 2000
    y = rng.standard_normal(n)
    f = np.column_stack([y + rng.standard_normal(n) * 0.4, y + rng.standard_normal(n) * 0.8])
    w = np.array([0.6, 0.4])
    out = ambiguity_decomposition(f, y, w)
    # ensemble MSE = member-weighted MSE − diversity (exact identity)
    assert out["ensemble_mse"] == pytest.approx(out["member_mse"] - out["diversity_term"], abs=1e-9)
    assert abs(out["residual"]) < 1e-9


def test_diversity_bonus_positive_with_spread_members() -> None:
    rng = np.random.default_rng(71)
    n = 2000
    y = rng.standard_normal(n)
    f = np.column_stack([y + 0.3 * rng.standard_normal(n), y - 0.3 * rng.standard_normal(n)])
    assert diversity_bonus(f, y, np.array([0.5, 0.5])) > 0.05


def test_diversity_bonus_zero_for_identical_members() -> None:
    rng = np.random.default_rng(72)
    n = 1000
    y = rng.standard_normal(n)
    f = np.column_stack([y + 0.3 * rng.standard_normal(n), y + 0.3 * rng.standard_normal(n)])
    # members share the same noise-free signal but are identical copies
    f[:, 1] = f[:, 0]
    assert diversity_bonus(f, y, np.array([0.5, 0.5])) == 0.0


def test_effective_ensemble_size() -> None:
    assert effective_ensemble_size(np.array([0.5, 0.5])) == pytest.approx(2.0)
    assert effective_ensemble_size(np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert effective_ensemble_size(np.array([0.7, 0.3])) == pytest.approx(1.0 / 0.58)


def test_effective_size_uniform_n_members() -> None:
    assert effective_ensemble_size(np.full(5, 0.2)) == pytest.approx(5.0)


def test_validation() -> None:
    with pytest.raises(ValueError):
        ambiguity_decomposition(np.zeros((5, 2)), np.zeros(4), np.ones(2))
    with pytest.raises(ValueError):
        effective_ensemble_size(np.zeros((2, 2)))
