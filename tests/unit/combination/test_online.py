import numpy as np
import pytest

from quant_fund.combination.online import (
    exponentiated_gradient,
    fixed_share,
    regret_bound,
)

pytestmark = pytest.mark.synthetic


def test_eg_concentrates_on_better_member() -> None:
    rng = np.random.default_rng(110)
    n = 2000
    losses = np.column_stack([rng.uniform(0.0, 0.2, n), rng.uniform(0.5, 0.7, n)])
    out = exponentiated_gradient(losses, eta=2.0)
    assert out["weights"][-1, 0] > 0.99
    # combined loss approaches the better member's mean
    assert float(np.mean(out["combined_loss"])) < 0.25


def test_eg_regret_within_bound() -> None:
    rng = np.random.default_rng(111)
    n = 500
    losses = rng.uniform(0.0, 1.0, (n, 4))
    eta = 0.5
    out = exponentiated_gradient(losses, eta=eta)
    cum_comb = float(np.sum(out["combined_loss"]))
    cum_members = np.cumsum(losses, axis=0)[-1]
    regret = cum_comb - float(np.min(cum_members))
    assert regret <= regret_bound(n, 4, eta, 1.0) + 1e-9
    assert regret >= -1e-9


def test_fixed_share_tracks_drifting_best() -> None:
    rng = np.random.default_rng(112)
    n = 1200
    losses = np.empty((n, 2))
    losses[: n // 2, 0] = rng.uniform(0.0, 0.2, n // 2)
    losses[: n // 2, 1] = rng.uniform(0.6, 0.8, n // 2)
    losses[n // 2 :, 0] = rng.uniform(0.6, 0.8, n - n // 2)
    losses[n // 2 :, 1] = rng.uniform(0.0, 0.2, n - n // 2)
    out = fixed_share(losses, eta=2.0, alpha=0.05)
    # after the switch the weight on the new best member recovers
    assert out["weights"][-1, 1] > 0.8


def test_pure_eg_recovers_slower_after_switch() -> None:
    # EG without fixed share needs many rounds to re-concentrate; the fixed
    # share variant should beat it right after a regime switch
    rng = np.random.default_rng(113)
    n = 1200
    losses = np.empty((n, 2))
    losses[: n // 2, 0] = rng.uniform(0.0, 0.2, n // 2)
    losses[: n // 2, 1] = rng.uniform(0.6, 0.8, n // 2)
    losses[n // 2 :, 0] = rng.uniform(0.6, 0.8, n - n // 2)
    losses[n // 2 :, 1] = rng.uniform(0.0, 0.2, n - n // 2)
    eg = exponentiated_gradient(losses, eta=2.0)
    fs = fixed_share(losses, eta=2.0, alpha=0.1)
    eg_post = float(np.mean(eg["combined_loss"][n // 2 : n // 2 + 100]))
    fs_post = float(np.mean(fs["combined_loss"][n // 2 : n // 2 + 100]))
    assert fs_post < eg_post


def test_regret_bound_grows_linearly() -> None:
    assert regret_bound(200, 4, 0.5, 1.0) > regret_bound(100, 4, 0.5, 1.0)


def test_validation() -> None:
    with pytest.raises(ValueError):
        exponentiated_gradient(np.zeros((3,)), 1.0)
    with pytest.raises(ValueError):
        fixed_share(np.zeros((3, 2)), 1.0, 1.5)
    with pytest.raises(ValueError):
        regret_bound(0, 4, 0.5, 1.0)
