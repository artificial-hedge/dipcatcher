"""Torch RNG isolation probes (SYNTHETIC correctness, no market evidence).

Library entry points that seed torch for their own init draws must do so
inside ``torch.random.fork_rng``: mutating the caller's global stream makes
unrelated downstream draws silently depend on library call order.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="requires the nn extra (torch)")

_PROBE_SEED = 20260707


def _assert_stream_untouched(call) -> None:
    import torch

    torch.manual_seed(_PROBE_SEED)
    reference = torch.randn(16)
    torch.manual_seed(_PROBE_SEED)
    call()
    resumed = torch.randn(16)
    assert torch.equal(resumed, reference)


def test_policy_gradient_ranker_init_keeps_global_stream() -> None:
    from quant_fund.models.deep_rl import PolicyGradientRanker

    _assert_stream_untouched(
        lambda: PolicyGradientRanker(n_features=3, hidden=(4, 4), epochs=1, seed=5)
    )


def test_c51_agent_init_keeps_global_stream() -> None:
    from quant_fund.models.c51_rl import C51Agent, C51AgentConfig

    cfg = C51AgentConfig(seed=3, n_atoms=11, hidden=(8,))
    _assert_stream_untouched(lambda: C51Agent(3, 2, cfg))


def test_deep_hedge_keeps_global_stream() -> None:
    from quant_fund.models.deep_hedging import deep_hedge

    rng = np.random.default_rng(0)
    paths = 100.0 * np.exp(np.cumsum(rng.standard_normal((64, 5)) * 0.01, axis=1))
    payoff = np.maximum(paths[:, -1] - 100.0, 0.0)
    _assert_stream_untouched(
        lambda: deep_hedge(paths, payoff, cost_rate=0.0, hidden=(4,), epochs=2, seed=1)
    )


def test_deep_bsde_solve_keeps_global_stream() -> None:
    from quant_fund.models import deep_bsde as db

    prob = db.BSDEProblem(
        name="toy",
        dim=1,
        horizon=0.5,
        n_steps=2,
        x0=np.zeros(1),
        mu=lambda t, x: 0.0 * x,
        sigma=lambda t, x: np.ones((x.shape[0], 1, 1)),
        f=lambda t, x, y, z: 0.0 * y,
        g=lambda x: x[:, 0],
    )
    _assert_stream_untouched(
        lambda: db.deep_bsde_solve(prob, hidden=(4,), epochs=1, n_paths=32, eval_paths=16, seed=0)
    )


def test_deep_kernel_hedge_keeps_global_stream() -> None:
    from quant_fund.models.deep_kernel_hedging import deep_kernel_hedge

    rng = np.random.default_rng(1)
    paths = 100.0 * np.exp(np.cumsum(rng.standard_normal((48, 6)) * 0.01, axis=1))
    payoff = np.maximum(paths[:, -1] - 100.0, 0.0)
    _assert_stream_untouched(
        lambda: deep_kernel_hedge(
            paths, payoff, hidden=(4,), p_dim=4, n_rff=8, order=1, epochs=2, seed=0
        )
    )


def test_diffpts_fit_keeps_global_stream() -> None:
    from quant_fund.models.diffpts import DiffPTSModel

    rng = np.random.default_rng(2)
    x = rng.standard_normal((24, 3))
    y = x @ np.array([0.3, -0.2, 0.1]) + 0.5 * rng.standard_normal(24)
    _assert_stream_untouched(
        lambda: DiffPTSModel(n_steps=4, hidden=(4,), epochs=2, seed=0).fit(x, y)
    )


def test_fit_deregime_keeps_global_stream() -> None:
    from quant_fund.models.deep_regime_mixture import fit_deregime

    rng = np.random.default_rng(4)
    x = rng.standard_normal((40, 3))
    y = x[:, :1] * 0.5 + 0.3 * rng.standard_normal((40, 2))
    _assert_stream_untouched(
        lambda: fit_deregime(x, y, n_regimes=2, hidden=(4,), epochs=2, n_nodes=4, seed=0)
    )
