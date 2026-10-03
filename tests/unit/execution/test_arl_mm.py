"""Tests for quant_fund.execution.arl_mm — ARLMM (Yang & Xu 2026).

References: Yang & Xu (2026, arXiv:2609.22785, robust market making as a
zero-sum stochastic game with Hawkes order flow and trade impact);
Pinto et al. (2017, ICML, arXiv:1703.02702, RARL); Avellaneda & Stoikov
(2008, Quantitative Finance 8); Hawkes (1971, Biometrika 58); Ogata (1981,
IEEE Trans. Inf. Theory 27); Williams (1992, Machine Learning 8,
REINFORCE); Hochreiter & Schmidhuber (1997, Neural Computation 9).

All data here is SYNTHETIC (seeded Hawkes/GBM market-making simulator) —
correctness evidence for the algorithm, never market evidence; no
live-trading claims; all mark-to-market accounting stays namespaced
sim_internal_*. Torch tests skip cleanly when the nn extra is absent.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from quant_fund.execution import arl_mm as am


def _torch_present() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # blocked or halted torch imports
        return False


_HAS_TORCH = _torch_present()
requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="ARLMM torch training requires the nn extra (torch)"
)

P = am.ArlMMEnvParams()
B = am.AdversaryBudget()


def _flat_episode(params: am.ArlMMEnvParams | None = None, seed: int = 0) -> am.EpisodeResult:
    p = params or P
    acts = np.full((p.n_steps, 2), 2.0)
    return am.run_episode(p, acts, seed=seed)


# ---------------------------------------------------------------------------
# Fail-closed parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        {"s0": 0.0},
        {"tick": 0.0},
        {"sigma": -0.1},
        {"jump_rate": -1.0},
        {"jump_scale": 0.0},
        {"hawkes_mu": 0.0},
        {"hawkes_alpha": 1.0},
        {"hawkes_alpha": -0.1},
        {"hawkes_beta": 0.0},
        {"p_buy": 1.5},
        {"p_buy": -0.1},
        {"mean_mo_size": 0.0},
        {"impact_eta": -1.0},
        {"k_fill": 0.0},
        {"inventory_cap": 0},
        {"risk_aversion": -0.1},
        {"terminal_penalty": -0.1},
        {"max_quote_ticks": 0},
        {"horizon": 0.0},
        {"n_steps": 1},
    ],
)
def test_env_params_fail_closed(kw: dict) -> None:
    with pytest.raises(ValueError):
        am.ArlMMEnvParams(**kw)


def test_env_params_dt_and_frozen() -> None:
    p = am.ArlMMEnvParams(horizon=20.0, n_steps=10)
    assert p.dt == pytest.approx(2.0)
    with pytest.raises(FrozenInstanceError):
        p.sigma = 0.5  # type: ignore[misc]


@pytest.mark.parametrize(
    "kw",
    [
        {"mu_range": (2.0, 1.0)},
        {"mu_range": (0.0, 2.0)},
        {"sigma_range": (-1.0, 2.0)},
        {"eta_range": (1.0, 1.0)},
        {"alpha_range": (0.0, 1.0)},
        {"alpha_range": (-0.5, 0.8)},
        {"alpha_range": (0.8, 0.2)},
        {"mu_range": [0.5, 2.0]},
        {"mu_range": (0.5,)},
        {"mu_range": (0.5, float("nan"))},
    ],
)
def test_adversary_budget_fail_closed(kw: dict) -> None:
    with pytest.raises(ValueError):
        am.AdversaryBudget(**kw)


def test_adversary_budget_valid() -> None:
    b = am.AdversaryBudget(mu_range=(0.25, 4.0), alpha_range=(0.1, 0.8))
    assert b.mu_range == (0.25, 4.0)
    assert b.alpha_range == (0.1, 0.8)


# ---------------------------------------------------------------------------
# Perturbation map (closed-form interpolation)
# ---------------------------------------------------------------------------


def test_apply_perturbation_identity() -> None:
    out = am.apply_perturbation(P, B, np.zeros(4))
    assert out == P


def test_apply_perturbation_corners() -> None:
    hi = am.apply_perturbation(P, B, np.ones(4))
    assert hi.hawkes_mu == pytest.approx(P.hawkes_mu * 2.0)
    assert hi.hawkes_alpha == pytest.approx(0.9)
    assert hi.sigma == pytest.approx(P.sigma * 3.0)
    assert hi.impact_eta == pytest.approx(P.impact_eta * 20.0)
    lo = am.apply_perturbation(P, B, -np.ones(4))
    assert lo.hawkes_mu == pytest.approx(P.hawkes_mu * 0.5)
    assert lo.hawkes_alpha == pytest.approx(0.0)
    assert lo.sigma == pytest.approx(P.sigma * 0.5)
    assert lo.impact_eta == pytest.approx(P.impact_eta * 0.25)


def test_apply_perturbation_geometric_midpoint() -> None:
    out = am.apply_perturbation(P, B, np.asarray([0.0, 0.0, 0.0, 0.5]))
    # a=+0.5 on a multiplier interpolates geometrically: eta * sqrt(hi)
    assert out.impact_eta == pytest.approx(P.impact_eta * math.sqrt(20.0))
    a2 = np.asarray([0.0, 0.5, 0.0, 0.0])
    out2 = am.apply_perturbation(P, B, a2)
    # alpha interpolates linearly toward the hi absolute bound
    assert out2.hawkes_alpha == pytest.approx(P.hawkes_alpha + 0.5 * (0.9 - P.hawkes_alpha))


@pytest.mark.parametrize(
    "a",
    [
        np.zeros(3),
        np.zeros(5),
        np.asarray([2.0, 0.0, 0.0, 0.0]),
        np.asarray([0.0, -1.5, 0.0, 0.0]),
        np.asarray([0.0, 0.0, float("nan"), 0.0]),
        np.asarray([0.0, 0.0, 0.0, float("inf")]),
    ],
)
def test_apply_perturbation_bad_actions(a: np.ndarray) -> None:
    with pytest.raises(ValueError):
        am.apply_perturbation(P, B, a)


def test_apply_perturbation_type_errors() -> None:
    with pytest.raises(TypeError):
        am.apply_perturbation("not_params", B, np.zeros(4))
    with pytest.raises(TypeError):
        am.apply_perturbation(P, "not_budget", np.zeros(4))


def test_perturbed_params_revalidated() -> None:
    # every point in the budget box must still satisfy env validation
    rng = np.random.default_rng(7)
    for _ in range(8):
        a = rng.uniform(-1.0, 1.0, 4)
        out = am.apply_perturbation(P, B, a)
        assert 0.0 <= out.hawkes_alpha < 1.0
        assert out.hawkes_mu > 0.0
        assert out.sigma > 0.0
        assert out.impact_eta >= 0.0


# ---------------------------------------------------------------------------
# Environment mechanics
# ---------------------------------------------------------------------------


def test_env_reset_and_obs() -> None:
    env = am.ArlMMEnv(P)
    env.reset(0)
    obs = env.obs()
    assert obs.shape == (am.MAKER_OBS_DIM,)
    assert np.all(np.isfinite(obs))
    assert obs[0] == pytest.approx(0.0)  # t_frac at reset
    assert obs[1] == pytest.approx(0.0)  # q / cap at reset
    assert env.mid == pytest.approx(P.s0)
    assert env.inventory == 0
    assert not env.done


def test_env_requires_reset() -> None:
    env = am.ArlMMEnv(P)
    with pytest.raises(RuntimeError):
        env.obs()
    with pytest.raises(RuntimeError):
        env.step((1.0, 1.0))
    with pytest.raises(RuntimeError):
        env.result()


def test_env_step_action_validation() -> None:
    env = am.ArlMMEnv(P)
    env.reset(0)
    with pytest.raises(ValueError):
        env.step((1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        env.step((-0.5, 1.0))
    with pytest.raises(ValueError):
        env.step((1.0, float("nan")))
    with pytest.raises(ValueError):
        env.step((1.0, P.max_quote_ticks + 0.5))


def test_env_done_and_result_fail_closed() -> None:
    env = am.ArlMMEnv(P)
    env.reset(0)
    with pytest.raises(RuntimeError):
        env.result()
    for _ in range(P.n_steps):
        env.step((2.0, 2.0))
    assert env.done
    res = env.result()
    assert math.isfinite(res.sim_internal_return)
    with pytest.raises(RuntimeError):
        env.step((1.0, 1.0))


def test_env_reset_determinism() -> None:
    e1, e2 = am.ArlMMEnv(P), am.ArlMMEnv(P)
    e1.reset(11)
    e2.reset(11)
    np.testing.assert_array_equal(e1.mo_times, e2.mo_times)
    np.testing.assert_array_equal(e1.obs(), e2.obs())
    for _ in range(P.n_steps):
        e1.step((2.0, 2.0))
        e2.step((2.0, 2.0))
    assert e1.result().sim_internal_return == pytest.approx(e2.result().sim_internal_return)


def test_env_zero_offset_fills_every_mo() -> None:
    p = am.ArlMMEnvParams(p_buy=1.0, inventory_cap=1000)
    env = am.ArlMMEnv(p)
    env.reset(3)
    for _ in range(p.n_steps):
        env.step((0.0, 0.0))
    res = env.result()
    # P(fill) = exp(0) = 1 on every buy MO; all are sells for the maker.
    assert res.n_fills_ask == res.n_mo
    assert res.n_fills_bid == 0
    assert res.terminal_inventory == -res.n_mo
    assert res.sim_internal_wealth_path.shape == (p.n_steps + 1,)


def test_env_inventory_cap_gates_fills() -> None:
    p = am.ArlMMEnvParams(p_buy=1.0, inventory_cap=1)
    env = am.ArlMMEnv(p)
    env.reset(3)
    for _ in range(p.n_steps):
        env.step((0.0, 0.0))
    res = env.result()
    assert res.n_fills_ask == 1
    assert res.terminal_inventory == -1
    assert res.max_abs_inventory == 1
    assert res.n_mo >= 1


def test_env_buy_impact_pushes_mid_up() -> None:
    p = am.ArlMMEnvParams(p_buy=1.0, impact_eta=0.5, sigma=1e-9)
    env = am.ArlMMEnv(p)
    env.reset(3)
    for _ in range(p.n_steps):
        env.step((8.0, 8.0))
    res = env.result()
    assert res.n_mo >= 1
    assert res.terminal_mid > p.s0


def test_env_terminal_penalty_closed_form() -> None:
    # The penalty only enters the reward, so two otherwise-identical
    # episodes differ by exactly terminal_penalty * (q_T / cap)^2.
    p0 = am.ArlMMEnvParams(p_buy=1.0, inventory_cap=1000, terminal_penalty=0.0)
    p10 = am.ArlMMEnvParams(p_buy=1.0, inventory_cap=1000, terminal_penalty=10.0)
    acts = np.zeros((p0.n_steps, 2))
    r0 = am.run_episode(p0, acts, seed=3)
    r10 = am.run_episode(p10, acts, seed=3)
    assert r10.terminal_inventory == r0.terminal_inventory == -r0.n_mo
    expected = 10.0 * (r0.terminal_inventory / 1000.0) ** 2
    assert r0.sim_internal_return - r10.sim_internal_return == pytest.approx(expected, rel=1e-9)


def test_run_episode_action_shape_fail_closed() -> None:
    with pytest.raises(ValueError):
        am.run_episode(P, np.zeros((P.n_steps, 3)), seed=0)
    with pytest.raises(ValueError):
        am.run_episode(P, np.zeros((P.n_steps - 1, 2)), seed=0)
    with pytest.raises(TypeError):
        am.run_episode("not_params", np.zeros((P.n_steps, 2)), seed=0)


def test_run_episode_determinism() -> None:
    r1 = _flat_episode(seed=9)
    r2 = _flat_episode(seed=9)
    assert r1.sim_internal_return == r2.sim_internal_return
    np.testing.assert_array_equal(r1.sim_internal_wealth_path, r2.sim_internal_wealth_path)
    np.testing.assert_array_equal(r1.mo_times, r2.mo_times)


def test_episode_result_fields_namespace() -> None:
    res = _flat_episode(seed=1)
    assert isinstance(res.sim_internal_return, float)
    assert res.sim_internal_wealth_path.shape == (P.n_steps + 1,)
    assert res.sim_internal_wealth_path[0] == pytest.approx(0.0)
    assert 0 <= res.n_fills_bid + res.n_fills_ask <= res.n_mo
    assert res.max_abs_inventory <= P.inventory_cap
    assert np.all(res.mo_times >= 0.0)
    assert np.all(res.mo_times < P.horizon)


# ---------------------------------------------------------------------------
# Hawkes cluster statistics
# ---------------------------------------------------------------------------


def test_hawkes_cluster_stats_clustered_vs_poisson() -> None:
    env = am.ArlMMEnv(P)
    env.reset(21)
    stats = am.hawkes_cluster_stats(
        env.mo_times,
        horizon=P.horizon,
        mu=P.hawkes_mu,
        alpha=P.hawkes_alpha,
        beta=P.hawkes_beta,
    )
    assert stats["n_mo"] == env.mo_times.size
    assert stats["fano_factor"] > 1.0  # self-exciting clustering
    assert stats["intensity_max_over_mu"] > 1.0
    assert 0.0 < stats["branching_ratio_mle"] < 1.0
    p_pois = am.ArlMMEnvParams(hawkes_alpha=0.0)
    env_p = am.ArlMMEnv(p_pois)
    env_p.reset(21)
    stats_p = am.hawkes_cluster_stats(env_p.mo_times, horizon=P.horizon, fit_mle=False)
    # Hawkes self-excitation shows a Fano factor clearly above the matched
    # Poisson control (binning noise alone can push either above 1).
    assert stats["fano_factor"] > stats_p["fano_factor"]


def test_hawkes_cluster_stats_fail_closed() -> None:
    with pytest.raises(ValueError):
        am.hawkes_cluster_stats(np.asarray([]), horizon=10.0)
    with pytest.raises(ValueError):
        am.hawkes_cluster_stats(np.asarray([1.0]), horizon=10.0)
    with pytest.raises(ValueError):
        am.hawkes_cluster_stats(np.asarray([1.0, 2.0]), horizon=0.0)
    with pytest.raises(ValueError):
        am.hawkes_cluster_stats(np.asarray([1.0, 12.0]), horizon=10.0)
    with pytest.raises(ValueError):
        am.hawkes_cluster_stats(np.asarray([1.0, 2.0]), horizon=10.0, n_windows=1)
    with pytest.raises(ValueError):
        am.hawkes_cluster_stats(np.asarray([1.0, 2.0]), horizon=10.0, mu=2.0, alpha=None, beta=3.0)


# ---------------------------------------------------------------------------
# Left-tail statistics (closed form)
# ---------------------------------------------------------------------------


def test_left_tail_stats_closed_form() -> None:
    r = np.asarray([3.0, -1.0, 2.0, -4.0, 0.5, -2.0])
    out = am.left_tail_stats(r, alpha=0.2)
    q = float(np.quantile(r, 0.2))
    assert out["q_alpha"] == pytest.approx(q)
    assert out["es_alpha"] == pytest.approx(float(r[r <= q].mean()))
    assert out["mean"] == pytest.approx(float(r.mean()))
    assert out["std"] == pytest.approx(float(r.std()))
    assert out["min"] == pytest.approx(-4.0)
    assert out["n"] == pytest.approx(6.0)


def test_left_tail_stats_single_tail_element() -> None:
    r = np.asarray([1.0, 2.0, 3.0, 4.0])
    out = am.left_tail_stats(r, alpha=0.05)
    assert out["q_alpha"] == pytest.approx(float(np.quantile(r, 0.05)))
    assert out["es_alpha"] <= out["q_alpha"] + 1e-12


def test_left_tail_stats_fail_closed() -> None:
    with pytest.raises(ValueError):
        am.left_tail_stats(np.asarray([]))
    with pytest.raises(ValueError):
        am.left_tail_stats(np.asarray([1.0, float("nan")]))
    with pytest.raises(ValueError):
        am.left_tail_stats(np.asarray([1.0, float("inf")]))
    for a in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            am.left_tail_stats(np.asarray([1.0, 2.0]), alpha=a)


# ---------------------------------------------------------------------------
# Adversarial environment grid
# ---------------------------------------------------------------------------


def test_adversary_env_grid_structure() -> None:
    grid = am.adversary_env_grid(P, B, n_interior=3, seed=5)
    names = [n for n, _ in grid]
    assert names[0] == "benign"
    assert names[1:5] == ["adv_mu_max", "adv_alpha_max", "adv_sigma_max", "adv_eta_max"]
    assert names[5] == "adv_all_max"
    assert names[6:9] == ["adv_interior_0", "adv_interior_1", "adv_interior_2"]
    assert len(grid) == 9
    for _, cell in grid:
        assert isinstance(cell, am.ArlMMEnvParams)
        assert cell.hawkes_alpha < 1.0
    w = dict(grid)["adv_eta_max"]
    assert w.impact_eta == pytest.approx(P.impact_eta * 20.0)


def test_adversary_env_grid_determinism_and_extras() -> None:
    g1 = am.adversary_env_grid(P, B, n_interior=2, seed=5)
    g2 = am.adversary_env_grid(P, B, n_interior=2, seed=5)
    assert g1 == g2
    act = np.asarray([0.9, -0.9, -0.5, 0.9])
    g3 = am.adversary_env_grid(P, B, n_interior=0, seed=5, extra_actions=[act])
    assert g3[-1][0] == "adv_learned_0"
    assert len(g3) == 7


def test_adversary_env_grid_fail_closed() -> None:
    with pytest.raises(ValueError):
        am.adversary_env_grid(P, B, n_interior=-1)


# ---------------------------------------------------------------------------
# Inventory bias check (bootstrap)
# ---------------------------------------------------------------------------


def test_inventory_bias_check_equal() -> None:
    q = np.asarray([-1.0, 0.5, -0.5, 1.0, 0.0, -0.25])
    out = am.inventory_bias_check(q, q.copy(), n_boot=50, seed=0)
    assert out["bias_gap"] == pytest.approx(0.0)
    assert out["mean_abs_q_a"] == pytest.approx(out["mean_abs_q_b"])
    assert 0.0 <= out["p_more_bias"] <= 1.0
    assert out["n_pairs"] == pytest.approx(6.0)


def test_inventory_bias_check_direction() -> None:
    qa = np.full(24, 2.0)  # strongly long-biased terminals
    qb = np.zeros(24)
    out = am.inventory_bias_check(qa, qb, n_boot=50, seed=0)
    assert out["bias_gap"] == pytest.approx(2.0)
    assert out["p_more_bias"] == pytest.approx(1.0)


def test_inventory_bias_check_determinism() -> None:
    qa = np.asarray([-1.0, 0.5, -0.5, 1.0, 0.0, -0.25])
    qb = np.asarray([0.5, -0.5, 0.25, -1.0, 1.0, 0.0])
    a = am.inventory_bias_check(qa, qb, n_boot=40, seed=3)
    b = am.inventory_bias_check(qa, qb, n_boot=40, seed=3)
    assert a == b


def test_inventory_bias_check_fail_closed() -> None:
    with pytest.raises(ValueError):
        am.inventory_bias_check(np.asarray([]), np.asarray([]))
    with pytest.raises(ValueError):
        am.inventory_bias_check(np.ones(3), np.ones(4))
    with pytest.raises(ValueError):
        am.inventory_bias_check(np.asarray([1.0, float("nan")]), np.ones(2))
    with pytest.raises(ValueError):
        am.inventory_bias_check(np.ones(3), np.ones(3), n_boot=2)


# ---------------------------------------------------------------------------
# Left-tail evaluation on fixed (non-torch) policies
# ---------------------------------------------------------------------------


def test_evaluate_left_tail_fixed_policies() -> None:
    grid = [("benign", P), ("worst", am.apply_perturbation(P, B, np.ones(4)))]
    acts = np.full((P.n_steps, 2), 2.0)
    ev = am.evaluate_left_tail({"uniform": None, "fixed": acts}, grid, n_episodes=4, seed=7)
    for pol in ("uniform", "fixed"):
        for cell in ("benign", "worst"):
            st = ev.stats[pol][cell]
            assert st["n"] == pytest.approx(4.0)
            assert math.isfinite(st["q_alpha"])
            assert ev.returns[pol][cell].shape == (4,)
            assert ev.terminal_inventory[pol][cell].shape == (4,)


def test_evaluate_left_tail_determinism() -> None:
    grid = [("benign", P)]
    ev1 = am.evaluate_left_tail({"u": None}, grid, n_episodes=3, seed=7)
    ev2 = am.evaluate_left_tail({"u": None}, grid, n_episodes=3, seed=7)
    np.testing.assert_array_equal(ev1.returns["u"]["benign"], ev2.returns["u"]["benign"])


def test_evaluate_left_tail_fail_closed() -> None:
    grid = [("benign", P)]
    with pytest.raises(ValueError):
        am.evaluate_left_tail({}, grid, n_episodes=1)
    with pytest.raises(ValueError):
        am.evaluate_left_tail({"u": None}, [], n_episodes=1)
    with pytest.raises(ValueError):
        am.evaluate_left_tail({"u": None}, grid, n_episodes=0)
    mixed = [("a", P), ("b", am.ArlMMEnvParams(n_steps=10))]
    with pytest.raises(ValueError):
        am.evaluate_left_tail({"u": None}, mixed, n_episodes=1)


def test_none_policy_is_mid_grid_quotes() -> None:
    grid = [("benign", P)]
    ev = am.evaluate_left_tail({"u": None}, grid, n_episodes=2, seed=7)
    acts = np.full((P.n_steps, 2), P.max_quote_ticks / 2.0)
    ref = am.run_episode(P, acts, seed=7 + 101)
    assert ev.returns["u"]["benign"][0] == pytest.approx(ref.sim_internal_return)


# ---------------------------------------------------------------------------
# Seed validation
# ---------------------------------------------------------------------------


def test_seed_fail_closed() -> None:
    env = am.ArlMMEnv(P)
    with pytest.raises(ValueError):
        env.reset(-1)
    with pytest.raises(ValueError):
        env.reset(1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        am.run_episode(P, np.zeros((P.n_steps, 2)), seed=-1)


def test_quote_table_shape() -> None:
    assert am.MAKER_QUOTE_TABLE.shape == (am.MAKER_ACT_DIM, 2)
    assert np.all(am.MAKER_QUOTE_TABLE >= 0.0)
    spreads = am.MAKER_QUOTE_TABLE[:, 0]
    assert np.all(np.diff(spreads) > 0.0)  # ascending aggressiveness


# ---------------------------------------------------------------------------
# Torch lane (skip cleanly without the nn extra)
# ---------------------------------------------------------------------------


@requires_torch
def test_build_actor_forward_shapes() -> None:
    torch = am._torch()
    torch.manual_seed(0)
    actor = am._build_lstm_actor(torch, am.MAKER_OBS_DIM, 8, am.MAKER_ACT_DIM)
    hx = actor.init_h(4)
    x = torch.zeros(4, am.MAKER_OBS_DIM)
    logits, hx2 = actor(x, hx)
    assert tuple(logits.shape) == (4, am.MAKER_ACT_DIM)
    assert hx2[0].shape == hx[0].shape


def test_skew_to_quotes_closed_form() -> None:
    # q = 0 -> symmetric (s, s)
    out = am._skew_to_quotes(np.asarray([2.0, 1.0]), 0, 5, 8.0)
    np.testing.assert_allclose(out, [2.0, 2.0])
    # q = +cap -> bid widened, ask tightened
    out = am._skew_to_quotes(np.asarray([2.0, 1.0]), 5, 5, 8.0)
    np.testing.assert_allclose(out, [3.0, 1.0])
    # q = -cap -> bid tightened, ask widened
    out = am._skew_to_quotes(np.asarray([2.0, 1.0]), -5, 5, 8.0)
    np.testing.assert_allclose(out, [1.0, 3.0])
    # clips at zero (never cross the mid) and at max_off
    out = am._skew_to_quotes(np.asarray([2.0, 4.0]), 5, 5, 8.0)
    np.testing.assert_allclose(out, [6.0, 0.0])


@requires_torch
def test_rollout_maker_batch_finite() -> None:
    torch = am._torch()
    torch.manual_seed(0)
    torch.set_num_threads(1)
    actor = am._build_lstm_actor(torch, am.MAKER_OBS_DIM, 8, am.MAKER_ACT_DIM)
    rets, logp, invs = am._rollout_maker_batch(
        torch, actor, [P] * 4, [1, 2, 3, 4], explore=True, explore_std=0.5
    )
    assert rets.shape == (4,)
    assert np.all(np.isfinite(rets))
    assert tuple(logp.shape) == (4,)
    assert np.all(np.isfinite(logp.detach().numpy()))
    assert len(invs) == 4


@requires_torch
def test_rollout_adversary_batch() -> None:
    torch = am._torch()
    torch.manual_seed(0)
    torch.set_num_threads(1)
    adv = am._build_lstm_actor(torch, am.ADV_OBS_DIM, 8, am.ADV_ACT_DIM)
    mk = am._build_lstm_actor(torch, am.MAKER_OBS_DIM, 8, am.MAKER_ACT_DIM)
    rets, logp, perturbed, acts = am._rollout_adversary_batch(
        torch,
        adv,
        mk,
        P,
        B,
        [1, 2, 3],
        explore=True,
        explore_std=0.5,
        maker_explore_std=0.0,
    )
    assert rets.shape == (3,)
    assert len(perturbed) == 3
    assert acts.shape == (3, 4)
    assert np.all(np.abs(acts) <= 1.0 + 1e-6)
    for p in perturbed:
        assert isinstance(p, am.ArlMMEnvParams)


@requires_torch
def test_adversary_greedy_action_bounds() -> None:
    torch = am._torch()
    torch.manual_seed(0)
    adv = am._build_lstm_actor(torch, am.ADV_OBS_DIM, 8, am.ADV_ACT_DIM)
    a = am._adversary_greedy_action(torch, adv, P, B)
    assert a.shape == (4,)
    assert np.all(np.abs(a) <= 1.0 + 1e-6)


@requires_torch
def test_torch_episode_deterministic() -> None:
    torch = am._torch()
    torch.manual_seed(0)
    actor = am._build_lstm_actor(torch, am.MAKER_OBS_DIM, 8, am.MAKER_ACT_DIM)
    r1 = am._torch_episode(actor, P, 5)
    r2 = am._torch_episode(actor, P, 5)
    assert r1.sim_internal_return == r2.sim_internal_return


@requires_torch
def test_train_baseline_maker_tiny() -> None:
    out = am.train_baseline_maker(P, n_updates=2, batch_size=4, hidden=8, seed=0)
    assert out.n_maker_updates == 2
    assert out.n_adv_updates == 0
    assert out.adversary is None
    assert out.maker_loss_curve.shape == (2,)
    assert out.maker_return_curve.shape == (2,)
    assert out.n_episodes == 8


@requires_torch
def test_train_arl_pair_tiny() -> None:
    out = am.train_arl_pair(
        P,
        B,
        n_rounds=1,
        adv_epochs=1,
        maker_epochs=1,
        batch_size=6,
        hidden=8,
        seed=0,
    )
    assert out.adversary is not None
    assert out.n_adv_updates == 1
    assert out.n_maker_updates == 2  # adversarial + benign updates per epoch
    assert out.maker_loss_curve.size >= 1
    assert out.adv_loss_curve is not None and out.adv_loss_curve.size >= 1
    assert np.all(np.isfinite(out.maker_return_curve))


@requires_torch
def test_train_arl_pair_determinism() -> None:
    kw = dict(
        n_rounds=1,
        adv_epochs=1,
        maker_epochs=1,
        batch_size=6,
        hidden=8,
        seed=0,
    )
    a = am.train_arl_pair(P, B, **kw)
    b = am.train_arl_pair(P, B, **kw)
    np.testing.assert_array_equal(a.maker_loss_curve, b.maker_loss_curve)
    np.testing.assert_array_equal(a.adv_loss_curve, b.adv_loss_curve)
    for pa, pb in zip(a.maker.parameters(), b.maker.parameters(), strict=True):
        np.testing.assert_array_equal(pa.detach().numpy(), pb.detach().numpy())


@requires_torch
def test_reinforce_step_updates_weights() -> None:
    torch = am._torch()
    torch.manual_seed(0)
    actor = am._build_lstm_actor(torch, am.MAKER_OBS_DIM, 8, am.MAKER_ACT_DIM)
    opt = torch.optim.Adam(actor.parameters(), lr=0.05)
    rets, logp, _ = am._rollout_maker_batch(
        torch, actor, [P] * 6, [1, 2, 3, 4, 5, 6], explore=True, explore_std=0.5
    )
    before = [t.detach().clone() for t in actor.parameters()]
    am._reinforce_step(torch, opt, logp, (rets - rets.mean()) / (rets.std() + 1e-6))
    after = list(actor.parameters())
    assert any(
        not np.array_equal(b.numpy(), a.detach().numpy())
        for b, a in zip(before, after, strict=True)
    )


@requires_torch
def test_train_fail_closed() -> None:
    with pytest.raises(ValueError):
        am.train_arl_pair(P, B, n_rounds=0)
    with pytest.raises(ValueError):
        am.train_arl_pair(P, B, batch_size=1)
    with pytest.raises(ValueError):
        am.train_arl_pair(P, B, lr=0.0)
    with pytest.raises(ValueError):
        am.train_arl_pair(P, B, adv_lr=0.0)
    with pytest.raises(ValueError):
        am.train_arl_pair(P, B, benign_frac=1.5)
    with pytest.raises(ValueError):
        am.train_baseline_maker(P, n_updates=0)
    with pytest.raises(TypeError):
        am.train_arl_pair("not_params", B)


# ---------------------------------------------------------------------------
# Bench (skip-gated on torch)
# ---------------------------------------------------------------------------


@requires_torch
def test_arl_mm_bench_tiny_blob() -> None:
    out = am.arl_mm_bench(
        n_rounds=1,
        adv_epochs=1,
        maker_epochs=1,
        batch_size=6,
        n_eval_episodes=3,
        n_interior=0,
        n_boot=40,
        seed=2,
    )
    assert out["label"] == "SYNTHETIC"
    assert out["research_only"] == 1.0
    assert out["live_pnl_claim"] == 0.0
    assert out["claim"] == "simulator_internal_diagnostic_only"
    assert out["kind"] == "arl_mm_bench"
    assert out["data_source"] == am.ARL_MM_REVISION
    for k, v in out.items():
        if isinstance(v, float):
            assert math.isfinite(v), k
    assert 0.0 <= out["synthetic_inventory_bias_p_more"] <= 1.0
    assert out["synthetic_hawkes_branching_true"] == pytest.approx(P.hawkes_alpha)


@requires_torch
def test_arl_mm_bench_determinism() -> None:
    kw = dict(
        n_rounds=1,
        adv_epochs=1,
        maker_epochs=1,
        batch_size=6,
        n_eval_episodes=3,
        n_interior=0,
        n_boot=40,
        seed=2,
    )
    a = am.arl_mm_bench(**kw)
    b = am.arl_mm_bench(**kw)
    assert a == b


@requires_torch
def test_arl_mm_bench_no_forbidden_metric_keys() -> None:
    out = am.arl_mm_bench(
        n_rounds=1,
        adv_epochs=1,
        maker_epochs=1,
        batch_size=6,
        n_eval_episodes=3,
        n_interior=0,
        n_boot=40,
        seed=2,
    )
    banned = ("sharpe", "sortino", "calmar", "pnl", "nav", "p&l")
    for k in out:
        if k == "live_pnl_claim":
            continue  # the explicit anti-claim honesty stamp, not a metric
        assert not any(t in k.lower() for t in banned), k
        assert "sim_internal" not in k


@requires_torch
@pytest.mark.slow
def test_arl_mm_bench_headline_diagnostics() -> None:
    """Full-seeded bench: the paper's comparative mechanism holds.

    Asserts the ARL maker beats the benign-trained baseline on the LEFT
    TAIL over the adversarial grid (pooled and on the learned worst cell),
    the trained adversary measurably worsens the baseline vs benign, the
    terminal directional-inventory bias does not grow, and the Hawkes
    cluster diagnostics show self-excitation.
    """
    out = am.arl_mm_bench()
    assert out["synthetic_arl_left_tail_q05_improvement"] > 0.0
    assert out["synthetic_arl_left_tail_es05_improvement"] > 0.0
    assert out["synthetic_arl_left_tail_q05_learned_improvement"] > 0.0
    assert out["synthetic_arl_left_tail_env_improvement_frac"] >= 0.5
    assert out["synthetic_adversary_effectiveness_baseline"] > 0.0
    assert out["synthetic_hawkes_fano_factor"] > 1.0
    assert out["synthetic_hawkes_fano_factor_poisson"] < out["synthetic_hawkes_fano_factor"]
    assert 0.0 < out["synthetic_hawkes_branching_mle"] < 1.0
    assert abs(out["synthetic_inventory_bias_gap"]) < 0.5
    assert out["synthetic_terminal_abs_inv_arl"] <= (
        out["synthetic_terminal_abs_inv_baseline"] + 0.25
    )
    for k in ("mu", "alpha", "sigma", "eta"):
        assert abs(out[f"synthetic_adversary_greedy_{k}"]) <= 1.0 + 1e-6
