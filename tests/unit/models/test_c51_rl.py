"""Tests for models/c51_rl.py — C51 core + ZI-LOB step API (backlog B4-ii).

Everything here is **labeled SYNTHETIC** correctness validation of a
categorical distributional RL layer over the lane B4-i zero-intelligence LOB
(``microstructure.zi_lob_simulator``), following Bellemare, Dabney, Munos
(2017), arXiv:1707.06887 (fixed atom support, Algorithm 1 categorical
projection, cross-entropy loss), van Hasselt, Guez, Silver (2016),
arXiv:1509.06461 (double-Q target action selection) and Mnih et al. (2015)
(periodic hard target sync). No market data is touched anywhere; there is no
live-trading claim.

Three groups:

- **numpy core, torch-free**: the Algorithm 1 projection is checked against
  hand-computed atoms on tiny grids (integer shift, interior mass split,
  edge clipping, terminal discount), plus mass/mean invariants, batch
  vectorization, and the fail-closed edges. The environment's step API is
  checked for its accounting identity (reward = MTM change minus the
  inventory penalty), the hard inventory cap, the never-marketable /
  never-crossed quote guarantee, FIFO-preserving smart quoting, and seeded
  repeatability.
- **torch-gated agent**: the torch projection inside :meth:`C51Agent.learn` is
  asserted equal to the numpy reference (that reference is the reason the
  projection is exposed separately), distributions are normalized, the target
  net hard-syncs on schedule, and the risk-quantile read-out is a lower tail.
- **one short seeded training run**: in a controlled *stationary* synthetic
  book (symmetric ZI flow) the trained greedy agent must beat a random-quote
  baseline on the ``sim_internal_*`` reward by :data:`DOCUMENTED_MARGIN`, must
  reach the best fixed action cell, and must reduce its cross-entropy loss by
  :data:`DOCUMENTED_LOSS_DROP`. The action economics it is graded against are
  measured **in process** (six fixed-cell episodes + the random baseline) so
  the assertions are relative to the landscape this machine produced and stay
  portable across BLAS/torch builds. It is NOT asserted to beat the
  Avellaneda-Stoikov / GLFT closed forms — at a unit-test budget the paper's
  "RLMM beats GLFT" result is out of reach, and claiming it here would be
  overclaiming (the full-budget comparison lives in the lane module's
  ``rl_mm_benchmark``, which is ``slow``-marked and documented-optional).

Honesty: every money-like key asserted here is namespaced ``sim_internal_*``;
the forbidden-headline-token check allows ``pnl`` only under that namespace
(same convention as ``microstructure/test_rl_market_maker.py``). Torch tests
skip cleanly when the ``nn`` extra is absent.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from quant_fund.microstructure.zi_lob_simulator import santa_fe_config
from quant_fund.models import c51_rl
from quant_fund.models.c51_rl import (
    C51_RL_REVISION,
    DEFAULT_QUOTE_ACTIONS,
    N_OBS_FEATURES,
    C51Agent,
    C51AgentConfig,
    EnvStep,
    ZILobQuoteEnv,
    ZILobQuoteEnvConfig,
    agent_policy,
    atom_support,
    categorical_projection,
    classic_action_policy,
    compare_c51_baselines,
    distribution_mean,
    distribution_quantile,
    nearest_quote_action,
    random_action_policy,
    run_env_episode,
    train_c51_on_zi_lob,
    validate_quote_actions,
)


def _torch_present() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # blocked or halted torch imports
        return False


_HAS_TORCH = _torch_present()
requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="C51 agent training requires the nn extra (torch)"
)

# Forbidden *headline* metric tokens (mirrors research.catalog registry).
# "pnl" is permitted ONLY under the simulator-internal diagnostic namespace.
FORBIDDEN_HEADLINE_TOKENS = ("sharpe", "sortino", "calmar", "nav")

# ---------------------------------------------------------------------------
# Pinned SYNTHETIC scenario (stationary directionally biased flow)
# ---------------------------------------------------------------------------

CAP = 6
PENALTY = 0.05
P_BUY = 0.5
HORIZON = 120.0
TRAIN_DECISIONS = 2500
TRAIN_SEED = 5
REWARD_WINDOW = 500
EVAL_SEEDS = tuple(901 + i for i in range(8))
EVAL_DECISIONS = 120

#: Rewards are reported in tick units and normalized by the inventory cap, so
#: the discounted return stays inside the canonical C51 support. Measured on
#: the pinned scenario: per-decision |reward| has p99 ~ 0.45 tick-units, so at
#: gamma = 0.90 the return p99 is ~ 4.5 against a [-10, 10] support — no
#: clipping. (An earlier [-3, 3] support clipped almost the whole return
#: distribution at gamma = 0.95 and the agent learned noise; the measured
#: return scale, not taste, sets these numbers.)
REWARD_SCALE = 1.0 / CAP
V_MIN, V_MAX = -10.0, 10.0
GAMMA = 0.90

#: Documented learning-signal margin, in ``sim_internal_reward_mean`` units
#: (cap-normalized tick-units per decision). Measured over four training seeds
#: on the pinned scenario, trained-greedy beat the random-quote baseline by
#: +0.0085 / +0.0129 / +0.0073 / +0.0038; the pinned seed measures +0.0085 and
#: the assertion asks for +0.003 (~2.8x headroom, and still a third of the
#: total achievable spread: best fixed cell +0.0071, worst -0.0032, random
#: -0.0020). The margin is checked against an in-process landscape measurement
#: rather than a hard-coded reward level, so it stays portable across BLAS and
#: torch builds.
DOCUMENTED_MARGIN = 0.003

#: Tolerance for "reached the best fixed action cell": the pinned run lands
#: -0.0006 below it (seed 11 lands +0.0038 above), so a state-dependent policy
#: is expected to be statistically indistinguishable from the best *open-loop*
#: cell, not necessarily better.
BEST_CELL_TOLERANCE = 0.002

#: Relative cross-entropy reduction required over training (measured 0.63-0.72
#: across four seeds; the assertion asks for 0.25).
DOCUMENTED_LOSS_DROP = 0.25


def _env_config(**over: object) -> ZILobQuoteEnvConfig:
    kwargs: dict[str, object] = {
        "lob": santa_fe_config(seed=3, p_buy=P_BUY),
        "horizon": HORIZON,
        "decision_interval": 1.0,
        "inventory_cap": CAP,
        "inventory_penalty": PENALTY,
        "reward_scale": REWARD_SCALE,
    }
    kwargs.update(over)
    return ZILobQuoteEnvConfig(**kwargs)  # type: ignore[arg-type]


def _agent_config(seed: int = TRAIN_SEED) -> C51AgentConfig:
    return C51AgentConfig(
        n_atoms=51,
        v_min=V_MIN,
        v_max=V_MAX,
        hidden=(64,),
        gamma=GAMMA,
        lr=1e-3,
        batch_size=64,
        buffer_capacity=8_000,
        target_update_every=50,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=1_200,
        seed=seed,
    )


def _fixed_action_policy(index: int) -> Callable[[np.ndarray, ZILobQuoteEnv], int]:
    def policy(obs: np.ndarray, env: ZILobQuoteEnv) -> int:
        del obs, env
        return index

    return policy


def _random_rows(cfg: ZILobQuoteEnvConfig, seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    env = ZILobQuoteEnv(cfg)
    return [
        run_env_episode(
            ZILobQuoteEnv(cfg),
            random_action_policy(seed=17 + s, n_actions=env.n_actions),
            seed=s,
            max_decisions=EVAL_DECISIONS,
        )
        for s in seeds
    ]


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(r[key]) for r in rows]))


def _assert_no_forbidden_headline(mapping: dict[str, object]) -> None:
    """Every forbidden token must sit under the ``sim_internal_*`` namespace."""
    for key in mapping:
        low = str(key).lower()
        tokens = low.replace("-", "_").split("_")
        for tok in tokens:
            if tok in FORBIDDEN_HEADLINE_TOKENS:
                raise AssertionError(f"forbidden headline token {tok!r} in key {key!r}")
        if "pnl" in tokens and not low.startswith("sim_internal_"):
            raise AssertionError(f"'pnl' key {key!r} is not sim_internal_-namespaced")


# ---------------------------------------------------------------------------
# Atom support
# ---------------------------------------------------------------------------


def test_atom_support_is_uniform_and_spans_the_paper_support() -> None:
    atoms, dz = atom_support(-10.0, 10.0, 51)
    assert atoms.shape == (51,)
    assert float(atoms[0]) == pytest.approx(-10.0)
    assert float(atoms[-1]) == pytest.approx(10.0)
    assert dz == pytest.approx(0.4)
    assert np.allclose(np.diff(atoms), dz)


def test_atom_support_fail_closed() -> None:
    with pytest.raises(ValueError, match="n_atoms"):
        atom_support(-1.0, 1.0, 1)
    with pytest.raises(ValueError, match="v_min < v_max"):
        atom_support(1.0, 1.0, 51)
    with pytest.raises(ValueError, match="v_min < v_max"):
        atom_support(2.0, -2.0, 51)
    with pytest.raises(ValueError, match="finite"):
        atom_support(float("-inf"), 1.0, 51)
    with pytest.raises(ValueError, match="n_atoms"):
        atom_support(-1.0, 1.0, True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Algorithm 1 categorical projection: hand-computed atoms
# ---------------------------------------------------------------------------


def test_projection_hand_computed_integer_shift_moves_whole_atoms() -> None:
    # support {0, 1, 2}, dz = 1; target mass 0.5 on z=0 and 0.5 on z=1;
    # r = 1, gamma = 1 -> Tz = {1, 2}: both land exactly on atoms, so the
    # projected distribution is 0.5 on z=1 and 0.5 on z=2.
    atoms, _ = atom_support(0.0, 2.0, 3)
    out = categorical_projection(np.array([[0.5, 0.5, 0.0]]), 1.0, 1.0, atoms=atoms)
    assert out.shape == (1, 3)
    assert np.allclose(out, [[0.0, 0.5, 0.5]])
    assert float(out.sum()) == pytest.approx(1.0)


def test_projection_hand_computed_interior_split() -> None:
    # support {0, 0.5, 1}, dz = 0.5; all mass on z=0; r = 0.25, gamma = 1
    # -> Tz = 0.25 -> b = 0.5, exactly halfway between atoms 0 and 1, so the
    # mass splits 50/50 and the mean is preserved at 0.25.
    atoms, _ = atom_support(0.0, 1.0, 3)
    out = categorical_projection(np.array([[1.0, 0.0, 0.0]]), 0.25, 1.0, atoms=atoms)
    assert np.allclose(out, [[0.5, 0.5, 0.0]])
    assert float((out @ atoms)[0]) == pytest.approx(0.25)


def test_projection_hand_computed_discounted_split() -> None:
    # support {0, 0.5, 1}; mass on z=1; r = 0, gamma = 0.5 -> Tz = 0.5, which
    # is exactly atom 1 -> no split, all mass on the middle atom.
    atoms, _ = atom_support(0.0, 1.0, 3)
    out = categorical_projection(np.array([[0.0, 0.0, 1.0]]), 0.0, 0.5, atoms=atoms)
    assert np.allclose(out, [[0.0, 1.0, 0.0]])


def test_projection_clips_at_the_support_edge_and_preserves_mass() -> None:
    # Tz = 0 + 1 * 1.75 = 1.75 > v_max = 1 -> clipped onto the top atom.
    atoms, _ = atom_support(0.0, 1.0, 3)
    out = categorical_projection(np.array([[0.0, 0.0, 1.0]]), 0.75, 1.0, atoms=atoms)
    assert np.allclose(out, [[0.0, 0.0, 1.0]])
    low = categorical_projection(np.array([[1.0, 0.0, 0.0]]), -0.75, 1.0, atoms=atoms)
    assert np.allclose(low, [[1.0, 0.0, 0.0]])
    assert float(out.sum()) == pytest.approx(1.0)
    assert float(low.sum()) == pytest.approx(1.0)


def test_projection_mean_equals_bellman_mean_when_unclipped() -> None:
    rng = np.random.default_rng(7)
    atoms, _ = atom_support(-2.0, 2.0, 9)
    logits = rng.normal(size=(6, atoms.size))
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    rewards = np.array([0.1, -0.2, 0.0, 0.05, -0.05, 0.3])
    gammas = np.full(6, 0.5)
    out = categorical_projection(probs, rewards, gammas, atoms=atoms)
    expected = np.clip(rewards + gammas * (probs @ atoms), atoms[0], atoms[-1])
    assert np.allclose(out @ atoms, expected)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.all(out >= 0.0)


def test_projection_terminal_gamma_collapses_onto_the_reward() -> None:
    atoms, _ = atom_support(-3.0, 3.0, 7)
    probs = np.array([[0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1]])
    out = categorical_projection(probs, 1.0, 0.0, atoms=atoms)
    # gamma = 0 kills the bootstrap: every atom maps onto r = 1 = atom index 4.
    assert np.allclose(out, [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])


def test_projection_is_vectorized_row_by_row() -> None:
    rng = np.random.default_rng(11)
    atoms, _ = atom_support(-1.0, 1.0, 5)
    probs = np.full((3, atoms.size), 1.0 / atoms.size)
    probs[1] = rng.dirichlet(np.ones(atoms.size))
    rewards = np.array([-0.3, 0.0, 0.4])
    gammas = np.array([0.9, 0.9, 0.4])
    batched = categorical_projection(probs, rewards, gammas, atoms=atoms)
    for i in range(3):
        single = categorical_projection(
            probs[i][None, :], float(rewards[i]), float(gammas[i]), atoms=atoms
        )
        assert np.allclose(batched[i], single[0])


def test_projection_fail_closed() -> None:
    atoms, _ = atom_support(0.0, 1.0, 3)
    good = np.array([[0.5, 0.5, 0.0]])
    with pytest.raises(ValueError, match="non-negative"):
        categorical_projection(np.array([[1.5, -0.5, 0.0]]), 0.0, 1.0, atoms=atoms)
    with pytest.raises(ValueError, match="sum to 1"):
        categorical_projection(np.array([[0.5, 0.2, 0.0]]), 0.0, 1.0, atoms=atoms)
    with pytest.raises(ValueError, match=r"\(batch, 3\)"):
        categorical_projection(np.array([[0.5, 0.5]]), 0.0, 1.0, atoms=atoms)
    with pytest.raises(ValueError, match="uniformly spaced"):
        categorical_projection(good, 0.0, 1.0, atoms=np.array([0.0, 0.1, 0.9]))
    with pytest.raises(ValueError, match="strictly increasing"):
        categorical_projection(good, 0.0, 1.0, atoms=np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="at least two atoms"):
        categorical_projection(np.array([[1.0]]), 0.0, 1.0, atoms=np.array([0.5]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        categorical_projection(good, 0.0, 1.5, atoms=atoms)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        categorical_projection(good, 0.0, -0.1, atoms=atoms)
    with pytest.raises(ValueError, match="finite"):
        categorical_projection(good, float("nan"), 1.0, atoms=atoms)
    with pytest.raises(ValueError, match="finite"):
        categorical_projection(np.array([[0.5, float("nan"), 0.0]]), 0.0, 1.0, atoms=atoms)


def test_distribution_mean_and_quantile_readouts() -> None:
    atoms, _ = atom_support(0.0, 4.0, 5)  # {0, 1, 2, 3, 4}
    probs = np.array([[0.0, 0.0, 1.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0, 0.5]])
    assert np.allclose(distribution_mean(probs, atoms), [2.0, 2.0])
    # degenerate distribution: every quantile is the single atom
    assert np.allclose(distribution_quantile(probs[:1], atoms, 0.01), [2.0])
    assert np.allclose(distribution_quantile(probs[:1], atoms, 1.0), [2.0])
    # two-point distribution: the lower tail sits on the low atom up to tau=0.5
    assert np.allclose(distribution_quantile(probs[1:], atoms, 0.25), [0.0])
    assert np.allclose(distribution_quantile(probs[1:], atoms, 0.5), [0.0])
    assert np.allclose(distribution_quantile(probs[1:], atoms, 0.75), [4.0])
    assert float(distribution_quantile(probs[1:], atoms, 1.0)[0]) == pytest.approx(4.0)
    assert distribution_quantile(probs, atoms, 0.5).shape == (2,)


def test_distribution_quantile_fail_closed() -> None:
    atoms, _ = atom_support(0.0, 1.0, 3)
    probs = np.array([[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="tau"):
        distribution_quantile(probs, atoms, 0.0)
    with pytest.raises(ValueError, match="tau"):
        distribution_quantile(probs, atoms, 1.5)
    with pytest.raises(ValueError, match="tau"):
        distribution_quantile(probs, atoms, float("nan"))


def test_validate_quote_actions_accepts_the_default_grid() -> None:
    cells = validate_quote_actions(DEFAULT_QUOTE_ACTIONS)
    assert cells == DEFAULT_QUOTE_ACTIONS
    assert len(cells) == 6
    # every cell stays inside the engine's [-1, 64] offset domain
    assert all(-1 <= v <= 64 for cell in cells for v in cell)


def test_validate_quote_actions_fail_closed() -> None:
    with pytest.raises(ValueError, match=">= 2 cells"):
        validate_quote_actions(((0, 0),))
    with pytest.raises(ValueError, match="duplicate"):
        validate_quote_actions(((0, 0), (0, 0)))
    with pytest.raises(ValueError, match=r"\[-1, 64\]"):
        validate_quote_actions(((0, 0), (-2, 0)))
    with pytest.raises(ValueError, match=r"\[-1, 64\]"):
        validate_quote_actions(((0, 0), (0, 65)))
    with pytest.raises(ValueError, match="must be ints"):
        validate_quote_actions(((0, 0), (0, True)))
    with pytest.raises(ValueError, match="pairs"):
        validate_quote_actions(((0, 0), (0, 0, 0)))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Environment step API (torch-free)
# ---------------------------------------------------------------------------


def test_env_reset_returns_a_normalized_observation() -> None:
    cfg = _env_config()
    env = ZILobQuoteEnv(cfg)
    obs = env.reset(seed=901)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (N_OBS_FEATURES,)
    assert obs.dtype == np.float64
    assert bool(np.all(np.isfinite(obs)))
    assert env.obs_dim == N_OBS_FEATURES
    assert env.n_actions == len(DEFAULT_QUOTE_ACTIONS)
    # flat inventory and no elapsed time at reset
    assert float(obs[4]) == pytest.approx(0.0)
    assert float(obs[8]) == pytest.approx(0.0)
    # documented normalization clips
    assert float(obs[0]) <= 8.0
    assert float(obs[1]) <= 8.0 and float(obs[2]) <= 8.0
    assert -1.0 <= float(obs[3]) <= 1.0
    assert -5.0 <= float(obs[5]) <= 5.0
    assert 0.0 <= float(obs[6]) <= 1.0 and 0.0 <= float(obs[7]) <= 1.0


@pytest.mark.parametrize("scale", [1.0, 1.0 / 6.0])
def test_env_reward_is_mtm_change_minus_inventory_penalty(scale: float) -> None:
    """Telescoping identity: sum(reward/scale + penalty) * tick == final MTM.

    ``reward_t = reward_scale * ((MTM_t - MTM_{t-1}) / tick - penalty_t)`` with
    ``MTM = cash + inventory * mid`` and ``MTM_0 = 0``, so the discounted-free
    sum telescopes exactly onto the episode's final mark-to-market. Checked at
    ``reward_scale = 1`` (the documented form) and at the cap-normalized scale
    the training scenario uses.
    """
    cfg = _env_config(reward_scale=scale)
    env = ZILobQuoteEnv(cfg)
    env.reset(seed=901)
    total_reward = 0.0
    total_penalty = 0.0
    done = False
    k = 0
    while not done:
        out = env.step(k % env.n_actions)
        k += 1
        assert isinstance(out, EnvStep)
        assert math.isfinite(out.reward)
        assert math.isfinite(float(out.info["sim_internal_mtm"]))
        total_reward += out.reward
        total_penalty += float(out.info["sim_internal_inventory_penalty"])
        done = out.terminated
    summary = env.episode_summary()
    lhs = (total_reward / scale + total_penalty) * cfg.lob.tick
    assert lhs == pytest.approx(float(summary["sim_internal_mtm_pnl_final"]), abs=1e-9)
    assert float(summary["sim_internal_reward_sum"]) == pytest.approx(total_reward)
    assert float(summary["sim_internal_inventory_penalty_sum"]) == pytest.approx(total_penalty)
    assert math.isfinite(float(summary["sim_internal_fill_gain_ticks_sum"]))
    assert float(summary["sim_internal_mtm_pnl_final"]) == pytest.approx(env.mtm)
    # the reward scale is a pure multiplier on the same accounting
    assert float(summary["sim_internal_reward_mean"]) == pytest.approx(
        total_reward / max(1, int(summary["n_decisions"]))
    )


def test_env_inventory_cap_is_hard_for_every_action_cell() -> None:
    cfg = _env_config()
    for index in range(len(DEFAULT_QUOTE_ACTIONS)):
        env = ZILobQuoteEnv(cfg)
        obs = env.reset(seed=905)
        done = False
        while not done:
            out = env.step(index)
            obs = out.obs
            assert abs(int(out.info["inventory"])) <= CAP
            assert abs(float(obs[4])) <= 1.0 + 1e-12
            done = out.terminated
        summary = env.episode_summary()
        assert int(summary["max_abs_inventory"]) <= CAP
        # a saturating policy must have had a leg suppressed at the wall
        if int(summary["max_abs_inventory"]) >= CAP:
            assert int(summary["n_quote_suppressions"]) > 0


def test_env_quotes_never_cross_and_never_go_marketable() -> None:
    """The engine raises on marketable limits; no cell may trip it."""
    cfg = _env_config()
    for index in range(len(DEFAULT_QUOTE_ACTIONS)):
        env = ZILobQuoteEnv(cfg)
        env.reset(seed=907)
        done = False
        seen_both = 0
        while not done:
            out = env.step(index)
            bid_level = out.info["bid_level"]
            ask_level = out.info["ask_level"]
            if bid_level is not None and ask_level is not None:
                seen_both += 1
                assert int(bid_level) < int(ask_level)
                assert int(bid_level) < env.sim.best_ask_level
                assert int(ask_level) > env.sim.best_bid_level
            done = out.terminated
        assert seen_both > 0


def test_env_smart_quoting_holds_fifo_queue_priority() -> None:
    """An unchanged target level keeps the resting order (no cancel/repost)."""
    cfg = _env_config()
    env = ZILobQuoteEnv(cfg)
    run_env_episode(env, _fixed_action_policy(1), seed=909)
    summary = env.episode_summary()
    held = int(summary["n_quotes_held"])
    reposted = int(summary["n_quotes_reposted"])
    assert held > 0
    assert reposted > 0
    # holding dominates: the touch does not move on most decision intervals
    assert held > reposted


def test_env_reset_is_seeded_and_repeatable() -> None:
    cfg = _env_config()
    first = run_env_episode(ZILobQuoteEnv(cfg), _fixed_action_policy(0), seed=911)
    second = run_env_episode(ZILobQuoteEnv(cfg), _fixed_action_policy(0), seed=911)
    other = run_env_episode(ZILobQuoteEnv(cfg), _fixed_action_policy(0), seed=912)
    assert float(first["sim_internal_reward_sum"]) == pytest.approx(
        float(second["sim_internal_reward_sum"])
    )
    assert int(first["n_fills"]) == int(second["n_fills"])
    assert int(first["max_abs_inventory"]) == int(second["max_abs_inventory"])
    assert float(first["sim_internal_reward_sum"]) != pytest.approx(
        float(other["sim_internal_reward_sum"])
    )


def test_env_config_fail_closed() -> None:
    with pytest.raises(ValueError, match="horizon"):
        _env_config(horizon=0.0)
    with pytest.raises(ValueError, match=">= decision_interval"):
        _env_config(horizon=0.5, decision_interval=1.0)
    with pytest.raises(ValueError, match="decision_interval"):
        _env_config(decision_interval=0.0)
    with pytest.raises(ValueError, match="inventory_cap"):
        _env_config(inventory_cap=0)
    with pytest.raises(ValueError, match="inventory_penalty"):
        _env_config(inventory_penalty=-0.1)
    with pytest.raises(ValueError, match="reward_scale"):
        _env_config(reward_scale=0.0)
    with pytest.raises(ValueError, match="spread_scale"):
        _env_config(spread_scale=0.0)
    with pytest.raises(ValueError, match="return_scale"):
        _env_config(return_scale=-1.0)
    with pytest.raises(ValueError, match=">= 2 cells"):
        _env_config(quote_actions=((0, 0),))
    with pytest.raises(TypeError):
        ZILobQuoteEnvConfig(lob="not-a-config")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ZILobQuoteEnv(_env_config(), flow_factory="nope")  # type: ignore[arg-type]


def test_env_step_fail_closed() -> None:
    cfg = _env_config(horizon=3.0)
    env = ZILobQuoteEnv(cfg)
    env.reset(seed=913)
    with pytest.raises(ValueError, match="action must be an int"):
        env.step(env.n_actions)
    with pytest.raises(ValueError, match="action must be an int"):
        env.step(-1)
    with pytest.raises(ValueError, match="action must be an int"):
        env.step(True)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="terminated"):
        while not env.terminated:
            env.step(0)
        env.step(0)
    with pytest.raises(ValueError, match="max_decisions"):
        run_env_episode(ZILobQuoteEnv(cfg), _fixed_action_policy(0), seed=1, max_decisions=0)
    with pytest.raises(TypeError):
        run_env_episode(ZILobQuoteEnv(cfg), "not-callable", seed=1)  # type: ignore[arg-type]


def test_nearest_quote_action_maps_closed_form_offsets() -> None:
    cfg = _env_config()
    env = ZILobQuoteEnv(cfg)
    env.reset(seed=915)
    tick = env.tick
    bb = env.sim.level_to_price(env.best_bid_level)
    ba = env.sim.level_to_price(env.best_ask_level)
    cases = (
        (bb + tick, ba - tick, (-1, -1)),
        (bb, ba, (0, 0)),
        (bb - tick, ba + tick, (1, 1)),
        (bb - 2 * tick, ba + 2 * tick, (2, 2)),
        (bb + tick, ba, (-1, 0)),
        (bb, ba - tick, (0, -1)),
    )
    for bid, ask, expected in cases:
        index = nearest_quote_action(env, bid=bid, ask=ask)
        assert DEFAULT_QUOTE_ACTIONS[index] == expected
    # a stood-down leg maps to the widest cell on that side
    stood_down = nearest_quote_action(env, bid=None, ask=None)
    assert DEFAULT_QUOTE_ACTIONS[stood_down] == (2, 2)
    with pytest.raises(ValueError, match="bid"):
        nearest_quote_action(env, bid=-1.0, ask=ba)
    with pytest.raises(ValueError, match="ask"):
        nearest_quote_action(env, bid=bb, ask=float("nan"))
    with pytest.raises(TypeError):
        nearest_quote_action("nope", bid=bb, ask=ba)  # type: ignore[arg-type]


def test_classic_and_random_policies_run_torch_free() -> None:
    """AS / GLFT references and the random control all complete episodes."""
    cfg = _env_config()
    tick = float(cfg.lob.tick)
    as_policy = classic_action_policy(
        kind="as", gamma=0.01, sigma=2 * tick, kappa=2.0 / tick, tick=tick
    )
    glft_policy = classic_action_policy(
        kind="glft", gamma=0.01, sigma=2 * tick, kappa=2.0 / tick, a_fill=2.7e-4, tick=tick
    )
    env = ZILobQuoteEnv(cfg)
    rows: dict[str, dict[str, Any]] = {}
    chosen: dict[str, set[int]] = {"as": set(), "glft": set()}

    def spy(name: str, inner: Callable[[np.ndarray, ZILobQuoteEnv], int]):
        def policy(obs: np.ndarray, e: ZILobQuoteEnv) -> int:
            index = inner(obs, e)
            chosen[name].add(index)
            return index

        return policy

    rows["as"] = run_env_episode(ZILobQuoteEnv(cfg), spy("as", as_policy), seed=917)
    rows["glft"] = run_env_episode(ZILobQuoteEnv(cfg), spy("glft", glft_policy), seed=917)
    rows["random"] = run_env_episode(
        ZILobQuoteEnv(cfg),
        random_action_policy(seed=3, n_actions=env.n_actions),
        seed=917,
    )
    for row in rows.values():
        assert row["label"] == "SYNTHETIC"
        assert row["live_pnl_claim"] is False
        assert row["research_only"] is True
        assert row["data_source"] == C51_RL_REVISION
        assert int(row["max_abs_inventory"]) <= CAP
        assert math.isfinite(float(row["sim_internal_reward_mean"]))
        assert math.isfinite(float(row["sim_internal_mtm_pnl_final"]))
        _assert_no_forbidden_headline({k: v for k, v in row.items() if isinstance(v, float)})
    # the closed forms participate (they are not pinned to the widest cell)
    widest = DEFAULT_QUOTE_ACTIONS.index((2, 2))
    assert chosen["as"] != {widest}
    assert min(chosen["as"]) <= 1
    assert len(chosen["glft"]) >= 1
    with pytest.raises(ValueError, match="kind"):
        classic_action_policy(kind="dqn", gamma=0.01, sigma=0.02, kappa=100.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="a_fill"):
        classic_action_policy(kind="glft", gamma=0.01, sigma=0.02, kappa=100.0)


def test_run_env_episode_summary_carries_the_honesty_envelope() -> None:
    cfg = _env_config(horizon=20.0)
    row = run_env_episode(ZILobQuoteEnv(cfg), _fixed_action_policy(1), seed=919)
    assert row["label"] == "SYNTHETIC"
    assert row["claim"] == "simulator_internal_diagnostic_only"
    assert row["live_pnl_claim"] is False
    assert row["research_only"] is True
    assert row["data_source"] == C51_RL_REVISION
    assert row["policy_seed"] == 919
    assert row["terminated_by_horizon"] is True
    assert row["truncated"] is False
    assert int(row["n_steps_run"]) > 0
    truncated = run_env_episode(
        ZILobQuoteEnv(cfg), _fixed_action_policy(1), seed=919, max_decisions=3
    )
    assert truncated["truncated"] is True
    assert int(truncated["n_steps_run"]) == 3


# ---------------------------------------------------------------------------
# C51 agent (torch-gated)
# ---------------------------------------------------------------------------


@requires_torch
def test_agent_config_fail_closed() -> None:
    with pytest.raises(ValueError, match="n_atoms"):
        C51AgentConfig(n_atoms=1)
    with pytest.raises(ValueError, match="v_min < v_max"):
        C51AgentConfig(v_min=1.0, v_max=1.0)
    with pytest.raises(ValueError, match="v_min < v_max"):
        C51AgentConfig(v_min=3.0, v_max=-3.0)
    with pytest.raises(ValueError, match="hidden"):
        C51AgentConfig(hidden=())
    with pytest.raises(ValueError, match="hidden"):
        C51AgentConfig(hidden=(0,))
    with pytest.raises(ValueError, match="gamma"):
        C51AgentConfig(gamma=1.0)
    with pytest.raises(ValueError, match="gamma"):
        C51AgentConfig(gamma=0.0)
    with pytest.raises(ValueError, match="lr"):
        C51AgentConfig(lr=0.0)
    with pytest.raises(ValueError, match="buffer_capacity"):
        C51AgentConfig(batch_size=64, buffer_capacity=8)
    with pytest.raises(ValueError, match="target_update_every"):
        C51AgentConfig(target_update_every=0)
    with pytest.raises(ValueError, match="eps_end"):
        C51AgentConfig(eps_start=0.1, eps_end=0.5)
    with pytest.raises(ValueError, match="eps_decay_steps"):
        C51AgentConfig(eps_decay_steps=0)
    with pytest.raises(ValueError, match="risk_quantile"):
        C51AgentConfig(risk_quantile=0.0)
    with pytest.raises(ValueError, match="risk_quantile"):
        C51AgentConfig(risk_quantile=1.0)
    with pytest.raises(ValueError, match="seed"):
        C51AgentConfig(seed=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="obs_dim"):
        C51Agent(0, 4, C51AgentConfig())
    with pytest.raises(TypeError):
        C51Agent(N_OBS_FEATURES, 4, "not-a-config")  # type: ignore[arg-type]


@requires_torch
def test_torch_projection_matches_the_numpy_reference() -> None:
    """The in-agent torch projection equals :func:`categorical_projection`.

    This is the point of exposing the numpy Algorithm 1 separately: the torch
    scatter path used by :meth:`C51Agent.learn` is checked against a
    reference that the hand-computed tests above pin down.
    """
    import torch

    cfg = C51AgentConfig(n_atoms=11, v_min=-2.0, v_max=2.0, hidden=(8,))
    agent = C51Agent(N_OBS_FEATURES, len(DEFAULT_QUOTE_ACTIONS), cfg)
    rng = np.random.default_rng(23)
    n_atoms = cfg.n_atoms
    logits = rng.normal(size=(7, n_atoms))
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    rewards = rng.normal(scale=0.7, size=7)
    gammas = rng.uniform(0.0, 1.0, size=7)
    atoms, _ = atom_support(cfg.v_min, cfg.v_max, n_atoms)
    expected = categorical_projection(probs, rewards, gammas, atoms=atoms)
    with torch.no_grad():
        got = agent.project_target_distribution(
            torch.as_tensor(probs, dtype=torch.float32),
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(gammas, dtype=torch.float32),
        ).numpy()
    assert np.allclose(got, expected, atol=2e-5)
    assert np.allclose(got.sum(axis=1), 1.0, atol=1e-5)
    # a hand-checked corner: mass on the top atom with a reward past v_max
    corner = np.zeros((1, n_atoms))
    corner[0, -1] = 1.0
    with torch.no_grad():
        got_corner = agent.project_target_distribution(
            torch.as_tensor(corner, dtype=torch.float32),
            torch.as_tensor([5.0], dtype=torch.float32),
            torch.as_tensor([1.0], dtype=torch.float32),
        ).numpy()
    assert np.allclose(got_corner, categorical_projection(corner, 5.0, 1.0, atoms=atoms))
    assert float(got_corner[0, -1]) == pytest.approx(1.0)


@requires_torch
def test_agent_distributions_are_normalized_and_greedy_is_deterministic() -> None:
    cfg = _agent_config(seed=3)
    agent = C51Agent(N_OBS_FEATURES, len(DEFAULT_QUOTE_ACTIONS), cfg)
    env = ZILobQuoteEnv(_env_config())
    obs = env.reset(seed=921)
    probs = agent.action_distributions(obs)
    assert probs.shape == (len(DEFAULT_QUOTE_ACTIONS), cfg.n_atoms)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs >= 0.0)
    q = agent.q_values(obs)
    assert q.shape == (len(DEFAULT_QUOTE_ACTIONS),)
    assert np.allclose(q, distribution_mean(probs, agent.atoms))
    assert np.allclose(q, distribution_mean(probs, np.linspace(cfg.v_min, cfg.v_max, cfg.n_atoms)))
    # greedy evaluation is a pure read-out: repeatable and state-preserving
    decisions_before = agent.n_decisions
    first = [agent.act(obs, greedy=True) for _ in range(5)]
    assert len(set(first)) == 1
    assert agent.n_decisions == decisions_before
    assert np.allclose(agent.action_distributions(obs), probs)
    with pytest.raises(ValueError, match="obs must have shape"):
        agent.act(np.zeros(N_OBS_FEATURES + 1))
    with pytest.raises(ValueError, match="finite"):
        agent.act(np.full(N_OBS_FEATURES, float("nan")))
    # the target net starts as an exact copy of the online net
    assert agent.n_target_syncs == 1


@requires_torch
def test_agent_learn_gates_on_buffer_and_hard_syncs_the_target() -> None:
    cfg = C51AgentConfig(
        n_atoms=21,
        v_min=-3.0,
        v_max=3.0,
        hidden=(16,),
        gamma=0.95,
        lr=1e-3,
        batch_size=8,
        buffer_capacity=64,
        target_update_every=3,
        eps_decay_steps=10,
        seed=2,
    )
    agent = C51Agent(N_OBS_FEATURES, len(DEFAULT_QUOTE_ACTIONS), cfg)
    env = ZILobQuoteEnv(_env_config(horizon=60.0))
    obs = env.reset(seed=923)
    # below batch_size there is nothing to learn from
    for _ in range(cfg.batch_size - 1):
        action = agent.act(obs)
        out = env.step(action)
        agent.store_transition(obs, action, out.reward, out.obs, out.terminated)
        obs = out.obs
    assert agent.buffer_size == cfg.batch_size - 1
    assert agent.learn() is None
    assert agent.n_updates == 0
    syncs_before = agent.n_target_syncs
    losses: list[float] = []
    for _ in range(9):
        action = agent.act(obs)
        out = env.step(action)
        agent.store_transition(obs, action, out.reward, out.obs, out.terminated)
        obs = out.obs
        loss = agent.learn()
        assert loss is not None
        assert math.isfinite(loss)
        losses.append(loss)
    assert agent.n_updates == 9
    # 9 updates with target_update_every=3 -> 3 scheduled hard syncs
    assert agent.n_target_syncs == syncs_before + 3
    assert agent.buffer_size <= cfg.buffer_capacity
    # epsilon decays monotonically with counted non-greedy decisions
    assert agent.epsilon <= cfg.eps_start
    assert agent.n_decisions > 0
    with pytest.raises(ValueError, match="action must be an int"):
        agent.store_transition(obs, agent.n_actions, 0.0, obs, False)
    with pytest.raises(ValueError, match="reward must be finite"):
        agent.store_transition(obs, 0, float("nan"), obs, False)


@requires_torch
def test_risk_quantile_readout_is_a_lower_tail() -> None:
    base = _agent_config(seed=4)
    mean_agent = C51Agent(N_OBS_FEATURES, len(DEFAULT_QUOTE_ACTIONS), base)
    # same seed -> identical initial weights, so the two read-outs are
    # comparable on the very same network
    risk_agent = C51Agent(
        N_OBS_FEATURES,
        len(DEFAULT_QUOTE_ACTIONS),
        replace(base, risk_quantile=0.1),
    )
    top_agent = C51Agent(
        N_OBS_FEATURES,
        len(DEFAULT_QUOTE_ACTIONS),
        replace(base, risk_quantile=0.999),
    )
    env = ZILobQuoteEnv(_env_config())
    obs = env.reset(seed=925)
    q = mean_agent.action_scores(obs)
    tail = risk_agent.action_scores(obs)
    top = top_agent.action_scores(obs)
    # the mean read-out is E[Z]; the quantile read-outs stay inside the support
    assert np.allclose(q, mean_agent.q_values(obs))
    assert mean_agent.config.risk_quantile is None
    assert risk_agent.config.risk_quantile == pytest.approx(0.1)
    assert tail.shape == q.shape
    assert np.all(tail <= float(base.v_max) + 1e-9)
    assert np.all(tail >= float(base.v_min) - 1e-9)
    # a higher tau is a weakly higher quantile of the same distribution
    assert np.all(top >= tail - 1e-9)
    assert not np.allclose(top, tail)


@requires_torch
def test_seeded_training_beats_random_quoting_by_the_documented_margin() -> None:
    """One short seeded run: the learning signal, in ``sim_internal_*`` units.

    Controlled *stationary* synthetic regime (symmetric ZI flow, ``p_buy=0.5``,
    quadratic inventory penalty 0.05), 2500 training decisions, greedy
    evaluation over eight unseen episode seeds.

    The action economics of this book are measured **in process** rather than
    hard-coded (six fixed-cell episodes plus the random baseline, ~1 s), so the
    assertions are relative to the landscape this machine actually produced and
    stay portable across BLAS/torch builds. Asserted:

    (i) the trained greedy agent beats the random-quote baseline on
        ``sim_internal_reward_mean`` by :data:`DOCUMENTED_MARGIN` (measured
        +0.0085 at the pinned seed; +0.0038 to +0.0129 across four seeds);
    (ii) it is positive in absolute terms and reaches the *best fixed* action
        cell within :data:`BEST_CELL_TOLERANCE` — i.e. it learned a
        state-dependent policy, not just "avoid the worst cell";
    (iii) it stays clearly above the worst fixed cell;
    (iv) its Q-ranking agrees with the measured landscape: averaged over the
        eight episode-start states, the best cell out-ranks the worst cell;
    (v) the cross-entropy loss falls by at least
        :data:`DOCUMENTED_LOSS_DROP` (measured 0.63-0.71 across four seeds);
    (vi) the inventory cap holds in every evaluated episode.

    Not asserted: beating the Avellaneda-Stoikov / GLFT closed forms. At a
    unit-test budget the paper's "RLMM beats GLFT across the risk-return
    frontier" is out of reach and claiming it here would be overclaiming; the
    full-budget comparison lives in ``microstructure.rl_market_maker.
    rl_mm_benchmark`` (``slow``-marked, documented-optional).
    """
    cfg = _env_config()
    cell_means = [
        _mean(
            [
                run_env_episode(
                    ZILobQuoteEnv(cfg),
                    _fixed_action_policy(i),
                    seed=s,
                    max_decisions=EVAL_DECISIONS,
                )
                for s in EVAL_SEEDS
            ],
            "sim_internal_reward_mean",
        )
        for i in range(len(DEFAULT_QUOTE_ACTIONS))
    ]
    baseline = _mean(_random_rows(cfg, EVAL_SEEDS), "sim_internal_reward_mean")
    best_cell = int(np.argmax(cell_means))
    worst_cell = int(np.argmin(cell_means))
    # guard against a vacuous scenario: the book must contain an edge to learn
    assert cell_means[best_cell] > baseline
    assert cell_means[best_cell] > cell_means[worst_cell]

    result = train_c51_on_zi_lob(
        env_config=cfg,
        total_decisions=TRAIN_DECISIONS,
        agent_config=_agent_config(TRAIN_SEED),
        learn_every=1,
        reward_window=REWARD_WINDOW,
        train_seed=TRAIN_SEED,
    )
    assert result.label == "SYNTHETIC"
    assert result.n_decisions == TRAIN_DECISIONS
    assert result.episodes > 1
    assert result.loss_curve.size > 0
    assert bool(np.all(np.isfinite(result.loss_curve)))
    assert result.reward_curve.size >= 2
    assert result.agent.n_target_syncs > 1

    trained_rows = [
        run_env_episode(
            ZILobQuoteEnv(cfg),
            agent_policy(result.agent, greedy=True),
            seed=s,
            max_decisions=EVAL_DECISIONS,
        )
        for s in EVAL_SEEDS
    ]
    trained = _mean(trained_rows, "sim_internal_reward_mean")
    context = (
        f"landscape={[round(v, 5) for v in cell_means]} best={best_cell} "
        f"worst={worst_cell} random={baseline:+.5f} trained={trained:+.5f}"
    )
    # (i) documented margin over the random-quote baseline
    assert trained - baseline >= DOCUMENTED_MARGIN, f"learning signal too weak: {context}"
    # (ii) positive, and at the level of the best fixed cell (state-dependent)
    assert trained > 0.0, context
    assert trained >= cell_means[best_cell] - BEST_CELL_TOLERANCE, context
    # (iii) clearly above the worst cell
    assert trained > cell_means[worst_cell] + DOCUMENTED_MARGIN, context
    for row in trained_rows:
        assert int(row["max_abs_inventory"]) <= CAP
        assert row["live_pnl_claim"] is False
        assert row["label"] == "SYNTHETIC"
        _assert_no_forbidden_headline({k: v for k, v in row.items() if isinstance(v, float)})

    # (iv) the learned Q-ranking agrees with the measured landscape, averaged
    # over the eight episode-start states (a mean over states is far more
    # stable than a single-observation argmax under cross-platform float noise)
    q_bar = np.mean(
        [result.agent.q_values(ZILobQuoteEnv(cfg).reset(seed=s)) for s in EVAL_SEEDS],
        axis=0,
    )
    assert q_bar.shape == (len(DEFAULT_QUOTE_ACTIONS),)
    assert bool(np.all(np.isfinite(q_bar)))
    assert q_bar[best_cell] > q_bar[worst_cell], (
        f"trained Q must rank the measured best cell {DEFAULT_QUOTE_ACTIONS[best_cell]} above "
        f"the measured worst cell {DEFAULT_QUOTE_ACTIONS[worst_cell]}; "
        f"q_bar={np.round(q_bar, 5).tolist()}; {context}"
    )

    # (v) the categorical cross-entropy loss falls materially over training
    metrics = result.metrics
    first = float(metrics["sim_internal_loss_mean_first"])
    last = float(metrics["sim_internal_loss_mean_last"])
    assert first > last
    assert 1.0 - last / first >= DOCUMENTED_LOSS_DROP, f"loss drop too small: {first} -> {last}"
    _assert_no_forbidden_headline(metrics)
    assert float(metrics["n_gradient_updates"]) > 0.0
    assert float(metrics["n_target_syncs"]) > 0.0


@requires_torch
def test_training_is_bit_deterministic_for_a_seed() -> None:
    cfg = _env_config(horizon=40.0)
    kwargs = {
        "env_config": cfg,
        "total_decisions": 600,
        "agent_config": _agent_config(TRAIN_SEED),
        "reward_window": 200,
        "train_seed": TRAIN_SEED,
    }
    first = train_c51_on_zi_lob(**kwargs)  # type: ignore[arg-type]
    second = train_c51_on_zi_lob(**kwargs)  # type: ignore[arg-type]
    assert first.metrics == second.metrics
    assert np.array_equal(first.loss_curve, second.loss_curve)
    assert np.array_equal(first.reward_curve, second.reward_curve)
    probe = ZILobQuoteEnv(cfg).reset(seed=927)
    assert np.array_equal(first.agent.q_values(probe), second.agent.q_values(probe))
    assert np.array_equal(
        first.agent.action_distributions(probe), second.agent.action_distributions(probe)
    )
    # a different seed must not reproduce the same trajectory
    other = train_c51_on_zi_lob(**{**kwargs, "agent_config": _agent_config(TRAIN_SEED + 1)})  # type: ignore[arg-type]
    assert not np.array_equal(first.loss_curve, other.loss_curve)


@requires_torch
def test_train_entry_point_fail_closed() -> None:
    cfg = _env_config(horizon=10.0)
    agent = C51Agent(N_OBS_FEATURES, len(DEFAULT_QUOTE_ACTIONS), _agent_config(0))
    with pytest.raises(TypeError):
        train_c51_on_zi_lob(env_config="nope", total_decisions=10)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="total_decisions"):
        train_c51_on_zi_lob(env_config=cfg, total_decisions=0)
    with pytest.raises(ValueError, match="learn_every"):
        train_c51_on_zi_lob(env_config=cfg, total_decisions=10, learn_every=0)
    with pytest.raises(ValueError, match="reward_window"):
        train_c51_on_zi_lob(env_config=cfg, total_decisions=10, reward_window=0)
    with pytest.raises(TypeError, match="C51Agent"):
        train_c51_on_zi_lob(env_config=cfg, total_decisions=10, agent="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not both"):
        train_c51_on_zi_lob(
            env_config=cfg,
            total_decisions=10,
            agent=agent,
            agent_config=_agent_config(0),
        )
    mismatched = C51Agent(N_OBS_FEATURES + 1, len(DEFAULT_QUOTE_ACTIONS), _agent_config(0))
    with pytest.raises(ValueError, match="obs_dim"):
        train_c51_on_zi_lob(env_config=cfg, total_decisions=10, agent=mismatched)


@requires_torch
def test_compare_c51_baselines_is_sim_internal_namespaced() -> None:
    cfg = _env_config(horizon=20.0)
    agent = C51Agent(N_OBS_FEATURES, len(DEFAULT_QUOTE_ACTIONS), _agent_config(0))
    out = compare_c51_baselines(agent=agent, env_config=cfg, seeds=[931, 932])
    assert out["label"] == "SYNTHETIC"
    assert out["data_source"] == C51_RL_REVISION
    assert out["live_pnl_claim"] is False
    assert out["claim"] == "simulator_internal_diagnostic_only"
    assert out["kind"] == "c51_vs_baseline_comparison"
    assert sorted(out["policies"]) == ["as", "c51", "glft", "random"]
    metrics = out["metrics"]
    for name in ("c51", "random", "as", "glft"):
        assert math.isfinite(float(metrics[f"sim_internal_reward_mean_{name}"]))
        assert math.isfinite(float(metrics[f"sim_internal_mtm_pnl_final_{name}"]))
        assert float(metrics[f"max_abs_inventory_{name}"]) <= CAP
        assert len(out["rows"][name]) == 2
    for name in ("random", "as", "glft"):
        key = f"sim_internal_reward_gap_c51_minus_{name}_mean"
        assert float(metrics[key]) == pytest.approx(
            float(metrics["sim_internal_reward_mean_c51"])
            - float(metrics[f"sim_internal_reward_mean_{name}"])
        )
    # honesty: every key is either a plain diagnostic or sim_internal-namespaced
    for key, value in metrics.items():
        assert isinstance(value, float)
        low = key.lower()
        for tok in low.split("_"):
            if tok in FORBIDDEN_HEADLINE_TOKENS:
                raise AssertionError(f"forbidden headline token {tok!r} in {key!r}")
        if "pnl" in low.split("_"):
            assert low.startswith("sim_internal_"), key
    assert out["n_seeds"] == 2.0
    with pytest.raises(TypeError):
        compare_c51_baselines(agent="nope", env_config=cfg, seeds=[1])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        compare_c51_baselines(agent=agent, env_config=cfg, seeds=[])
    with pytest.raises(ValueError, match="classic_params"):
        compare_c51_baselines(agent=agent, env_config=cfg, seeds=[1], classic_params={"nope": 1.0})


# ---------------------------------------------------------------------------
# Torch-free import surface
# ---------------------------------------------------------------------------


def test_module_surface_is_torch_free() -> None:
    """The numpy core imports and runs without torch (deep_hedging pattern)."""
    assert not hasattr(c51_rl, "torch")
    atoms, dz = atom_support(-1.0, 1.0, 5)
    probs = np.full((1, atoms.size), 1.0 / atoms.size)
    out = categorical_projection(probs, 0.25, 0.5, atoms=atoms)
    assert np.allclose(out.sum(), 1.0)
    assert math.isfinite(dz)
    env = ZILobQuoteEnv(_env_config(horizon=10.0))
    row = run_env_episode(env, _fixed_action_policy(1), seed=933)
    assert row["label"] == "SYNTHETIC"
    assert math.isfinite(float(row["sim_internal_reward_mean"]))


def test_torch_guard_fails_closed_with_install_guidance(monkeypatch) -> None:
    """A blocked torch must surface the install guidance, not an AttributeError.

    Exercised through the real torch entry point (the agent constructor) rather
    than the private guard, so the whole lazy-import path is covered.
    """
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match="uv sync --extra nn"):
        C51Agent(N_OBS_FEATURES, len(DEFAULT_QUOTE_ACTIONS), C51AgentConfig())
