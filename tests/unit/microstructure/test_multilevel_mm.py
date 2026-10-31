"""Tests for microstructure/multilevel_mm.py — wave-17 lane.

Multi-level market making after Cheridito & Weiss (2026, arXiv:2608.18195,
verified against the arXiv abstract + PDF full text): a deep-set
(permutation-invariant) order encoder plus a logistic-normal policy over the
``S^{2(K+1)}`` allocation simplex, Hamilton apportionment onto unit lots,
potential-shaped Eq. 5 rewards (telescoping to Eq. 7), and the terminal Eq. 6
liquidation — all running on the composed ``zi_lob_simulator`` FIFO engine.

Everything is **labeled SYNTHETIC** correctness validation at TINY training
budgets — never market evidence, no live-trading claim. Tests assert the
plumbing, the paper's invariants (permutation invariance, simplex geometry,
argmax-preserving shaping, reward telescoping), determinism, and fail-closed
edges; they do NOT assert the trained policy beats GLFT (that comparison
lives in the documented-optional ``multilevel_mm_benchmark``). Simulator-
internal accounting is namespaced ``sim_internal_*``. Torch tests skip
cleanly when the ``nn`` extra is absent; the numpy core (simplex, shaping,
features, the allocation-policy and classic-policy session paths) runs
without torch.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from typing import Any

import numpy as np
import pytest

from quant_fund.microstructure import multilevel_mm as mlmm
from quant_fund.microstructure.multilevel_mm import (
    MMObs,
    MultiLevelMMConfig,
    MultiLevelSpec,
    RandomAllocationPolicy,
    action_component_map,
    alr_transform,
    build_base_features,
    collect_order_elements,
    deep_set_pool,
    evaluate_multilevel_mm,
    hamilton_apportionment,
    kappa_fractions,
    logistic_normal_logpdf,
    logistic_normal_sample,
    mm_reward,
    potential_shaping,
    run_multilevel_mm_session,
    shape_episode_rewards,
    shaped_optimal_q,
    simplex_transform,
    terminal_position_limit,
    terminal_reward,
)
from quant_fund.microstructure.zi_lob_simulator import (
    MM_TAG,
    ZILobSimulator,
    as_policy,
    glft_policy,
    santa_fe_config,
)

if importlib.util.find_spec("torch") is not None:
    from quant_fund.microstructure.multilevel_mm import (
        MultiLevelMMAgent,
        TrajectoryStep,
        train_multilevel_mm,
    )


def _torch_present() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # blocked or halted torch imports
        return False


_HAS_TORCH = _torch_present()
requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="multi-level MM training requires the nn extra (torch)"
)

# Forbidden *headline* metric tokens (mirrors research.catalog registry). "pnl"
# is permitted ONLY under the simulator-internal diagnostic namespace.
FORBIDDEN_HEADLINE_TOKENS = ("sharpe", "sortino", "calmar", "nav")

SPEC = MultiLevelSpec()  # K=3, M=4, cap=8
SPEC1 = MultiLevelSpec(n_levels=1, lots=2, inventory_cap=4)
CFG = santa_fe_config(seed=7)


def _all_keys(obj: object) -> list[str]:
    """Recursively collect mapping keys (dicts only; walk list/tuple values)."""
    keys: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.append(str(k))
            keys.extend(_all_keys(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            keys.extend(_all_keys(item))
    return keys


def _flat_obs(spec: MultiLevelSpec = SPEC, inventory: int = 0) -> MMObs:
    return MMObs(
        t=1.0,
        mid=100.0,
        best_bid=99.95,
        best_ask=100.05,
        inventory=inventory,
        tau=10.0,
        base_features=np.zeros(spec.n_base_features),
        order_feats=np.zeros((0, 2)),
        order_slots=np.zeros(0, dtype=np.int64),
    )


def _alloc_bundle(spec: MultiLevelSpec = SPEC, seed: int = 3, **kw: Any) -> dict:
    return run_multilevel_mm_session(
        config=CFG,
        horizon=60.0,
        decision_interval=1.0,
        allocation_policy=RandomAllocationPolicy(spec, seed=seed),
        spec=spec,
        **kw,
    )


# ---------------------------------------------------------------------------
# Problem spec + action simplex
# ---------------------------------------------------------------------------


def test_spec_dimensions() -> None:
    assert SPEC.n_components == 2 * 3 + 3
    assert SPEC.n_logits == 2 * 3 + 2
    assert SPEC.n_order_slots == 2 * 3
    assert SPEC.n_base_features == 4 * 3 + 10
    assert SPEC.state_dim == 2 * 3 * SPEC.emb_dim + SPEC.n_base_features


def test_spec_fail_closed() -> None:
    with pytest.raises(ValueError):
        MultiLevelSpec(n_levels=0)
    with pytest.raises(ValueError):
        MultiLevelSpec(lots=0)
    with pytest.raises(ValueError):
        MultiLevelSpec(depth_scale=0.0)


def test_action_component_map_layout() -> None:
    m = action_component_map(3)
    assert len(m) == 9
    assert m[0].kind == "hold"
    assert (m[1].kind, m[1].side) == ("market", "buy")
    assert [(c.side, c.level_offset) for c in m[2:5]] == [("buy", i) for i in range(3)]
    assert (m[5].kind, m[5].side) == ("market", "sell")
    assert [(c.side, c.level_offset) for c in m[6:9]] == [("sell", i) for i in range(3)]


def test_action_component_map_k1_is_single_quote() -> None:
    """K=1 degenerates to single-quote placement: one bid + one ask leg."""
    m = action_component_map(1)
    assert len(m) == 5
    assert m[2].kind == "limit" and m[2].side == "buy" and m[2].level_offset == 0
    assert m[4].kind == "limit" and m[4].side == "sell" and m[4].level_offset == 0


def test_hamilton_exact_split() -> None:
    w = np.array([0.1, 0.25, 0.15, 0.05, 0.2, 0.1, 0.05, 0.05, 0.05])
    lots = hamilton_apportionment(w, 4)
    assert lots.dtype == np.int64
    assert int(lots.sum()) == 4
    assert np.all(lots >= 0)
    # largest-remainder: the top quotas are w*M = (.4,1,.6,.2,.8,.4,.2,.2,.2)
    # floors (0,1,0,0,0,0,0,0,0) -> 3 remainders to (.6,.8,.4 ties -> lowest idx)
    assert lots.tolist() == [1, 1, 1, 0, 1, 0, 0, 0, 0]


def test_hamilton_renormalizes_and_sums() -> None:
    rng = np.random.default_rng(0)
    for m in (1, 2, 7):
        w = rng.random(9) * 5.0  # not normalized on purpose
        lots = hamilton_apportionment(w, m)
        assert int(lots.sum()) == m


def test_hamilton_tie_break_lowest_index() -> None:
    w = np.full(4, 0.25)
    lots = hamilton_apportionment(w, 2)
    assert lots.tolist() == [1, 1, 0, 0]  # equal remainders: lowest index first


def test_hamilton_fail_closed() -> None:
    with pytest.raises(ValueError):
        hamilton_apportionment([0.5, -0.1, 0.6], 3)
    with pytest.raises(ValueError):
        hamilton_apportionment([], 3)
    with pytest.raises(ValueError):
        hamilton_apportionment([0.0, 0.0], 3)
    with pytest.raises(ValueError):
        hamilton_apportionment([0.5, float("nan")], 3)
    with pytest.raises(ValueError):
        hamilton_apportionment([0.5, 0.5], 0)


# ---------------------------------------------------------------------------
# Logistic-normal layer
# ---------------------------------------------------------------------------


def test_simplex_transform_open_simplex() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        a = simplex_transform(rng.normal(size=8) * 4)
        assert a.shape == (9,)
        assert np.all(a > 0.0)
        assert math.isclose(float(a.sum()), 1.0, rel_tol=0, abs_tol=1e-12)


def test_simplex_transform_reference_value() -> None:
    x = np.array([0.5, -1.0])
    a = simplex_transform(x)
    denom = 1.0 + math.exp(0.5) + math.exp(-1.0)
    assert math.isclose(a[0], 1.0 / denom, rel_tol=1e-12)
    assert math.isclose(a[1], math.exp(0.5) / denom, rel_tol=1e-12)


def test_alr_is_exact_inverse() -> None:
    rng = np.random.default_rng(2)
    for _ in range(10):
        x = rng.normal(size=8)
        a = simplex_transform(x)
        assert np.allclose(alr_transform(a), x, atol=1e-12)
        assert np.allclose(simplex_transform(alr_transform(a)), a, atol=1e-12)


def test_alr_fail_closed() -> None:
    with pytest.raises(ValueError):
        alr_transform([0.5, 0.0, 0.5])  # zero component (boundary)
    with pytest.raises(ValueError):
        alr_transform([0.5, 0.4])  # sums to 0.9
    with pytest.raises(ValueError):
        alr_transform([1.0])


def test_logistic_normal_logpdf_formula() -> None:
    """Independent path: ALR gaussian density times the Jacobian correction."""
    from scipy.stats import multivariate_normal

    rng = np.random.default_rng(5)
    for _ in range(10):
        a = simplex_transform(rng.normal(size=4))
        mu = rng.normal(size=4)
        lv = rng.uniform(-2.0, 0.5, size=4)
        y = np.log(a[1:] / a[0])
        expect = float(
            multivariate_normal.logpdf(y, mean=mu, cov=np.diag(np.exp(lv))) - np.log(a).sum()
        )
        assert math.isclose(logistic_normal_logpdf(a, mu, lv), expect, rel_tol=1e-10, abs_tol=1e-10)


def test_logistic_normal_density_integrates_to_one() -> None:
    """Grid-integrate the 3-component (K=1) density over the 2-simplex."""
    n = 240
    us = (np.arange(n) + 0.5) / n
    mu = np.array([0.4, -0.2])
    lv = np.array([-0.5, -0.8])
    mass = 0.0
    for u in us:
        for v in us:
            if u + v >= 1.0:
                continue
            a = np.array([1.0 - u - v, u, v])
            mass += math.exp(logistic_normal_logpdf(a, mu, lv)) / (n * n)
    assert 0.95 < mass < 1.05


def test_logistic_normal_sample_seeded() -> None:
    mu = np.zeros(8)
    lv = np.zeros(8)
    a1 = logistic_normal_sample(mu, lv, np.random.default_rng(9))
    a2 = logistic_normal_sample(mu, lv, np.random.default_rng(9))
    a3 = logistic_normal_sample(mu, lv, np.random.default_rng(10))
    assert np.array_equal(a1, a2)
    assert not np.array_equal(a1, a3)
    assert np.all(a1 > 0.0) and math.isclose(float(a1.sum()), 1.0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Potential-based reward shaping + paper rewards
# ---------------------------------------------------------------------------


def test_potential_shaping_value() -> None:
    assert math.isclose(potential_shaping(2.0, 5.0, gamma=1.0), 3.0)
    assert math.isclose(potential_shaping(2.0, 5.0, gamma=0.5), 0.5)


def test_potential_shaping_fail_closed() -> None:
    with pytest.raises(ValueError):
        potential_shaping(1.0, float("nan"))
    with pytest.raises(ValueError):
        potential_shaping(1.0, 2.0, gamma=0.0)
    with pytest.raises(ValueError):
        potential_shaping(1.0, 2.0, gamma=1.5)


def test_shape_episode_rewards_telescopes() -> None:
    """sum r' = sum r + gamma^N * Phi_N - Phi_0 (action-independent offset)."""
    rng = np.random.default_rng(4)
    r = rng.normal(size=6)
    phi = rng.normal(size=7)
    out = shape_episode_rewards(r, phi, gamma=1.0)
    assert np.allclose(out, r + phi[1:] - phi[:-1])
    assert math.isclose(float(out.sum()), float(r.sum()) + phi[-1] - phi[0], rel_tol=1e-12)
    with pytest.raises(ValueError):
        shape_episode_rewards(r, phi[:-1])  # wrong length


def test_shaped_optimal_q_preserves_argmax() -> None:
    """Ng-Harada-Russell invariance: Q'*(s,a) = Q*(s,a) - Phi(s)."""
    rng = np.random.default_rng(6)
    for _ in range(10):
        q = rng.normal(size=9)
        phi = float(rng.normal())
        qp = shaped_optimal_q(q, phi)
        assert np.allclose(qp, q - phi)
        assert int(np.argmax(qp)) == int(np.argmax(q))


def test_mm_reward_closed_form() -> None:
    # Eq. 5: r = (cf + (Q' p' - Q p) - gamma |Q'|) / M
    r = mm_reward(
        cash_flow=0.5, q_prev=2, p_prev=10.0, q_next=-1, p_next=11.0, inv_gamma=0.01, lots=4
    )
    expect = (0.5 + (-1 * 11.0 - 2 * 10.0) - 0.01 * 1) / 4
    assert math.isclose(r, expect, rel_tol=1e-12)


def test_terminal_position_limit_and_reward() -> None:
    assert terminal_position_limit(4, 1.0) == 4
    assert terminal_position_limit(4, 0.5) == 2
    assert terminal_position_limit(5, 0.5) == 3  # ceil
    # Eq. 6: g = (p (Q+ - Q) + MO) / M; liquidating 2 of +5 at p=10 for 19.5.
    g = terminal_reward(q_n=5, p_n=10.0, q_plus=3, mo_cash=19.5, lots=4)
    assert math.isclose(g, (10.0 * (3 - 5) + 19.5) / 4, rel_tol=1e-12)
    with pytest.raises(ValueError):  # inconsistent liquidation accounting
        terminal_reward(q_n=2, p_n=10.0, q_plus=3, mo_cash=0.0, lots=4)
    with pytest.raises(ValueError):  # sign flip is not a partial liquidation
        terminal_reward(q_n=2, p_n=10.0, q_plus=-1, mo_cash=0.0, lots=4)


# ---------------------------------------------------------------------------
# Deep-set encoder + features
# ---------------------------------------------------------------------------


def test_deep_set_pool_permutation_invariant() -> None:
    rng = np.random.default_rng(8)
    feats = rng.normal(size=(6, 2))
    slots = np.array([0, 1, 0, 2, 1, 0])
    ref = deep_set_pool(feats, slots, 3)
    for _ in range(5):
        perm = rng.permutation(6)
        out = deep_set_pool(feats[perm], slots[perm], 3)
        assert np.array_equal(out, ref)


def test_deep_set_pool_means_and_empty_levels() -> None:
    feats = np.array([[1.0, 3.0], [3.0, 5.0], [9.0, 9.0]])
    slots = np.array([0, 0, 2])
    out = deep_set_pool(feats, slots, 3).reshape(3, 2)
    assert np.allclose(out[0], [2.0, 4.0])  # mean of slot-0 rows
    assert np.allclose(out[1], [0.0, 0.0])  # empty level -> zeros
    assert np.allclose(out[2], [9.0, 9.0])


def test_deep_set_pool_fail_closed() -> None:
    with pytest.raises(ValueError):
        deep_set_pool(np.zeros((2, 2)), np.array([0]), 2)  # slot/row mismatch
    with pytest.raises(ValueError):
        deep_set_pool(np.zeros((2, 2)), np.array([0, 5]), 2)  # slot out of range


def test_collect_order_elements_slots() -> None:
    sim = ZILobSimulator(CFG)
    assert sim.best_bid_level is not None and sim.best_ask_level is not None
    bb, ba = sim.best_bid_level, sim.best_ask_level
    orders: dict[int, tuple[Any, int]] = {}
    # two bids: at the best bid (slot 0) and one tick deeper (slot 1)
    for lvl in (bb, bb - 1):
        oid = sim.submit_limit_order("buy", sim.level_to_price(lvl), MM_TAG)
        orders[oid] = ("buy", lvl)
    oid = sim.submit_limit_order("sell", sim.level_to_price(ba + 2), MM_TAG)
    orders[oid] = ("sell", ba + 2)
    feats, slots = collect_order_elements(orders, sim, SPEC)
    assert feats.shape == (3, 2) and slots.shape == (3,)
    assert sorted(slots.tolist()) == [0, 1, SPEC.n_levels + 2]
    # unit-lot size feature is 1/M; queue feature is (position + 1)/scale.
    assert np.allclose(feats[:, 1], 1.0 / SPEC.lots)


def test_kappa_fractions_buckets() -> None:
    sim = ZILobSimulator(CFG)
    bb, ba = sim.best_bid_level, sim.best_ask_level
    assert bb is not None and ba is not None
    orders: dict[int, tuple[Any, int]] = {}
    for lvl, side in ((bb, "buy"), (bb - 5, "buy"), (ba, "sell")):
        oid = sim.submit_limit_order(side, sim.level_to_price(lvl), MM_TAG)
        orders[oid] = (side, lvl)
    kap = kappa_fractions(orders, sim, SPEC)
    assert kap.shape == (2 * (SPEC.n_levels + 1),)
    assert math.isclose(kap[0], 1 / SPEC.lots)  # bid at the best bid
    assert math.isclose(kap[SPEC.n_levels], 1 / SPEC.lots)  # >=K bucket (5 ticks)
    assert math.isclose(kap[SPEC.n_levels + 1], 1 / SPEC.lots)  # ask at best ask
    assert math.isclose(float(kap.sum()), 3 / SPEC.lots)


def test_build_base_features_shape_and_ranges() -> None:
    out = build_base_features(
        SPEC,
        bid_ret=0.5,
        ask_ret=-0.3,
        bid_volumes=[10.0, 20.0, 30.0],
        ask_volumes=[40.0, 50.0, 60.0],
        mid_drift=0.01,
        mo_imbalance=0.2,
        lo_rate=1.0,
        cxl_rate=0.5,
        t_frac=0.25,
        q_norm=0.5,
        kappa=np.zeros(8),
    )
    assert out.shape == (SPEC.n_base_features,)
    assert np.all(np.isfinite(out))
    assert out[2] == 0.1  # volume / depth_scale(100)
    with pytest.raises(ValueError):
        build_base_features(
            SPEC,
            bid_ret=0.0,
            ask_ret=0.0,
            bid_volumes=[1.0],  # wrong length
            ask_volumes=[1.0, 1.0, 1.0],
            mid_drift=0.0,
            mo_imbalance=0.0,
            lo_rate=0.0,
            cxl_rate=0.0,
            t_frac=0.5,
            q_norm=0.0,
            kappa=np.zeros(8),
        )


def test_random_allocation_policy_seeded_simplex() -> None:
    obs = _flat_obs()
    p1 = RandomAllocationPolicy(SPEC, seed=11)
    p2 = RandomAllocationPolicy(SPEC, seed=11)
    w1, w2 = p1(obs), p2(obs)
    assert np.array_equal(w1, w2)
    assert w1.shape == (SPEC.n_components,)
    assert np.all(w1 > 0.0) and math.isclose(float(w1.sum()), 1.0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Session runner (numpy arms: allocation policy + classic QuotePolicy)
# ---------------------------------------------------------------------------


def test_session_bundle_schema_and_synthetic_labels() -> None:
    b = _alloc_bundle()
    assert b["label"] == "SYNTHETIC"
    assert b["research_only"] is True and b["live_pnl_claim"] is False
    assert b["session_completed"] is True
    assert b["n_decisions"] > 0
    keys = [k.lower() for k in _all_keys(b)]
    for tok in FORBIDDEN_HEADLINE_TOKENS:
        assert not any(tok in k for k in keys), f"forbidden token {tok!r}"
    for k in keys:
        if "pnl" in k:
            assert k == "live_pnl_claim" or k.startswith("sim_internal_")


def test_session_determinism_bit_identical() -> None:
    b1 = _alloc_bundle()
    b2 = _alloc_bundle()
    assert b1["sim_internal_mtm_pnl_final"] == b2["sim_internal_mtm_pnl_final"]
    assert b1["sim_internal_reward_path"] == b2["sim_internal_reward_path"]
    assert b1["inventory_path"] == b2["inventory_path"]
    assert b1["n_fills"] == b2["n_fills"]


def test_session_telescoping_and_cash_consistency() -> None:
    """Paper Eq. 7: shaped rewards + g telescope to the unshaped sum."""
    b = _alloc_bundle()
    assert abs(b["sim_internal_telescope_residual"]) < 1e-9
    assert abs(b["sim_internal_cash_consistency_residual"]) < 1e-9


def test_session_trajectory_covers_decisions() -> None:
    b = _alloc_bundle(return_trajectory=True)
    traj = b["trajectory"]
    assert len(traj) == b["n_decisions"]
    total = sum(s.reward for s in traj)
    # Every settled interval reward lands in a trajectory step.
    assert math.isclose(total, b["sim_internal_reward_sum"], rel_tol=0, abs_tol=1e-12)
    for s in traj:
        assert s.action.shape == (SPEC.n_components,)
        assert math.isfinite(s.reward)


def test_session_k1_runs_as_single_quote() -> None:
    b = _alloc_bundle(spec=SPEC1, seed=4)
    assert b["session_completed"] is True
    assert b["n_levels"] == 1 and b["n_decisions"] > 0
    assert abs(b["sim_internal_telescope_residual"]) < 1e-9


def test_session_glft_policy_mode() -> None:
    pol = glft_policy(gamma=1.0, sigma=0.02, kappa=1000.0, a_fill=1.0, tick=CFG.tick)
    b = run_multilevel_mm_session(config=CFG, horizon=60.0, decision_interval=1.0, policy=pol)
    assert b["policy_kind"] == "policy"
    assert b["n_decisions"] > 0 and b["n_mm_posted"] > 0
    assert abs(b["sim_internal_telescope_residual"]) < 1e-9


def test_session_as_policy_mode() -> None:
    pol = as_policy(gamma=0.002, sigma=0.02, kappa=1000.0, tick=CFG.tick)
    b = run_multilevel_mm_session(config=CFG, horizon=60.0, decision_interval=1.0, policy=pol)
    assert b["session_completed"] is True and b["n_decisions"] > 0


def test_session_inventory_cap_gating() -> None:
    """All-in market-buy allocation is gated once inventory reaches the cap."""
    lots, cap = SPEC.lots, 4
    w = np.zeros(SPEC.n_components)
    w[1] = 1.0  # all weight on the market-buy leg

    def greedy(obs: MMObs) -> np.ndarray:
        return w

    b = run_multilevel_mm_session(
        config=CFG,
        horizon=40.0,
        decision_interval=1.0,
        allocation_policy=greedy,
        spec=SPEC,
        inventory_cap=cap,
    )
    # Post-hoc gate fires at inv >= cap; an earlier leg can overshoot by M-1.
    assert b["max_abs_inventory"] <= cap + lots - 1
    assert b["n_inventory_gated"] >= 1
    assert b["inventory_terminal_plus"] <= terminal_position_limit(lots, 1.0)


def test_session_fail_closed_edges() -> None:
    with pytest.raises(ValueError):  # two arms at once
        run_multilevel_mm_session(
            config=CFG,
            horizon=10.0,
            allocation_policy=RandomAllocationPolicy(SPEC, seed=0),
            policy=glft_policy(gamma=1.0, sigma=0.02, kappa=1.0, a_fill=1.0, tick=CFG.tick),
            spec=SPEC,
        )
    with pytest.raises(ValueError):  # no arm
        run_multilevel_mm_session(config=CFG, horizon=10.0)
    with pytest.raises(ValueError):  # degenerate horizon
        run_multilevel_mm_session(
            config=CFG,
            horizon=0.0,
            allocation_policy=RandomAllocationPolicy(SPEC, seed=0),
            spec=SPEC,
        )
    with pytest.raises(ValueError):  # training without the agent
        run_multilevel_mm_session(
            config=CFG,
            horizon=10.0,
            allocation_policy=RandomAllocationPolicy(SPEC, seed=0),
            spec=SPEC,
            training=True,
        )
    with pytest.raises(ValueError):  # allocation arm without a spec
        run_multilevel_mm_session(
            config=CFG,
            horizon=10.0,
            allocation_policy=RandomAllocationPolicy(SPEC, seed=0),
        )
    with pytest.raises(ValueError):  # allocation returns the wrong shape
        run_multilevel_mm_session(
            config=CFG,
            horizon=10.0,
            allocation_policy=lambda obs: np.ones(3),
            spec=SPEC,
        )
    with pytest.raises(TypeError):  # wrong types
        run_multilevel_mm_session(config=CFG, horizon=10.0, agent="not-an-agent")


def test_evaluate_multilevel_mm_metrics() -> None:
    out = evaluate_multilevel_mm(
        config=CFG,
        horizon=40.0,
        n_seeds=2,
        seed_base=900,
        decision_interval=1.0,
        spec=SPEC,
        arms=("glft", "random"),
    )
    assert out["label"] == "SYNTHETIC"
    m = out["metrics"]
    assert m["glft_session_completion_rate"] == 1.0
    assert m["random_session_completion_rate"] == 1.0
    assert math.isfinite(m["sim_internal_mtm_pnl_final_mean_glft"])
    assert math.isfinite(m["sim_internal_mtm_pnl_final_mean_random"])


def test_multilevel_mm_import_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Torch-gated paths fail closed with install guidance, import stays clean."""
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match="nn"):
        mlmm._torch()
    # numpy arms stay usable
    b = _alloc_bundle()
    assert b["session_completed"] is True


# ---------------------------------------------------------------------------
# Torch-gated agent tests
# ---------------------------------------------------------------------------


@requires_torch
def test_agent_io_shapes() -> None:
    agent = MultiLevelMMAgent(SPEC, MultiLevelMMConfig(seed=0))
    obs = _flat_obs()
    mu = agent.mu_logits([obs, obs])
    v = agent.values([obs, obs])
    assert mu.shape == (2, SPEC.n_logits)
    assert v.shape == (2,)
    w = simplex_transform(mu[0])
    lp = agent.log_probs([obs], np.asarray([w]))
    assert lp.shape == (1,) and math.isfinite(lp[0])


@requires_torch
def test_agent_act_simplex_and_lots() -> None:
    agent = MultiLevelMMAgent(SPEC, MultiLevelMMConfig(seed=0))
    obs = _flat_obs()
    w, lots = agent.act(obs, rng=np.random.default_rng(1), sample=True)
    assert w.shape == (SPEC.n_components,)
    assert np.all(w > 0.0) and math.isclose(float(w.sum()), 1.0, abs_tol=1e-12)
    assert lots.shape == (SPEC.n_components,)
    assert int(lots.sum()) == SPEC.lots and np.all(lots >= 0)


@requires_torch
def test_agent_determinism() -> None:
    obs = _flat_obs()
    a1 = MultiLevelMMAgent(SPEC, MultiLevelMMConfig(seed=5))
    a2 = MultiLevelMMAgent(SPEC, MultiLevelMMConfig(seed=5))
    assert np.array_equal(a1.mu_logits([obs]), a2.mu_logits([obs]))
    r1, r2 = np.random.default_rng(3), np.random.default_rng(3)
    w1, _ = a1.act(obs, rng=r1, sample=True)
    w2, _ = a2.act(obs, rng=r2, sample=True)
    assert np.array_equal(w1, w2)
    w3, _ = a1.act(obs, sample=False)  # deterministic mean action
    assert np.array_equal(a1.mu_logits([obs]), a1.mu_logits([obs]))
    assert math.isclose(float(w3.sum()), 1.0, abs_tol=1e-12)


@requires_torch
def test_agent_permutation_invariant_encoding() -> None:
    """Permuting the order-element rows leaves the encoding identical."""
    agent = MultiLevelMMAgent(SPEC, MultiLevelMMConfig(seed=0))
    rng = np.random.default_rng(7)
    feats = rng.uniform(0.0, 0.2, size=(6, 2))
    slots = np.array([0, 1, 2, 3, 4, 5])
    obs_a = MMObs(
        t=0.0,
        mid=100.0,
        best_bid=99.9,
        best_ask=100.1,
        inventory=0,
        tau=5.0,
        base_features=np.zeros(SPEC.n_base_features),
        order_feats=feats,
        order_slots=slots,
    )
    perm = rng.permutation(6)
    obs_b = MMObs(
        t=0.0,
        mid=100.0,
        best_bid=99.9,
        best_ask=100.1,
        inventory=0,
        tau=5.0,
        base_features=np.zeros(SPEC.n_base_features),
        order_feats=feats[perm],
        order_slots=slots[perm],
    )
    assert np.allclose(agent.mu_logits([obs_a]), agent.mu_logits([obs_b]), atol=1e-6)
    assert np.allclose(agent.values([obs_a]), agent.values([obs_b]), atol=1e-6)


@requires_torch
def test_agent_init_bias_favors_trade_components() -> None:
    """Paper App. A.1: actor output bias 1s -> a_0 starts below uniform."""
    agent = MultiLevelMMAgent(SPEC, MultiLevelMMConfig(seed=0))
    w = simplex_transform(agent.mu_logits([_flat_obs()])[0])
    assert w[0] < 1.0 / SPEC.n_components


@requires_torch
def test_agent_obs_fail_closed() -> None:
    agent = MultiLevelMMAgent(SPEC, MultiLevelMMConfig(seed=0))
    bad = MMObs(
        t=0.0,
        mid=1.0,
        best_bid=0.9,
        best_ask=1.1,
        inventory=0,
        tau=1.0,
        base_features=np.zeros(3),
        order_feats=np.zeros((0, 2)),
        order_slots=np.zeros(0, dtype=np.int64),
    )
    with pytest.raises(ValueError):
        agent.act(bad, sample=False)
    with pytest.raises(ValueError):  # non-simplex action
        agent.log_probs([_flat_obs()], np.asarray([np.full(SPEC.n_components, 0.5)]))


@requires_torch
def test_policy_gradient_direction() -> None:
    """Planted advantage pushes up log pi of the advantaged action (Eq. 14)."""
    agent = MultiLevelMMAgent(SPEC, MultiLevelMMConfig(seed=0))
    obs = _flat_obs()
    a_adv = simplex_transform(np.array([3.0] + [0.0] * (SPEC.n_logits - 1)))
    lp0 = float(agent.log_probs([obs], np.asarray([a_adv]))[0])
    steps = [TrajectoryStep(obs=obs, action=a_adv, reward=1.0)] * 4
    for _ in range(8):
        agent.learn([(steps, 0.0)])
    lp1 = float(agent.log_probs([obs], np.asarray([a_adv]))[0])
    assert lp1 > lp0


@requires_torch
def test_critic_fits_returns() -> None:
    """Repeated fitting on one episode batch shrinks the critic MSE."""
    agent = MultiLevelMMAgent(SPEC, MultiLevelMMConfig(seed=1))
    obs = _flat_obs()
    a = simplex_transform(np.zeros(SPEC.n_logits))
    steps = [TrajectoryStep(obs=obs, action=a, reward=0.5)] * 4
    ep = [(steps, 0.3)]
    m0 = agent.evaluate_loss(ep)["critic_mse"]
    for _ in range(20):
        agent.learn(ep)
    m1 = agent.evaluate_loss(ep)["critic_mse"]
    assert m1 < m0


@requires_torch
def test_agent_session_and_training_smoke() -> None:
    agent = MultiLevelMMAgent(SPEC, MultiLevelMMConfig(seed=0))
    b = run_multilevel_mm_session(config=CFG, horizon=40.0, decision_interval=1.0, agent=agent)
    assert b["session_completed"] is True
    assert b["n_decisions"] > 0
    assert abs(b["sim_internal_telescope_residual"]) < 1e-9
    out = train_multilevel_mm(
        agent=agent, config=CFG, horizon=30.0, n_episodes=2, decision_interval=1.0
    )
    assert out["n_updates"] == 2
    assert all(math.isfinite(x) for x in out["loss_curve"])
    keys = [k.lower() for k in _all_keys(out)]
    for tok in FORBIDDEN_HEADLINE_TOKENS:
        assert not any(tok in k for k in keys)


@requires_torch
def test_evaluate_agent_vs_arms_paired_seeds() -> None:
    agent = MultiLevelMMAgent(SPEC, MultiLevelMMConfig(seed=0))
    out = evaluate_multilevel_mm(
        config=CFG,
        horizon=40.0,
        agent=agent,
        n_seeds=2,
        seed_base=950,
        decision_interval=1.0,
        arms=("agent", "glft", "random"),
    )
    m = out["metrics"]
    for arm in ("agent", "glft", "random"):
        assert m[f"{arm}_session_completion_rate"] == 1.0
        assert math.isfinite(m[f"sim_internal_mtm_pnl_final_mean_{arm}"])
    assert math.isfinite(m["sim_internal_mtm_pnl_gap_agent_minus_glft_mean"])
    assert math.isfinite(m["sim_internal_mtm_pnl_gap_agent_minus_random_mean"])
    keys = [k.lower() for k in _all_keys(out)]
    for tok in FORBIDDEN_HEADLINE_TOKENS:
        assert not any(tok in k for k in keys)


@requires_torch
def test_agent_k1_single_quote_reduction() -> None:
    agent = MultiLevelMMAgent(SPEC1, MultiLevelMMConfig(seed=0))
    obs = _flat_obs(SPEC1)
    w, lots = agent.act(obs, sample=False)
    assert w.shape == (5,)
    assert int(lots.sum()) == SPEC1.lots


@requires_torch
def test_agent_constructor_fail_closed() -> None:
    with pytest.raises(TypeError):
        MultiLevelMMAgent("not-a-spec", MultiLevelMMConfig(seed=0))
    with pytest.raises(TypeError):
        MultiLevelMMAgent(SPEC, "not-a-config")
    with pytest.raises(ValueError):
        MultiLevelMMConfig(lr=0.0)
    with pytest.raises(ValueError):
        MultiLevelMMConfig(hidden_actor=())
