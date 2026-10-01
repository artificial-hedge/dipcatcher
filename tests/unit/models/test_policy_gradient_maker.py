"""SYNTHETIC PPO/GAE and FIFO/cash accounting correctness, not market evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quant_fund.microstructure.zi_lob_simulator import ZILobConfig
from quant_fund.models.policy_gradient_maker import (
    N_ACTIONS,
    FrozenMakerPolicy,
    MakerConfig,
    MakerEnvironment,
    MakerScenario,
    PPOConfig,
    _ppo_objective,
    clipped_surrogate,
    evaluate_ppo_maker,
    generalized_advantages,
    replay_maker_episode,
    run_maker_episode,
    train_ppo_maker,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="optional torch unavailable")


def scenario(seed: int, regime: Any = "stationary") -> MakerScenario:
    return MakerScenario(
        f"SYNTHETIC_EPISODE_{seed}", ZILobConfig(seed=seed, mu=2.0, lam=0.08, init_depth=2), regime
    )


def env_config() -> MakerConfig:
    return MakerConfig(
        decisions=16,
        events_per_decision=4,
        inventory_cap=3,
        initial_inventory=1,
        target_inventory=1,
        initial_cash=300.0,
    )


def ppo_config() -> PPOConfig:
    return PPOConfig(rounds=2, epochs=2, minibatch=8, hidden=8)


@pytest.fixture(autouse=True)
def cpu_threads() -> Iterator[None]:
    if not HAS_TORCH:
        yield
        return
    import torch

    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


@pytest.fixture
def fitted() -> FrozenMakerPolicy:
    return train_ppo_maker((scenario(3), scenario(5)), env_config(), ppo_config())


@pytest.mark.parametrize("gamma,lam", [(0.0, 0.0), (0.7, 0.0), (0.7, 0.5), (1.0, 1.0)])
def test_gae_matches_independent_discounted_delta_sums_and_terminal_boundaries(
    gamma: float, lam: float
) -> None:
    rewards = np.array([1.0, -0.3, 2.0, -0.2])
    values = np.array([0.2, 0.7, 0.6, -0.1, 100.0])
    terminated = (False, True, False, False)
    expected = []
    for t in range(len(rewards)):
        total = 0.0
        for j in range(t, len(rewards)):
            delta = rewards[j] + gamma * (not terminated[j]) * values[j + 1] - values[j]
            total += (gamma * lam) ** (j - t) * delta
            if terminated[j]:
                break
        expected.append(total)
    advantages, targets = generalized_advantages(rewards, values, terminated, gamma=gamma, lam=lam)
    assert advantages == pytest.approx(expected)
    assert targets == pytest.approx(np.array(expected) + values[:-1])
    changed = values.copy()
    changed[2:] += 500
    again, _ = generalized_advantages(rewards, changed, terminated, gamma=gamma, lam=lam)
    assert again[:2] == pytest.approx(advantages[:2])


def test_ppo_clipping_uses_pessimistic_branch_for_both_advantage_signs() -> None:
    ratios = np.array([1.4, 0.6, 1.1, 1.4, 0.6, 0.9])
    advantages = np.array([2.0, 2.0, 2.0, -3.0, -3.0, -3.0])
    expected = sum([2.4, 1.2, 2.2, -4.2, -2.4, -2.7]) / 6
    assert clipped_surrogate(np.log(ratios), np.zeros(6), advantages, clip=0.2) == pytest.approx(
        expected
    )


@requires_torch
def test_actual_ppo_actor_critic_objective_entropy_mask_and_finite_difference_gradients() -> None:
    import torch

    config = PPOConfig(rounds=1, epochs=1, value_coefficient=0.4, entropy_coefficient=0.03)
    initial = np.array([[0.4, -0.2, 0.5], [-0.5, 0.8, 0.1]])
    masks = torch.tensor([[True, True, False], [True, True, True]])
    actions = torch.tensor([0, 2])
    old = torch.tensor([math.log(0.5), math.log(0.2)], dtype=torch.float64)
    adv = torch.tensor([0.8, -0.4], dtype=torch.float64)
    targets = torch.tensor([0.2, 0.7], dtype=torch.float64)
    logits = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    values = torch.tensor([0.3, 0.1], dtype=torch.float64, requires_grad=True)
    loss, diagnostic = _ppo_objective(
        logits, values, masks, actions, old, adv, targets, config, torch
    )
    probabilities = []
    entropy = []
    selected = []
    for row, mask, action in zip(initial, masks.numpy(), actions.numpy(), strict=True):
        weights = [math.exp(float(v)) if m else 0.0 for v, m in zip(row, mask, strict=True)]
        p = [w / sum(weights) for w in weights]
        probabilities.append(p)
        entropy.append(-sum(v * math.log(v) for v in p if v > 0))
        selected.append(math.log(p[action]))
    expected_surrogate = clipped_surrogate(
        np.array(selected), old.numpy(), adv.numpy(), clip=config.clip
    )
    expected_value_error = ((0.3 - 0.2) ** 2 + (0.1 - 0.7) ** 2) / 2
    expected_loss = (
        -expected_surrogate
        + config.value_coefficient * expected_value_error
        - config.entropy_coefficient * np.mean(entropy)
    )
    assert float(loss.detach()) == pytest.approx(expected_loss)
    assert float(diagnostic["entropy"].detach()) == pytest.approx(np.mean(entropy))
    grad_logits, grad_values = torch.autograd.grad(loss, (logits, values))
    assert grad_logits[0, 2] == 0 and torch.isfinite(grad_logits).all()
    epsilon = 1e-6
    for i, j in ((0, 0), (1, 2)):
        shifted = initial.copy()
        shifted[i, j] += epsilon
        plus = _ppo_objective(
            torch.tensor(shifted), values.detach(), masks, actions, old, adv, targets, config, torch
        )[0]
        shifted[i, j] -= 2 * epsilon
        minus = _ppo_objective(
            torch.tensor(shifted), values.detach(), masks, actions, old, adv, targets, config, torch
        )[0]
        assert float(grad_logits[i, j]) == pytest.approx(
            float((plus - minus) / (2 * epsilon)), abs=1e-8
        )
    assert grad_values.numpy() == pytest.approx(config.value_coefficient * np.array([0.1, -0.6]))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(decisions=True),
        dict(events_per_decision=0),
        dict(inventory_cap=0),
        dict(initial_inventory=9),
        dict(initial_cash=-1),
        dict(maker_fee_bps=-1),
        dict(quote_floor=math.nan),
    ],
)
def test_invalid_environment_configuration_fails_closed(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        MakerConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(rounds=0),
        dict(epochs=9),
        dict(minibatch=3),
        dict(hidden=1000),
        dict(learning_rate=True),
        dict(clip=0),
        dict(seed=-1),
    ],
)
def test_invalid_ppo_configuration_fails_closed(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        PPOConfig(**kwargs)


def test_action_mask_financing_inventory_price_limits_and_crossing() -> None:
    cfg = env_config()
    empty = MakerEnvironment(scenario(1), replace(cfg, initial_inventory=0, initial_cash=1.0))
    assert empty.observation().action_mask == (True,) + (False,) * (N_ACTIONS - 1)
    with pytest.raises(ValueError, match="mask"):
        empty.step(1)
    capped = MakerEnvironment(scenario(1), replace(cfg, initial_inventory=3))
    levels = capped._levels(1)
    assert levels is not None and levels[0] is None and levels[1] is not None
    cash_poor = MakerEnvironment(scenario(1), replace(cfg, initial_cash=0.0))
    levels = cash_poor._levels(1)
    assert levels is not None and levels[0] is None
    limits = MakerEnvironment(scenario(1), replace(cfg, quote_floor=150.0))
    assert not any(limits.observation().action_mask[1:])
    ordinary = MakerEnvironment(scenario(1), cfg)
    assert not ordinary.observation().action_mask[4]  # both improvements cross in a two-tick spread
    ordinary.step(0)
    assert ordinary.inventory == cfg.initial_inventory and ordinary.cash == cfg.initial_cash


def test_cash_asset_endowment_and_positive_price_walk_bound_are_explicit() -> None:
    cfg = env_config()
    with pytest.raises(ValueError, match="endowment"):
        MakerEnvironment(scenario(1), replace(cfg, initial_inventory=0, initial_cash=0))
    with pytest.raises(ValueError, match="price/tick"):
        MakerEnvironment(replace(scenario(1), book=replace(scenario(1).book, tick=1.0)), cfg)


def test_smart_quotes_retain_true_fifo_position_and_cancel_on_change() -> None:
    env = MakerEnvironment(scenario(1), env_config())
    observation = env.observation()
    level = env.book.best_bid_level
    assert level is not None
    env._post("buy", level, observation)
    resting = env._orders["buy"]
    assert resting is not None and env.book.queue_position(resting[0]) == 2
    env._post("buy", level, observation)
    assert env._orders["buy"] == resting and env.book.queue_position(resting[0]) == 2
    env._post("buy", level - 1, observation)
    moved = env._orders["buy"]
    assert moved is not None and moved[0] != resting[0] and not env.book.order_alive(resting[0])


def test_naive_actual_fifo_episode_financing_fees_conservation_and_independent_ledger() -> None:
    cfg = env_config()
    result = run_maker_episode(scenario(19), cfg)
    cash, inventory = cfg.initial_cash, cfg.initial_inventory
    fees = spread = 0.0
    assert result["complete"] and result["fills"] > 0
    for transition in result["transitions"]:
        before = transition["observation"]
        assert before["observed_time"] < transition["next_observation"]["observed_time"]
        assert before["action_mask"][transition["action"]]
        for fill in transition["fills"]:
            quantity = 1 if fill["maker_side"] == "buy" else -1
            inventory += quantity
            cash -= quantity * fill["price"] + fill["fee"]
            fees += fill["price"] * cfg.maker_fee_bps / 10000
            spread += quantity * (fill["decision_mid"] - fill["price"])
            assert fill["price"] == fill["limit_price"] and fill["t"] > fill["decision_time"]
            assert fill["qty"] == 1 and fill["maker_queue_ahead_at_submit"] >= 0
        assert transition["next_observation"]["cash"] == pytest.approx(cash)
        assert transition["next_observation"]["inventory"] == inventory
        assert 0 <= inventory <= cfg.inventory_cap and cash >= 0
        penalty = (
            cfg.inventory_penalty * ((inventory - cfg.target_inventory) / cfg.inventory_cap) ** 2
        )
        assert transition["sim_internal_inventory_penalty"] == pytest.approx(penalty)
    assert result["fees_paid"] == pytest.approx(fees)
    assert result["sim_internal_spread_capture"] == pytest.approx(spread)
    assert result["terminal_inventory"] == inventory and not result["terminal_inventory_liquidated"]
    counts = result["final_book_conservation"]
    assert (
        counts["n_orders_created"]
        == counts["n_fills"] + counts["n_cancellations"] + counts["resting"]
    )
    assert (
        result["synthetic"]
        and not result["market_evidence"]
        and not result["proper_forecast_score"]
    )
    assert replay_maker_episode(result)["ledger_replayed"]


@requires_torch
def test_actual_minibatch_actor_and_critic_learning_and_all_round_outcomes_retained(
    fitted: FrozenMakerPolicy,
) -> None:
    metadata = fitted.metadata()
    assert fitted.parameter_sha256 != metadata["initial_parameter_sha256"]
    assert metadata["updates"] == 16 and metadata["training_transitions"] == 64
    assert len(metadata["loss_history"]) == 16 and len(metadata["training_rollouts"]) == 4
    assert all(e["complete"] for r in metadata["rounds"] for e in r["episodes"])
    initial = {p[0]: p[2] for p in metadata["round_policy_parameters"][0]}
    final = {p[0]: p[2] for p in fitted.parameters}
    assert initial["actor"] != list(final["actor"]) and initial["critic"] != list(final["critic"])
    for rollout in metadata["training_rollouts"]:
        assert replay_maker_episode(rollout)["ledger_replayed"]
    obs = MakerEnvironment(scenario(41), env_config()).observation()
    probabilities, value = fitted.distribution(obs)
    assert probabilities.sum() == pytest.approx(1.0) and math.isfinite(value)
    assert all(
        prob == 0 for prob, mask in zip(probabilities, obs.action_mask, strict=True) if not mask
    )


@requires_torch
def test_heldout_seed_regime_comparison_retains_every_ppo_and_naive_outcome(
    fitted: FrozenMakerPolicy,
) -> None:
    result = evaluate_ppo_maker(
        fitted,
        (scenario(41, "buy_bias"), scenario(43, "switching"), scenario(47, "sell_bias")),
        env_config(),
    )
    assert len(result["outcomes"]) == 3
    for outcome in result["outcomes"]:
        assert outcome["ppo"]["complete"] and outcome["naive"]["complete"]
        assert replay_maker_episode(outcome["ppo"], policy=fitted)["policy_replayed"]
        assert replay_maker_episode(outcome["naive"])["ledger_replayed"]
    comparison = result["comparison"]
    assert comparison["heldout_regimes"] == ["buy_bias", "sell_bias", "switching"]
    assert comparison["environment_constraints_matched"] and comparison["heldout_seeds"]
    assert not comparison["compute_matched"] and not comparison["identical_realized_tapes"]
    assert not comparison["benefit_asserted"] and comparison["c51_status"].startswith("UNAVAILABLE")
    with pytest.raises(ValueError, match="overlap"):
        evaluate_ppo_maker(fitted, (scenario(3, "switching"),), env_config())


@requires_torch
def test_local_generators_and_cpu_device_preserve_global_rng_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    threads = torch.get_num_threads()

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("global RNG/thread mutation")

    monkeypatch.setattr(torch, "manual_seed", forbidden)
    monkeypatch.setattr(torch, "set_num_threads", forbidden)
    with torch.device("meta"):
        policy = train_ppo_maker(
            (scenario(2),), env_config(), replace(ppo_config(), rounds=1, epochs=1)
        )
    assert policy.metadata()["device"] == "cpu"
    assert torch.equal(state, torch.get_rng_state()) and torch.get_num_threads() == threads
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])


@requires_torch
def test_frozen_json_roundtrip_without_torch_and_exact_action_ledger_replay(
    fitted: FrozenMakerPolicy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "ppo.json"
    fitted.save(path)

    def unavailable() -> None:
        raise AssertionError("frozen inference/persistence must not require torch")

    import quant_fund.models.policy_gradient_maker as module

    monkeypatch.setattr(module, "_torch", unavailable)
    restored = FrozenMakerPolicy.load(path)
    assert (
        restored.policy_sha256 == fitted.policy_sha256 and restored.parameters == fitted.parameters
    )
    expected = run_maker_episode(scenario(41), env_config(), policy=fitted)
    assert run_maker_episode(scenario(41), env_config(), policy=restored) == expected
    with pytest.raises(FileExistsError):
        fitted.save(path)
    payload = json.loads(path.read_text())
    payload["parameters"][0][2][0] += 0.1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash"):
        FrozenMakerPolicy.load(path)


@requires_torch
def test_rehashed_artifacts_reject_false_honesty_and_training_ledger(
    fitted: FrozenMakerPolicy, tmp_path: Path
) -> None:
    path = tmp_path / "tampered.json"
    fitted.save(path)
    payload = json.loads(path.read_text())
    payload["metadata"]["training_rollouts"][0]["sim_internal_terminal_cash"] += 1.0
    rollout = payload["metadata"]["training_rollouts"][0]
    raw = dict(rollout)
    raw.pop("episode_sha256")
    rollout["episode_sha256"] = hashlib.sha256(
        json.dumps(raw, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    forged = FrozenMakerPolicy(
        payload["hidden"],
        tuple((p[0], tuple(p[1]), tuple(p[2])) for p in payload["parameters"]),
        json.dumps(payload["metadata"], sort_keys=True),
    )
    payload["policy_sha256"] = forged.policy_sha256
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="replay"):
        FrozenMakerPolicy.load(path)
    metadata = fitted.metadata()
    metadata["synthetic"] = False
    with pytest.raises(ValueError, match="honesty"):
        FrozenMakerPolicy(fitted.hidden, fitted.parameters, json.dumps(metadata))


@requires_torch
def test_resource_budget_and_seed_duplication_reject_before_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_fund.models.policy_gradient_maker as module

    def unavailable() -> None:
        raise AssertionError("invalid work must reject before optional torch")

    monkeypatch.setattr(module, "_torch", unavailable)
    with pytest.raises(ValueError, match="resource"):
        train_ppo_maker(
            (scenario(1), scenario(2)), replace(env_config(), decisions=256), PPOConfig(rounds=64)
        )
    with pytest.raises(ValueError, match="distinct"):
        train_ppo_maker((scenario(1), scenario(1)), env_config(), ppo_config())


@requires_torch
def test_current_observation_and_policy_do_not_read_hidden_regime_or_future_events(
    fitted: FrozenMakerPolicy, monkeypatch: pytest.MonkeyPatch
) -> None:
    stationary = MakerEnvironment(scenario(61), env_config())
    future_switching = MakerEnvironment(scenario(61, "switching"), env_config())

    def forbidden() -> None:
        raise AssertionError("decision may not advance to a future event")

    monkeypatch.setattr(future_switching.book, "step", forbidden)
    first, second = stationary.observation(), future_switching.observation()
    assert first == second
    assert fitted.distribution(first)[0] == pytest.approx(fitted.distribution(second)[0])
    assert fitted.action(first) == fitted.action(second)


@requires_torch
@pytest.mark.parametrize("case", ["method", "counts", "constraints", "hidden", "scenario_binding"])
def test_rehashed_policy_cannot_contradict_method_budget_or_rollout_bindings(
    fitted: FrozenMakerPolicy, tmp_path: Path, case: str
) -> None:
    path = tmp_path / "rehashed.json"
    fitted.save(path)
    payload = json.loads(path.read_text())
    metadata = payload["metadata"]
    if case == "method":
        metadata["algorithm"] = "C51_RELABELED"
    elif case == "counts":
        metadata["updates"] += 1
    elif case == "constraints":
        metadata["environment"]["initial_cash"] += 1000
    elif case == "hidden":
        metadata["config"]["hidden"] += 1
    elif case == "scenario_binding":
        metadata["training_scenarios"][0]["book"]["seed"] = 901
    forged = FrozenMakerPolicy(
        fitted.hidden, fitted.parameters, json.dumps(metadata, sort_keys=True)
    )
    payload["policy_sha256"] = forged.policy_sha256
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        FrozenMakerPolicy.load(path)


@requires_torch
def test_frozen_artifact_resource_bound_and_metadata_copy_preserve_original(
    fitted: FrozenMakerPolicy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quant_fund.models.policy_gradient_maker as module

    detached = fitted.metadata()
    detached["environment"]["inventory_cap"] = 500
    assert fitted.metadata()["environment"]["inventory_cap"] == env_config().inventory_cap
    monkeypatch.setattr(module, "_MAX_BYTES", 10)
    path = tmp_path / "bounded.json"
    with pytest.raises(ValueError, match="resource"):
        fitted.save(path)
    assert not path.exists()
    path.write_bytes(b" " * 11)
    with pytest.raises(ValueError, match="resource"):
        FrozenMakerPolicy.load(path)


def test_episode_terminal_state_and_replay_length_are_fail_closed() -> None:
    cfg = env_config()
    env = MakerEnvironment(scenario(2), cfg)
    for _ in range(cfg.decisions):
        env.step(0)
    assert env.done
    with pytest.raises(RuntimeError, match="complete"):
        env.step(0)
    result = run_maker_episode(scenario(2), cfg)
    result["transitions"] = result["transitions"] * 100
    with pytest.raises(ValueError, match="resource"):
        replay_maker_episode(result)
