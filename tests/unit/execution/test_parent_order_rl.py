"""Synthetic correctness only: no market-quality inference from these tests."""

from __future__ import annotations

import builtins
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quant_fund.execution.parent_order_rl import (
    AlmgrenChrissPolicy,
    BookSnapshot,
    DQNConfig,
    ExecutionScenario,
    ParentOrder,
    ParentOrderDQN,
    ParentOrderEnvironment,
    RealizedLiquidity,
    SimulatorConfig,
    SyntheticExecutionTape,
    TradeAction,
    TWAPPolicy,
    evaluate_agent,
    execution_analytics,
    grid_action,
    simulate_episode,
    verify_episode,
    verify_execution_receipt,
    write_execution_receipt,
)


def scenario(
    day: int,
    *,
    rising_cost: bool = True,
    side: str = "buy",
    quantity: float = 40,
    market_depth: float = 80,
    auction_depth: float = 80,
    volume: float = 400,
    cash: float = 10_000,
    participation: float = 0.5,
    fees: float = 0,
    impact: float = 0,
) -> ExecutionScenario:
    """Public volume forecast is a planted noisy-regime cue, not future clearing.

    Higher forecast volume identifies a regime with adverse later prices in
    this contrived generator. It demonstrates representation learning only.
    Future queue allocations and realized prices are held in separate types.
    """
    start = datetime(2024, 1, 1, 15, 58, tzinfo=UTC) + timedelta(days=day)
    rng = np.random.default_rng(day + 700)
    sign = 1 if side == "buy" else -1
    market_mid = 100 + float(rng.normal(0, 0.005))
    clearing = 100 + sign * (2 if rising_cost else -2) + float(rng.normal(0, 0.03))
    e1_time, closing = start + timedelta(minutes=1), start + timedelta(minutes=2)
    e1 = RealizedLiquidity(
        e1_time,
        e1_time + timedelta(seconds=1),
        market_mid - 0.05,
        market_mid + 0.05,
        market_depth,
        market_depth,
        volume,
    )
    e2 = RealizedLiquidity(
        closing,
        closing + timedelta(seconds=1),
        clearing - 0.05,
        clearing + 0.05,
        80,
        80,
        volume,
        auction_price=clearing,
        auction_quantity=auction_depth,
    )
    cue = 400 if rising_cost else 20
    observations = (
        BookSnapshot(start, start + timedelta(seconds=1), 99.95, 100.05, 80, 80, cue),
        BookSnapshot(
            e1.event_time, e1.available_time, e1.bid, e1.ask, e1.bid_quantity, e1.ask_quantity, cue
        ),
        BookSnapshot(
            e2.event_time, e2.available_time, e2.bid, e2.ask, e2.bid_quantity, e2.ask_quantity, 0
        ),
    )
    tape = SyntheticExecutionTape(
        f"synthetic-{day}", day + 700, observations, (e1, e2), e1_time + timedelta(seconds=30)
    )
    order = ParentOrder(quantity, side, 100, cash, participation, closing, 8.0)  # type: ignore[arg-type]
    return ExecutionScenario(tape, order, SimulatorConfig(fees, impact))


@pytest.fixture(scope="module")
def fitted() -> ParentOrderDQN:
    pytest.importorskip("torch")
    train = tuple(
        scenario(i, rising_cost=i % 2 == 0, side="buy" if i % 4 < 2 else "sell") for i in range(16)
    )
    return ParentOrderDQN(DQNConfig(rounds=100, seed=7)).fit(
        train,
        fit_as_of=datetime(2024, 1, 20, tzinfo=UTC),
    )


def test_actual_dqn_learns_observed_context_and_changes_parameters(fitted: ParentOrderDQN) -> None:
    trace = fitted.training_trace
    assert trace is not None
    assert trace.initial_parameters_sha256 != trace.final_parameters_sha256
    assert trace.transitions == 3200
    assert trace.optimizer_steps > 3000 and trace.target_copies > 70
    assert len(trace.td_losses) == trace.optimizer_steps
    assert np.isfinite(trace.td_losses).all()
    assert np.mean(trace.episode_objectives[-160:]) < np.mean(trace.episode_objectives[:160])
    choices = []
    for i in range(40, 48):
        env = ParentOrderEnvironment(
            scenario(i, rising_cost=i % 2 == 0, side="buy" if i % 4 < 2 else "sell")
        )
        state = env.state()
        choices.append(
            int(np.argmax(np.where(state.action_mask(), fitted.q_values(state), -np.inf)))
        )
    assert choices == [2, 0, 2, 0, 2, 0, 2, 0]


def test_trained_policy_and_baselines_use_matched_heldout_episodes(fitted: ParentOrderDQN) -> None:
    heldout = tuple(
        scenario(i, rising_cost=i % 2 == 0, side="buy" if i % 4 < 2 else "sell")
        for i in range(40, 48)
    )
    comparison = evaluate_agent(fitted, heldout)
    assert len(comparison.episodes) == 3 * len(heldout)
    for s in heldout:
        rows = [e for e in comparison.episodes if e.episode_id == s.tape.episode_id]
        assert len({e.scenario_sha256 for e in rows}) == 1
        assert {e.policy_id for e in rows} == {
            fitted.policy_id,
            "twap_catchup",
            "almgren_chriss_catchup",
        }
        for row in rows:
            verify_episode(s, row)
            assert row.analytics.completion_fraction == pytest.approx(1)
    report = comparison.receipt_payload()
    assert report["synthetic"] and not report["promote"] and not report["sota_claim"]
    scores = report["diagnostics"]
    # A deliberately planted simulator mechanism, not execution outperformance.
    assert (
        scores[fitted.policy_id]["sim_internal_objective_mean"]
        < scores["twap_catchup"]["sim_internal_objective_mean"] - 20
    )


def test_state_contains_no_next_realization_and_is_suffix_invariant() -> None:
    first = scenario(40)
    future = replace(first.tape.events[-1], auction_price=150, auction_quantity=1)
    changed = replace(first, tape=replace(first.tape, events=(first.tape.events[0], future)))
    a, b = ParentOrderEnvironment(first), ParentOrderEnvironment(changed)
    assert a.state() == b.state()
    np.testing.assert_array_equal(a.state().vector(), b.state().vector())
    a.step(grid_action(a.state(), 0))
    b.step(grid_action(b.state(), 0))
    np.testing.assert_array_equal(a.state().vector(), b.state().vector())
    fa, fb = a.step(grid_action(a.state(), 4)).fill, b.step(grid_action(b.state(), 4)).fill
    assert fa is not None and fb is not None
    assert fa.price != fb.price and fa.quantity != fb.quantity
    assert not hasattr(a.state(), "events")


def test_next_event_only_fill_and_delayed_publication() -> None:
    s = scenario(40)
    env = ParentOrderEnvironment(s)
    current = env.state()
    assert current.remaining_seconds == 119
    step = env.step(grid_action(current, 2))
    assert step.fill is not None
    assert step.fill.decision_time == current.book.available_time
    assert step.fill.event_time == s.tape.events[0].event_time
    assert step.fill.available_time == s.tape.events[0].available_time
    assert current.book.available_time < step.fill.event_time < step.state.book.available_time
    assert step.state.remaining_seconds == 59
    assert step.state.cash_remaining == pytest.approx(s.order.cash_budget - step.fill.price * 40)
    with pytest.raises(ValueError, match="currently published"):
        env.step(TradeAction(s.tape.events[1].available_time, "wait", 0))


def test_quantity_participation_cash_fees_and_impact_bound_fill() -> None:
    s = scenario(40, market_depth=19, volume=40, participation=0.25, cash=800, fees=0.2, impact=0.5)
    env = ParentOrderEnvironment(s)
    step = env.step(grid_action(env.state(), 2))
    f = step.fill
    assert f is not None
    assert f.quantity < 10 and f.quantity < 19
    assert f.price == pytest.approx(s.tape.events[0].ask + 0.5 * f.quantity)
    assert f.quantity * f.price + f.fees == pytest.approx(800)
    assert step.state.cash_remaining == pytest.approx(0, abs=1e-9)
    env.step(grid_action(env.state(), 4))
    row = env.result("manual")
    verify_episode(s, row)
    assert row.analytics.unfilled_quantity > 30
    assert row.analytics.sim_internal_filled_shortfall == pytest.approx(
        (f.price - 100) * f.quantity
    )
    assert row.analytics.sim_internal_fees == pytest.approx(0.2 * f.quantity)
    assert row.analytics.sim_internal_terminal_preference == pytest.approx(
        8 * row.analytics.unfilled_quantity
    )


def test_sell_ledger_has_proceeds_and_correct_cost_sign() -> None:
    s = scenario(42, side="sell", cash=0, fees=0.1, impact=0.01)
    row = simulate_episode(TWAPPolicy(), s)
    verify_episode(s, row)
    proceeds = sum(f.price * f.quantity - f.fees for f in row.fills)
    shortfall = sum((100 - f.price) * f.quantity for f in row.fills)
    assert row.analytics.cash_remaining == pytest.approx(proceeds)
    assert row.analytics.sim_internal_filled_shortfall == pytest.approx(shortfall)


def test_auction_allocations_can_leave_inventory_and_penalty_is_separate() -> None:
    s = scenario(40, auction_depth=7, volume=10, participation=0.5)
    env = ParentOrderEnvironment(s)
    env.step(grid_action(env.state(), 0))
    step = env.step(grid_action(env.state(), 4))
    assert step.fill is not None and step.fill.quantity == 5
    assert step.done
    row = env.result("wait_auction")
    verify_episode(s, row)
    assert row.analytics.completion_fraction == pytest.approx(5 / 40)
    assert row.analytics.unfilled_quantity == 35
    assert row.analytics.sim_internal_terminal_preference == 280
    assert row.analytics.cash_remaining == pytest.approx(10_000 - 5 * step.fill.price)
    assert row.analytics.auction_filled_quantity == 5


def test_missed_auction_deadline_masks_and_rejects_submission() -> None:
    s = scenario(40)
    s = replace(
        s, tape=replace(s.tape, auction_submit_deadline=s.tape.observations[0].available_time)
    )
    env = ParentOrderEnvironment(s)
    env.step(grid_action(env.state(), 0))
    assert not env.state().auction_eligible
    assert env.state().action_mask().tolist() == [True, False, False, False, False]
    with pytest.raises(ValueError, match="submission deadline"):
        env.step(TradeAction(env.state().book.available_time, "auction", 40))
    env.step(grid_action(env.state(), 0))
    assert env.result("missed").analytics.completion_fraction == 0


def test_auction_only_at_closing_and_no_market_crossing_in_closing() -> None:
    env = ParentOrderEnvironment(scenario(40))
    with pytest.raises(ValueError, match="unavailable"):
        env.step(TradeAction(env.state().book.available_time, "auction", 40))
    env.step(grid_action(env.state(), 0))
    with pytest.raises(ValueError, match="only auction"):
        env.step(TradeAction(env.state().book.available_time, "market", 40))


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_one_event_limit_respects_its_submitted_price(side: str) -> None:
    s = scenario(40, side=side)
    event = replace(s.tape.events[0], passive_buy_quantity=8, passive_sell_quantity=9)
    s = replace(s, tape=replace(s.tape, events=(event, s.tape.events[1])))
    env = ParentOrderEnvironment(s)
    limit = event.bid - 0.01 if side == "buy" else event.ask + 0.01
    assert env.step(TradeAction(env.state().book.available_time, "limit", 20, limit)).fill is None
    env = ParentOrderEnvironment(s)
    limit = event.bid if side == "buy" else event.ask
    step = env.step(TradeAction(env.state().book.available_time, "limit", 20, limit))
    assert step.fill is not None
    assert step.fill.quantity == (8 if side == "buy" else 9)
    assert step.fill.price == limit
    assert step.state.remaining_inventory == 40 - step.fill.quantity


def test_no_invented_terminal_fill_when_no_liquidity() -> None:
    s = scenario(40, market_depth=0, auction_depth=0)
    row = simulate_episode(TWAPPolicy(), s)
    assert not row.fills
    assert row.remaining_path == (40, 40, 40)
    assert row.analytics.sim_internal_filled_shortfall == 0
    assert row.analytics.sim_internal_terminal_preference == 320
    verify_episode(s, row)


def test_reward_sum_matches_separate_cost_and_inventory_preferences() -> None:
    s = scenario(40)
    s = replace(s, order=replace(s.order, inventory_penalty_per_step=3))
    env = ParentOrderEnvironment(s)
    rewards = [env.step(grid_action(env.state(), 1)).reward]
    rewards.append(env.step(grid_action(env.state(), 0)).reward)
    row = env.result("small_then_wait")
    assert row.analytics.sim_internal_inventory_preference == pytest.approx(2 * 3 * 0.75**2)
    assert -sum(rewards) * 40 == pytest.approx(row.analytics.sim_internal_objective)
    verify_episode(s, row)


def test_zero_risk_ac_schedule_matches_twap_under_shared_caps() -> None:
    s = scenario(40, market_depth=6, auction_depth=40)
    twap = simulate_episode(TWAPPolicy(), s)
    ac = simulate_episode(AlmgrenChrissPolicy(risk_aversion=0), s)
    assert twap.fills == ac.fills
    assert twap.actions == ac.actions
    assert twap.analytics == ac.analytics


def test_partial_terminal_and_overfill_actions_are_rejected() -> None:
    env = ParentOrderEnvironment(scenario(40))
    with pytest.raises(ValueError, match="partial"):
        env.result("manual")
    with pytest.raises(ValueError, match="remaining inventory"):
        env.step(TradeAction(env.state().book.available_time, "market", 41))
    env.step(grid_action(env.state(), 2))
    assert env.state().action_mask().tolist() == [True, False, False, False, False]
    env.step(grid_action(env.state(), 0))
    with pytest.raises(ValueError, match="terminal"):
        env.step(TradeAction(env.state().book.available_time, "wait", 0))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_nonfinite_or_negative_book_and_order_inputs_fail(value: float) -> None:
    s = scenario(40)
    with pytest.raises(ValueError):
        replace(s.tape.observations[0], bid_quantity=value)
    with pytest.raises(ValueError):
        replace(s.tape.events[0], traded_volume=value)
    with pytest.raises(ValueError):
        replace(s.order, cash_budget=value)


def test_tape_clocks_alignment_and_auction_validation_fail_closed() -> None:
    s = scenario(40)
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(s.tape.observations[0], event_time=datetime(2024, 1, 1))
    with pytest.raises(ValueError, match="publication"):
        replace(
            s.tape.observations[0],
            available_time=s.tape.observations[0].event_time - timedelta(seconds=1),
        )
    with pytest.raises(ValueError, match="strictly after"):
        replace(
            s.tape,
            observations=(
                replace(s.tape.observations[0], available_time=s.tape.events[0].event_time),
                *s.tape.observations[1:],
            ),
        )
    with pytest.raises(ValueError, match="align"):
        replace(
            s.tape,
            observations=(
                s.tape.observations[0],
                replace(s.tape.observations[1], ask=101),
                s.tape.observations[2],
            ),
        )
    with pytest.raises(ValueError, match="exactly the final"):
        replace(s.tape, events=(replace(s.tape.events[0], auction_price=100), s.tape.events[1]))
    with pytest.raises(ValueError, match="deadline"):
        replace(s, order=replace(s.order, deadline=s.order.deadline + timedelta(seconds=1)))
    with pytest.raises(ValueError, match="nonpositive"):
        scenario(40, side="sell", impact=3)
    with pytest.raises(ValueError, match="one initial"):
        replace(s.tape, observations=s.tape.observations[:2])


def test_independent_verifier_rejects_corrupt_cost_clock_capacity_and_identity() -> None:
    s = scenario(40, fees=0.1)
    result = simulate_episode(TWAPPolicy(), s)
    first = result.fills[0]
    corruptions = (
        replace(result, scenario_sha256="0" * 64),
        replace(result, analytics=replace(result.analytics, completion_fraction=1.1)),
        replace(result, fills=(replace(first, fees=0), *result.fills[1:])),
        replace(result, fills=(replace(first, event_time=first.decision_time), *result.fills[1:])),
        replace(result, fills=(replace(first, participation_capacity=0), *result.fills[1:])),
        replace(result, fills=(replace(first, price=first.price + 1), *result.fills[1:])),
        replace(result, remaining_path=(41, *result.remaining_path[1:])),
        replace(result, fills=(first, first, *result.fills[1:])),
    )
    for corrupted in corruptions:
        with pytest.raises(ValueError):
            verify_episode(s, corrupted)
    with pytest.raises(ValueError, match="quantity"):
        execution_analytics(s.order, result.fills, 1)


def test_train_eval_identity_and_clock_separation(fitted: ParentOrderDQN) -> None:
    with pytest.raises(ValueError, match="overlap"):
        evaluate_agent(fitted, [scenario(0)])
    renamed = replace(scenario(0), tape=replace(scenario(0).tape, episode_id="renamed", seed=900))
    with pytest.raises(ValueError, match="overlap"):
        evaluate_agent(fitted, [renamed])
    base = scenario(0)
    delta = timedelta(days=40)
    tape = replace(
        base.tape,
        episode_id="redated",
        seed=900,
        observations=tuple(
            replace(o, event_time=o.event_time + delta, available_time=o.available_time + delta)
            for o in base.tape.observations
        ),
        events=tuple(
            replace(e, event_time=e.event_time + delta, available_time=e.available_time + delta)
            for e in base.tape.events
        ),
        auction_submit_deadline=base.tape.auction_submit_deadline + delta,
    )
    redated = replace(
        base, tape=tape, order=replace(base.order, deadline=base.order.deadline + delta)
    )
    assert redated.tape.path_sha256 == base.tape.path_sha256
    assert redated.sha256 != base.sha256
    with pytest.raises(ValueError, match="overlap"):
        evaluate_agent(fitted, [redated])
    with pytest.raises(ValueError, match="fit cutoff"):
        fitted.q_values(ParentOrderEnvironment(scenario(0)).state())
    with pytest.raises(ValueError, match="strictly after"):
        evaluate_agent(fitted, [scenario(18)])
    with pytest.raises(ValueError, match="unique"):
        evaluate_agent(fitted, [scenario(40), scenario(40)])
    with pytest.raises(ValueError, match="chronological"):
        evaluate_agent(fitted, [scenario(41), scenario(40)])
    with pytest.raises(ValueError, match="fully published"):
        ParentOrderDQN().fit([scenario(0)], fit_as_of=scenario(0).tape.events[-1].event_time)
    with pytest.raises(ValueError, match="unique"):
        ParentOrderDQN().fit(
            [scenario(0), scenario(0)], fit_as_of=datetime(2024, 1, 20, tzinfo=UTC)
        )
    with pytest.raises(ValueError, match="chronological"):
        ParentOrderDQN().fit(
            [scenario(1), scenario(0)], fit_as_of=datetime(2024, 1, 20, tzinfo=UTC)
        )


def test_actual_fit_required_and_no_optimizer_budget_is_not_a_success() -> None:
    policy = ParentOrderDQN()
    with pytest.raises(ValueError, match="actually fitted"):
        policy.act(ParentOrderEnvironment(scenario(40)).state())
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="no optimizer"):
        ParentOrderDQN(DQNConfig(rounds=1, batch_size=32)).fit(
            [scenario(0)], fit_as_of=datetime(2024, 1, 2, tzinfo=UTC)
        )


def test_optional_torch_missing_and_exported_inference_needs_no_torch(
    monkeypatch: pytest.MonkeyPatch,
    fitted: ParentOrderDQN,
) -> None:
    original = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch" or name.startswith("torch."):
            raise ImportError("blocked optional dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert fitted.act(ParentOrderEnvironment(scenario(40)).state()).kind == "market"
    with pytest.raises(ImportError, match="optional torch"):
        ParentOrderDQN().fit([scenario(0)], fit_as_of=datetime(2024, 1, 2, tzinfo=UTC))


def test_write_once_model_and_comparison_receipts_bind_exact_artifacts(
    tmp_path: Path,
    fitted: ParentOrderDQN,
) -> None:
    model_path = tmp_path / "model.json"
    digest = fitted.save_artifact(model_path)
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == digest
    artifact = json.loads(model_path.read_bytes())
    blob = json.dumps(
        artifact["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    assert hashlib.sha256(blob).hexdigest() == artifact["model_sha256"] == fitted.model_sha256
    assert artifact["payload"]["training_path_sha256"]
    assert artifact["payload"]["trace"]["optimizer_steps"] > 0
    assert len(artifact["payload"]["weights"]) == 4
    loaded = ParentOrderDQN.load_artifact(model_path)
    state = ParentOrderEnvironment(scenario(40)).state()
    np.testing.assert_array_equal(loaded.q_values(state), fitted.q_values(state))
    with pytest.raises(FileExistsError):
        fitted.save_artifact(model_path)
    comparison = evaluate_agent(fitted, [scenario(40), scenario(41, rising_cost=False)])
    path = tmp_path / "receipt.json"
    digest = write_execution_receipt(path, comparison)
    receipt = json.loads(path.read_bytes())
    blob = json.dumps(
        receipt["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    assert receipt["receipt_sha256"] == hashlib.sha256(blob).hexdigest() == digest
    assert receipt["payload"]["model_sha256"] == fitted.model_sha256
    assert not set(receipt["payload"]["training_path_sha256"]) & set(
        receipt["payload"]["evaluation_path_sha256"]
    )
    with pytest.raises(FileExistsError):
        write_execution_receipt(path, comparison)
    corrupt = replace(
        comparison.episodes[0],
        analytics=replace(comparison.episodes[0].analytics, sim_internal_objective=0),
    )
    with pytest.raises(ValueError, match="independent ledger"):
        write_execution_receipt(
            tmp_path / "corrupt.json",
            replace(comparison, episodes=(corrupt, *comparison.episodes[1:])),
        )
    assert not (tmp_path / "corrupt.json").exists()
    verified = verify_execution_receipt(path)
    assert verified["receipt_sha256"] == digest
    receipt["payload"]["diagnostics"][fitted.policy_id]["completion_fraction_mean"] = 0
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            receipt["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    fabricated = tmp_path / "fabricated.json"
    fabricated.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="claims disagree"):
        verify_execution_receipt(fabricated)
    with pytest.raises(ValueError, match="one matched"):
        replace(comparison, episodes=comparison.episodes[:-1]).receipt_payload()
    with pytest.raises(ValueError, match="declared model/baseline"):
        replace(
            comparison, baseline_config=(("sigma", 0.1), ("eta", 0.01), ("risk_aversion", 0))
        ).receipt_payload()
    assert receipt["payload"]["paired_episode_differences"]["episode_count"] == 2
    assert receipt["payload"]["diagnostics"][fitted.policy_id]["decision_latency_p95_ms"] >= 0
    artifact["payload"]["weights"][0][0][0] += 1
    corrupt_model = tmp_path / "corrupt-model.json"
    corrupt_model.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="content hash"):
        ParentOrderDQN.load_artifact(corrupt_model)


def test_model_parameter_or_provenance_mutation_rejected(fitted: ParentOrderDQN) -> None:
    import copy

    corrupt = copy.deepcopy(fitted)
    assert corrupt._weights is not None
    corrupt._weights[0].setflags(write=True)
    corrupt._weights[0][0, 0] += 10
    with pytest.raises(ValueError, match="changed"):
        corrupt.act(ParentOrderEnvironment(scenario(40)).state())
    corrupt = copy.deepcopy(fitted)
    assert corrupt._artifact is not None
    corrupt._artifact["training_path_sha256"] = []
    with pytest.raises(ValueError, match="changed"):
        _ = corrupt.model_sha256


def test_malformed_state_and_action_grid_are_rejected() -> None:
    state = ParentOrderEnvironment(scenario(40)).state()
    for values in (
        {"time_index": -1},
        {"horizon": 0},
        {"remaining_inventory": 41},
        {"cash_remaining": float("nan")},
        {"next_is_auction": True},
        {"auction_eligible": True},
    ):
        with pytest.raises(ValueError):
            replace(state, **values)
    for action in (True, -1, 5):
        with pytest.raises(ValueError, match="outside"):
            grid_action(state, action)
    with pytest.raises(ValueError, match="unavailable"):
        grid_action(state, 4)


def test_sell_net_fee_cash_is_constrained_and_arrival_reference_is_observed() -> None:
    s = scenario(40, side="sell", cash=10, fees=101)
    env = ParentOrderEnvironment(s)
    step = env.step(grid_action(env.state(), 2))
    assert step.fill is not None and step.fill.quantity < 10
    assert step.state.cash_remaining == pytest.approx(0, abs=1e-8)
    env.step(grid_action(env.state(), 4))
    result = env.result("net_fee")
    verify_episode(s, result)
    assert result.analytics.cash_remaining == pytest.approx(0, abs=1e-8)
    with pytest.raises(ValueError, match="initial published midpoint"):
        replace(s, order=replace(s.order, arrival_price=110))


def test_training_resource_budget_and_empty_splits_are_rejected() -> None:
    with pytest.raises(ValueError, match="training requires"):
        ParentOrderDQN().fit([], fit_as_of=datetime(2024, 1, 20, tzinfo=UTC))
    training = [scenario(i) for i in range(51)]
    with pytest.raises(ValueError, match="one-million"):
        ParentOrderDQN(DQNConfig(rounds=10000)).fit(
            training, fit_as_of=datetime(2024, 3, 1, tzinfo=UTC)
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rounds": 0},
        {"rounds": True},
        {"hidden_dim": 129},
        {"gamma": 1.1},
        {"epsilon_end": 1.1},
        {"batch_size": 33, "replay_capacity": 32},
        {"seed": -1},
        {"learning_rate": float("nan")},
    ],
)
def test_dqn_resource_and_parameter_failures(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        DQNConfig(**kwargs)
