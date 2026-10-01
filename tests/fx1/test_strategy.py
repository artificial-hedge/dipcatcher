"""SYNTHETIC correctness and adversarial/PIT checks for strategy generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from fx1.strategy import (
    GeneratedCode,
    ReplayBar,
    StrategySpec,
    backtest,
    evaluate,
    parse_strategy,
    run_strategy,
    text_to_code,
    verify_strategy_receipt,
    write_receipt,
)

CODE = """def strategy(history):
    return {'probability_up': 0.7 if mean(history, 2) > mean(history, 3) else 0.3,
            'target_weight': 1 if mean(history, 2) > mean(history, 3) else -1}
"""


def spec() -> StrategySpec:
    return StrategySpec("sma", "SMA", "Compare causal two- and three-close moving means.", {})


class Backend:
    def __init__(self, code: str = CODE) -> None:
        self.code = code
        self.messages = []

    def complete(self, messages):
        self.messages.append(messages)
        return self.code


def generated(code: str = CODE) -> GeneratedCode:
    return text_to_code(spec(), Backend(code), backend_id="SYNTHETIC_test_backend")


def bars(prices=(100, 101, 102, 101, 103, 102)) -> list[ReplayBar]:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    return [
        ReplayBar(start + timedelta(days=i), start + timedelta(days=i), float(p))
        for i, p in enumerate(prices)
    ]


def replay(code: GeneratedCode | None = None):
    tape = bars()
    return backtest(
        code or generated(),
        tape,
        decision_times=[row.event_time for row in tape[2:-1]],
        data_source="SYNTHETIC_test_tape",
        synthetic=True,
        cost_bps=2,
    )


def test_generation_calls_actual_supplied_backend_and_binds_spec() -> None:
    model = Backend()
    result = text_to_code(spec(), model, backend_id="injected-model")
    assert len(model.messages) == 1
    assert json.loads(model.messages[0][1]["content"]) == spec().payload()
    assert result.spec_sha256 == spec().sha256
    assert result.code_sha256 == hashlib.sha256(CODE.encode()).hexdigest()
    assert result.code_text == CODE
    assert result.generation_seconds >= 0


@pytest.mark.parametrize(
    "code",
    [
        "import os\ndef strategy(history): return {'probability_up':0.5,'target_weight':0}",
        "def strategy(history): return {'probability_up': history.__class__, 'target_weight': 0}",
        "def strategy(history): return {'probability_up': __import__('os').system('x'), 'target_weight':0}",
        "def strategy(history): return {'probability_up':history[-1], 'target_weight':0}",
        "def strategy(history): return {'probability_up':0.5 if 1 else open('x'), 'target_weight':0}",
        "def strategy(history):\n while True: pass\n return {'probability_up':0.5,'target_weight':0}",
        "def strategy(history): return {'probability_up':2**1000000,'target_weight':0}",
        "def strategy(history): return {'probability_up':mean(history, 2+3), 'target_weight':0}",
        "def strategy(history): return {'probability_up':0.5,'target_weight':0,'extra':1}",
        "def strategy(history): return {'probability_up':0.5,'probability_up':0.2}",
        "@decorator\ndef strategy(history): return {'probability_up':0.5,'target_weight':0}",
        "def strategy(history=[]): return {'probability_up':0.5,'target_weight':0}",
        "def strategy(history): return {'probability_up':True, 'target_weight':0}",
        "def strategy(history): return {'probability_up':mean(history, True), 'target_weight':0}",
        "def strategy(history): return {'probability_up':mean(history, 0), 'target_weight':0}",
        "def strategy(history): return {'probability_up':mean(history, 10001), 'target_weight':0}",
        "def strategy(history): return {'probability_up':1e309, 'target_weight':0}",
    ],
)
def test_hostile_programs_rejected_including_dead_branches(code) -> None:
    with pytest.raises(ValueError):
        parse_strategy(code)


def test_interpreter_matches_independent_indicator_reference() -> None:
    history = [100, 98, 103, 102]
    decision = run_strategy(CODE, history)
    expected = np.mean(history[-2:]) > np.mean(history[-3:])
    assert decision.probability_up == (0.7 if expected else 0.3)
    assert decision.target_weight == (1 if expected else -1)
    other = "def strategy(history): return {'probability_up':clip(0.5 + return_over(history, 2),0.1,0.9), 'target_weight':clip(last(history) - minimum(history, 2),-1,1)}"
    actual = run_strategy(other, history)
    assert actual.probability_up == pytest.approx(0.5 + history[-1] / history[-3] - 1)
    assert actual.target_weight == 0


@pytest.mark.parametrize(
    "code,history",
    [
        (CODE, [100, 101]),
        (CODE, [100, float("nan"), 102]),
        (CODE, [100, 0, 102]),
        ("def strategy(history): return {'probability_up':1/0, 'target_weight':0}", [100]),
        ("def strategy(history): return {'probability_up':1.1, 'target_weight':0}", [100]),
        ("def strategy(history): return {'probability_up':0.5, 'target_weight':2}", [100]),
        ("def strategy(history): return {'probability_up':history, 'target_weight':0}", [100]),
    ],
)
def test_runtime_failures_are_closed(code, history) -> None:
    with pytest.raises(ValueError):
        run_strategy(code, history)


def test_replay_scores_predictions_and_costs_against_independent_reference() -> None:
    result = replay()
    assert result.probabilities_up == (0.7, 0.7, 0.3)
    assert result.realized_up == (0, 1, 0)
    assert result.brier_score == pytest.approx((0.49 + 0.09 + 0.09) / 3)
    assert result.binary_log_score == pytest.approx(-(np.log(0.3) + 2 * np.log(0.7)) / 3)
    assert result.target_weights == (1, 1, -1)
    assert result.turnover == 3
    assert result.transaction_cost_bps == 6
    assert result.synthetic and result.research_only and not result.live_pnl_claim


def test_future_suffix_cannot_change_decisions() -> None:
    code = generated()
    tape = bars()
    clocks = [row.event_time for row in tape[2:4]]
    changed = tape[:4] + [replace(row, close=row.close * 20) for row in tape[4:]]
    a = backtest(code, tape, decision_times=clocks, data_source="SYNTHETIC", synthetic=True)
    b = backtest(code, changed, decision_times=clocks, data_source="SYNTHETIC", synthetic=True)
    assert a.probabilities_up == b.probabilities_up
    assert a.target_weights == b.target_weights
    assert a.data_sha256 != b.data_sha256


def test_unavailable_current_event_refuses_to_change_forecast_horizon() -> None:
    tape = bars()
    tape[3] = replace(tape[3], available_time=tape[3].event_time + timedelta(hours=1))
    with pytest.raises(ValueError, match="current event close is unavailable"):
        backtest(
            generated(),
            tape,
            decision_times=[tape[3].event_time],
            data_source="SYNTHETIC",
            synthetic=True,
        )


def test_pit_history_excludes_late_older_observation() -> None:
    tape = bars()
    tape[0] = replace(tape[0], close=10_000, available_time=tape[-1].event_time)
    code = generated(
        "def strategy(history): return {'probability_up':clip(mean(history, 2)/200, 0.1, 0.9), 'target_weight':0}"
    )
    result = backtest(
        code, tape, decision_times=[tape[2].event_time], data_source="SYNTHETIC", synthetic=True
    )
    assert result.probabilities_up == ((101 + 102) / 400,)


def test_specification_judge_is_required_and_errors_do_not_become_passes() -> None:
    code = generated()
    result = replay(code)
    absent = evaluate(spec(), code, result)
    assert absent.specification_correct is None and not absent.judge_passed
    assert absent.failures == ("specification_judge_missing",)
    passed = evaluate(
        spec(), code, result, judge=lambda *_: True, judge_id="independent_test_oracle"
    )
    assert passed.judge_passed
    with pytest.raises(ValueError, match="explicit boolean"):
        evaluate(spec(), code, result, judge=lambda *_: None, judge_id="broken")

    def broken(*_):
        raise RuntimeError("judge unavailable")

    with pytest.raises(RuntimeError, match="judge unavailable"):
        evaluate(spec(), code, result, judge=broken, judge_id="broken")


def test_receipt_immutable_hash_bound_and_missing_judge_not_success(tmp_path) -> None:
    code = generated()
    result = replay(code)
    assessment = evaluate(spec(), code, result)
    path = tmp_path / "receipt.json"
    digest = write_receipt(path, spec(), code, result, assessment)
    payload = json.loads(path.read_text())
    assert payload.pop("receipt_sha256") == digest
    assert (
        hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        == digest
    )
    assert payload["specification_judge_passed"] is False
    assert payload["market_evidence"] is False
    assert payload["trained_checkpoint"] is False
    assert verify_strategy_receipt(path)["replay_reproduced"] is True
    with pytest.raises(FileExistsError):
        write_receipt(path, spec(), code, result, assessment)
    with pytest.raises(ValueError, match="evaluation is not bound"):
        write_receipt(
            tmp_path / "other.json", spec(), code, result, replace(assessment, data_sha256="f" * 64)
        )


def test_verifier_rejects_rehashed_unreproducible_prediction_trace(tmp_path) -> None:
    code = generated()
    result = replay(code)
    assessment = evaluate(spec(), code, result)
    path = tmp_path / "receipt.json"
    write_receipt(path, spec(), code, result, assessment)
    payload = json.loads(path.read_text())
    payload["replay"]["source_bars"][0][2] = 9999
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source data hash mismatch"):
        verify_strategy_receipt(path)


def test_poisoned_result_cannot_claim_honesty_or_scores() -> None:
    result = replay()
    with pytest.raises(ValueError, match="honesty"):
        replace(result, live_pnl_claim=True)
    with pytest.raises(ValueError, match="brier_score"):
        replace(result, brier_score=0)
    with pytest.raises(ValueError, match="aligned"):
        replace(result, realized_up=(1,))
    with pytest.raises(ValueError, match="explicit boolean"):
        replace(result, success_flag="yes")


def test_judge_receipt_cannot_transfer_to_different_decisions_on_same_data(tmp_path) -> None:
    code = generated()
    tape = bars()
    first = backtest(
        code, tape, decision_times=[tape[2].event_time], data_source="SYNTHETIC", synthetic=True
    )
    second = backtest(
        code, tape, decision_times=[tape[3].event_time], data_source="SYNTHETIC", synthetic=True
    )
    assert first.data_sha256 == second.data_sha256
    assert first.sha256 != second.sha256
    judgment = evaluate(spec(), code, first, judge=lambda *_: True, judge_id="fixture_oracle")
    with pytest.raises(ValueError, match="evaluation is not bound"):
        write_receipt(tmp_path / "forged.json", spec(), code, second, judgment)


def test_specification_snapshot_does_not_follow_mutable_parameters() -> None:
    values = {"lookback": 3}
    request = StrategySpec("x", "x", "x", values)
    fingerprint = request.sha256
    values["lookback"] = 100
    assert request.sha256 == fingerprint
    with pytest.raises(TypeError):
        request.parameters["lookback"] = 4


def test_timestamp_and_data_source_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        ReplayBar(datetime(2020, 1, 1), datetime(2020, 1, 1), 100)
    tape = bars()
    with pytest.raises(ValueError, match="synthetic source"):
        backtest(
            generated(),
            tape,
            decision_times=[tape[2].event_time],
            data_source="SYNTHETIC",
            synthetic=False,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        backtest(
            generated(),
            list(reversed(tape)),
            decision_times=[tape[2].event_time],
            data_source="SYNTHETIC",
            synthetic=True,
        )


def test_decision_budget_applies_even_with_two_observations() -> None:
    tape = bars((100, 101))
    clocks = [tape[0].event_time + timedelta(microseconds=i) for i in range(10_001)]
    code = generated("def strategy(history): return {'probability_up':0.5,'target_weight':0}")
    with pytest.raises(ValueError, match="decision count budget"):
        backtest(code, tape, decision_times=clocks, data_source="SYNTHETIC", synthetic=True)
