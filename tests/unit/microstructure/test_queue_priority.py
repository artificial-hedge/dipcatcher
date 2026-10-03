"""Tests for microstructure/queue_priority.py — queue-priority lane.

Fill probability as a function of queue position at submit, measured on the
seeded SYNTHETIC ZI-LOB (iid + MarkovRegimeFlow arms). All results are
labeled SYNTHETIC correctness diagnostics — never market evidence, no
live-trading claim.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from quant_fund.microstructure.queue_priority import (
    QUEUE_PRIORITY_SCHEMA,
    QueueSubmission,
    fill_curve,
    fill_probability_by_queue,
    queue_priority_bench,
    run_queue_arm,
)
from quant_fund.microstructure.zi_lob_simulator import TradeEvent, ZILobConfig
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


def _trade(oid: int, qa: int, *, t: float = 1.0, t_submit: float = 0.0) -> TradeEvent:
    return TradeEvent(
        t=t,
        aggressor="buy",
        price=100.0,
        level=0,
        qty=1,
        maker_order_id=oid,
        maker_side="sell",
        maker_tag="zi",
        maker_t_submit=t_submit,
        maker_queue_ahead_at_submit=qa,
    )


# ---------------------------------------------------------------------------
# fill_curve — histogram correctness on a constructed trade list
# ---------------------------------------------------------------------------


def test_fill_curve_histogram_and_pmf() -> None:
    trades = [
        _trade(1, 0, t=1.0, t_submit=0.0),
        _trade(2, 0, t=2.0, t_submit=1.0),
        _trade(3, 1, t=3.0, t_submit=0.0),
        _trade(4, 0, t=4.0, t_submit=0.0),
        _trade(5, 2, t=5.0, t_submit=0.0),
    ]
    curve = fill_curve(trades)
    assert curve["n_fills"] == 5
    assert curve["queue_ahead_hist"] == {"0": 3, "1": 1, "2": 1}
    assert curve["queue_ahead_pmf"] == {"0": 0.6, "1": 0.2, "2": 0.2}
    assert curve["p_queue_ahead_zero"] == pytest.approx(0.6)
    assert curve["mode_queue_ahead"] == 0
    assert curve["front_dominant"] is True
    assert curve["mean_queue_ahead"] == pytest.approx(0.6)
    assert curve["median_queue_ahead"] == pytest.approx(0.0)
    assert curve["max_queue_ahead"] == 2
    # wait = t - maker_t_submit: [1.0, 1.0, 3.0, 4.0, 5.0]
    assert curve["mean_fill_wait_seconds"] == pytest.approx(2.8)
    assert curve["median_fill_wait_seconds"] == pytest.approx(3.0)
    assert curve["max_fill_wait_seconds"] == pytest.approx(5.0)
    assert curve["fill_wait_by_queue"]["0"]["n_fills"] == 3
    assert curve["fill_wait_by_queue"]["0"]["mean_wait_seconds"] == pytest.approx(2.0)
    assert curve["fill_wait_by_queue"]["1"]["mean_wait_seconds"] == pytest.approx(3.0)


def test_fill_curve_front_not_dominant() -> None:
    trades = [_trade(i, 3) for i in range(4)] + [_trade(9, 0)]
    curve = fill_curve(trades)
    assert curve["mode_queue_ahead"] == 3
    assert curve["front_dominant"] is False
    assert curve["p_queue_ahead_zero"] == pytest.approx(0.2)


def test_fill_curve_empty_fails_closed() -> None:
    with pytest.raises(ValueError, match="non-empty trade stream"):
        fill_curve([])


# ---------------------------------------------------------------------------
# fill_probability_by_queue — submission-side join
# ---------------------------------------------------------------------------


def test_fill_probability_by_queue_join() -> None:
    # orders 1,4 filled at q=0; order 3 canceled at q=0; order 5 resting at q=0
    # order 2 filled at q=2; order 6 canceled at q=2
    trades = [_trade(1, 0, t=2.0, t_submit=1.0), _trade(4, 0), _trade(2, 2, t=5.0)]
    subs = [
        QueueSubmission(1, 0, resting=False),
        QueueSubmission(3, 0, resting=False),
        QueueSubmission(4, 0, resting=False),
        QueueSubmission(5, 0, resting=True),
        QueueSubmission(2, 2, resting=False),
        QueueSubmission(6, 2, resting=False),
    ]
    out = fill_probability_by_queue(subs, trades)
    assert out["n_submissions"] == 6
    assert out["n_filled"] == 3
    assert out["n_canceled"] == 2
    assert out["n_resting_censored"] == 1
    assert out["p_fill_overall"] == pytest.approx(0.5)
    q0 = out["per_queue_position"]["0"]
    assert q0["n_submitted"] == 4 and q0["n_filled"] == 2 and q0["n_canceled"] == 1
    assert q0["n_resting"] == 1
    assert q0["p_fill"] == pytest.approx(0.5)
    assert q0["p_cancel"] == pytest.approx(0.25)
    assert q0["p_censored"] == pytest.approx(0.25)
    assert q0["mean_fill_wait_seconds"] == pytest.approx(1.0)  # waits 1.0, 1.0
    q2 = out["per_queue_position"]["2"]
    assert q2["p_fill"] == pytest.approx(0.5)
    assert q2["p_cancel"] == pytest.approx(0.5)


def test_fill_probability_by_queue_empty_fails_closed() -> None:
    with pytest.raises(ValueError, match="non-empty submission stream"):
        fill_probability_by_queue([], [])


# ---------------------------------------------------------------------------
# run_queue_arm — determinism + tracked submission stream
# ---------------------------------------------------------------------------


def test_run_queue_arm_deterministic() -> None:
    cfg_a = ZILobConfig(seed=11)
    cfg_b = ZILobConfig(seed=11)
    a = run_queue_arm(cfg_a, horizon=400.0, warmup=50.0)
    b = run_queue_arm(cfg_b, horizon=400.0, warmup=50.0)
    assert a == b


def test_run_queue_arm_submission_conservation() -> None:
    out = run_queue_arm(ZILobConfig(seed=3), horizon=400.0, warmup=0.0)
    surv = out["queue_survival"]
    assert (
        surv["n_filled"] + surv["n_canceled"] + surv["n_resting_censored"]
        == surv["n_submissions"]
        == out["n_tracked_submissions"]
    )
    # every window fill resolves against a tracked submission
    assert out["fill_curve"]["n_fills"] == surv["n_filled"]
    # conservation: created = fills + cancels + resting (pre-window untracked is 0)
    ec = out["event_counts"]
    assert ec["n_orders_created"] == surv["n_submissions"]
    assert out["n_untracked_pre_window"] == 0


def test_run_queue_arm_fill_consistency_with_hist() -> None:
    out = run_queue_arm(ZILobConfig(seed=5), horizon=400.0, warmup=50.0)
    hist = out["fill_curve"]["queue_ahead_hist"]
    assert sum(hist.values()) == out["fill_curve"]["n_fills"]
    assert abs(sum(out["fill_curve"]["queue_ahead_pmf"].values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# queue_priority_bench — schema + sealed receipt
# ---------------------------------------------------------------------------


def test_queue_priority_bench_schema_and_seal(tmp_path: Path) -> None:
    out = tmp_path / "queue_priority.json"
    payload = queue_priority_bench(seed=13, horizon=400.0, warmup=50.0, out_path=out)
    assert payload["kind"] == "queue_priority"
    assert payload["schema"] == QUEUE_PRIORITY_SCHEMA
    assert payload["data_label"] == "SYNTHETIC"
    assert payload["research_only"] is True
    assert payload["live_pnl_claim"] is False
    assert isinstance(payload["code_revision"], str) and payload["code_revision"]
    assert set(payload["arms"]) == {"iid", "markov_regime"}
    # seal verifies: digest over canonical payload minus the seal key itself
    unsigned = {k: v for k, v in payload.items() if k != "receipt_sha256"}
    assert payload["receipt_sha256"] == hash_bytes(canonical_json_bytes(unsigned))
    # file written, parses, byte-identical to the sealed payload, no NaN/Inf
    text = out.read_text()
    on_disk = json.loads(text)
    assert on_disk == payload
    json.dumps(payload, allow_nan=False)


def test_queue_priority_bench_payload_no_nan_or_inf(tmp_path: Path) -> None:
    payload = queue_priority_bench(
        seed=17, horizon=300.0, warmup=50.0, out_path=tmp_path / "q.json"
    )

    def _walk(obj: object) -> None:
        if isinstance(obj, float):
            assert math.isfinite(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)

    _walk(payload)


def test_queue_priority_bench_deterministic_payload(tmp_path: Path) -> None:
    a = queue_priority_bench(seed=19, horizon=300.0, warmup=50.0, out_path=tmp_path / "a.json")
    b = queue_priority_bench(seed=19, horizon=300.0, warmup=50.0, out_path=tmp_path / "b.json")
    assert a == b


def test_run_queue_arm_empty_trades_fails_closed() -> None:
    # mu ~ 0 => no market orders => an empty fill stream -> fail closed.
    with pytest.raises(ValueError, match="non-empty trade stream"):
        run_queue_arm(ZILobConfig(seed=23, mu=1e-9), horizon=5.0, warmup=0.0)
