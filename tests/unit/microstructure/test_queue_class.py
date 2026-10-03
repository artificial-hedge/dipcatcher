"""Tests for placement-class fate tracking + microstructure/queue_class_bench.py.

Every resting order is tagged at submit with ``join`` / ``improve`` /
``deep`` vs the own-side touch; fills and cancels tally against the class.
SYNTHETIC correctness only — never market evidence.
"""

from __future__ import annotations

import pytest

from quant_fund.microstructure.queue_class_bench import (
    QUEUE_CLASS_SCHEMA,
    queue_class_bench,
)
from quant_fund.microstructure.zi_lob_simulator import (
    PLACEMENT_CLASSES,
    ZILobSimulator,
    santa_fe_config,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


def _run(seed: int, *, band: int = 5, horizon: float = 300.0) -> ZILobSimulator:
    sim = ZILobSimulator(santa_fe_config(seed=seed, band=band))
    while sim.t < horizon:
        sim.step()
    return sim


def test_placement_class_covers_every_rest() -> None:
    sim = ZILobSimulator(santa_fe_config(seed=3))
    assert {o.placement_class for o in sim._orders.values()} <= set(PLACEMENT_CLASSES)
    counts = sim.event_counts()
    placed = sum(counts[f"placed_{c}"] for c in PLACEMENT_CLASSES)
    assert placed == counts["n_orders_created"]


def test_fate_counters_conserve() -> None:
    sim = _run(seed=11)
    fate = sim.fate_by_class()
    counts = sim.event_counts()
    assert set(fate) == set(PLACEMENT_CLASSES)
    for c in PLACEMENT_CLASSES:
        f = fate[c]
        assert f["placed"] == f["fills"] + f["cancels"] + f["resting"]
        assert counts[f"fills_{c}"] == f["fills"]
        assert counts[f"cancels_{c}"] == f["cancels"]
    assert sum(f["fills"] for f in fate.values()) == counts["n_fills"]
    assert sum(f["cancels"] for f in fate.values()) == counts["n_cancellations"]
    assert sum(f["resting"] for f in fate.values()) == counts["resting"]
    # fate_log has one record per resolved order.
    assert len(sim.fate_log) == counts["n_fills"] + counts["n_cancellations"]
    assert {r[0] for r in sim.fate_log} <= set(PLACEMENT_CLASSES)
    assert {r[2] for r in sim.fate_log} <= {"fill", "cancel"}


def test_manual_placement_classes() -> None:
    sim = ZILobSimulator(santa_fe_config(seed=5))
    # Seed book: bids at -1..-3, asks at +1..+3 (init_levels=3, init_depth=5).
    # Classification is against the touch at submit time: join the -1 bid,
    # then improve to level 0 (new best), then sit behind it.
    oid_join = sim.submit_limit_order("buy", sim.level_to_price(-1))
    oid_improve = sim.submit_limit_order("buy", sim.level_to_price(0))
    oid_deep = sim.submit_limit_order("buy", sim.level_to_price(-4))
    orders = sim._orders
    assert orders[oid_join].placement_class == "join"
    assert orders[oid_improve].placement_class == "improve"
    assert orders[oid_deep].placement_class == "deep"
    # Fills and cancels tally against the class of the removed order.
    tr = sim.inject_market_order("sell", qty=1)[0]
    assert tr.maker_placement_class == "improve"
    assert sim.event_counts()["fills_improve"] == 1
    assert sim.cancel_order(oid_deep)
    assert sim.event_counts()["cancels_deep"] == 1


def test_twin_run_bit_identical() -> None:
    a, b = _run(seed=29), _run(seed=29)
    assert a.event_counts() == b.event_counts()
    assert a.fate_log == b.fate_log
    assert [(t.t, t.aggressor, t.level, t.maker_order_id) for t in a.trades] == [
        (t.t, t.aggressor, t.level, t.maker_order_id) for t in b.trades
    ]


@pytest.mark.slow
def test_bench_seal_and_contract() -> None:
    from quant_fund.research.receipt_v2 import verify_receipt_payload

    payload = queue_class_bench(horizon=400.0, seed=13)
    assert payload["schema"] == QUEUE_CLASS_SCHEMA
    assert payload["kind"] == "queue_class"
    assert payload["data_label"] == "MIXED"
    assert payload["research_only"] is True
    assert set(payload["arms"]) == {"deep_band14", "touch_band1"}
    seal = payload["receipt_sha256"]
    assert seal == hash_bytes(
        canonical_json_bytes({k: v for k, v in payload.items() if k != "receipt_sha256"})
    )
    result = verify_receipt_payload(payload)
    assert result["valid"], result["errors"]
