"""Engine-fuzz lane: invariants hold on a fresh sim and verdict seals."""

from __future__ import annotations

import json

from quant_fund.formal.engine_fuzz import (
    check_book_invariants,
    engine_fuzz,
    engine_fuzz_bench,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator, santa_fe_config
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_fresh_book_satisfies_invariants() -> None:
    sim = ZILobSimulator(santa_fe_config())
    assert check_book_invariants(sim) == []


def test_deterministic_seed_reproduces() -> None:
    a = engine_fuzz(seed=7, n_steps=120)
    b = engine_fuzz(seed=7, n_steps=120)
    assert a["n_violations"] == b["n_violations"]
    assert a["counters"] == b["counters"]
    assert a["n_fills"] == b["n_fills"]


def test_bench_seals_and_verifies() -> None:
    receipt = engine_fuzz_bench(seeds=(0,), n_steps=120)
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "engine_fuzz_test.json")
    assert result["valid"], result.get("errors")


def test_bench_writes_canonical_receipt_shape() -> None:
    receipt = engine_fuzz_bench(seeds=(0,), n_steps=60)
    assert receipt["kind"] == "engine_fuzz"
    assert receipt["schema"] == "engine_fuzz.v1"
    assert receipt["data_label"] == "SYNTHETIC"
    json.dumps(receipt)  # serializable
