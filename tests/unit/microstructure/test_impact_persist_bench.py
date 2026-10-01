"""impact_persist_bench — post-fill drift kernel vs the real tape."""

from __future__ import annotations

import pytest

from quant_fund.microstructure.impact_persist_bench import (
    IMPACT_PERSIST_SCHEMA,
    impact_persist_bench,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


def test_bench_smoke_and_seal() -> None:
    out = impact_persist_bench(horizon=3000, seed=5)
    assert out["schema"] == IMPACT_PERSIST_SCHEMA
    assert out["kind"] == "impact_persist_bench"
    assert out["data_label"] == "SYNTHETIC"
    assert out["research_only"] is True
    assert set(out["arms"]) == {"iid", "split", "deep", "deep_split", "lv", "lv_split"}
    for arm in out["arms"].values():
        assert set(arm["kernel_mean_ticks"]) == {"1", "5", "20", "50", "200"}
    seal = out.pop("receipt_sha256")
    assert seal == hash_bytes(canonical_json_bytes(out))


def test_bench_deterministic() -> None:
    a = impact_persist_bench(horizon=2000, seed=11)
    b = impact_persist_bench(horizon=2000, seed=11)
    assert a["receipt_sha256"] == b["receipt_sha256"]


def test_divergences_are_strings() -> None:
    out = impact_persist_bench(horizon=2500, seed=3)
    assert isinstance(out["divergences"], list)
    for d in out["divergences"]:
        assert isinstance(d, str)


@pytest.mark.parametrize("arm", ["iid", "split"])
def test_finite_arm_fills(arm: str) -> None:
    out = impact_persist_bench(horizon=4000, seed=9)
    assert out["arms"][arm]["n_fills"] > 0
