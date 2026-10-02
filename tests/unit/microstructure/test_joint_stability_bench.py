"""Tests for joint_stability_bench."""

from __future__ import annotations

import pytest

from quant_fund.microstructure.joint_stability_bench import (
    JOINT_STABILITY_SCHEMA,
    joint_stability_bench,
)


@pytest.mark.slow
def test_bench_smoke() -> None:
    out = joint_stability_bench(horizon=8000)
    assert out["schema"] == JOINT_STABILITY_SCHEMA
    assert out["research_only"] is True
    assert len(out["cells"]) == 5
    for row in out["cells"]:
        assert len(row["seeds"]) == 4
        for s in row["seeds"]:
            assert s["n_fills"] > 0
            assert set(s["pins"]) == {
                "empty_share",
                "spread_mean",
                "crown_share",
                "hidden_share",
            }
            assert s["joint"] == all(s["pins"].values())


def test_per_pin_pass_counts() -> None:
    out = joint_stability_bench(horizon=4000)
    for row in out["cells"]:
        for pin, n in row["per_pin_pass"].items():
            assert 0 <= n <= 4
            assert n == sum(s["pins"][pin] for s in row["seeds"])
        assert row["n_joint_seeds"] == sum(s["joint"] for s in row["seeds"])
