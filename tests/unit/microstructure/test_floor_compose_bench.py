"""Tests for floor_compose_bench."""

from quant_fund.microstructure.floor_compose_bench import floor_compose_bench


def test_floor_compose_shape() -> None:
    out = floor_compose_bench(horizon=3000, seed=7)
    assert out["schema"] == "floor_compose.v1"
    assert out["research_only"] is True
    assert len(out["cells"]) == 6
    assert [c["min_quote_dist"] for c in out["cells"]] == [8, 8, 8, 6, 10, 8]
    for c in out["cells"]:
        assert c["n_pins_ok"] == sum(c["pins"].values())
    assert set(out["claims"]) == {
        "cells_measured",
        "spread_survives_scan",
        "floor_composes",
        "near_closure",
    }
    assert len(out["receipt_sha256"]) == 64
