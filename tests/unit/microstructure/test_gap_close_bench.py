"""Tests for gap_close_bench."""

from quant_fund.microstructure.gap_close_bench import gap_close_bench


def test_gap_close_shape() -> None:
    out = gap_close_bench(horizon=3000, seed=7)
    assert out["schema"] == "gap_close.v1"
    assert out["research_only"] is True
    assert len(out["cells"]) == 6
    assert [c["repost_band"] for c in out["cells"]] == [4, 8, 12, 8, 6, 8]
    for c in out["cells"]:
        assert c["n_pins_ok"] == sum(c["pins"].values())
    assert set(out["claims"]) == {
        "cells_measured",
        "gap_closes",
        "full_closure",
        "geometry_moves_gap",
    }
    assert len(out["receipt_sha256"]) == 64
