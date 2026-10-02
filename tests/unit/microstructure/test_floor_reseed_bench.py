"""Tests for floor_reseed_bench."""

from quant_fund.microstructure.floor_reseed_bench import floor_reseed_bench


def test_floor_reseed_shape() -> None:
    out = floor_reseed_bench(horizon=3000, seed=7)
    assert out["schema"] == "floor_reseed.v1"
    assert out["research_only"] is True
    assert len(out["cells"]) == 6
    assert [c["repost_frac"] for c in out["cells"]] == [0.0, 0.3, 0.6, 0.6, 0.6, 0.6]
    assert [c["fill_repost_delay"] for c in out["cells"]] == [320, 320, 320, 320, 240, 280]
    for c in out["cells"]:
        assert c["n_pins_ok"] == sum(c["pins"].values())
    assert set(out["claims"]) == {
        "cells_measured",
        "reseed_recovered",
        "full_closure",
        "spread_survives_reseed",
    }
    assert len(out["receipt_sha256"]) == 64
