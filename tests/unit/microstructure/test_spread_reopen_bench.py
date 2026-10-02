"""Tests for spread_reopen_bench."""

from quant_fund.microstructure.spread_reopen_bench import spread_reopen_bench


def test_spread_reopen_shape() -> None:
    out = spread_reopen_bench(horizon=3000, seed=7)
    assert out["schema"] == "spread_reopen.v1"
    assert out["research_only"] is True
    assert len(out["cells"]) == 5
    assert [c["lo_offset"] for c in out["cells"]] == [4, 8, 12, 8, 12]
    assert [c["flow_intensity"] for c in out["cells"]] == [None, None, None, 2.0, 2.0]
    for c in out["cells"]:
        assert c["n_pins_ok"] == sum(c["pins"].values())
    assert set(out["claims"]) == {
        "cells_measured",
        "spread_reopens_under_iid",
        "emptied_touch_survives",
        "spread_bounded_above",
    }
    assert len(out["receipt_sha256"]) == 64
