"""Tests for zone_card_bench."""

from quant_fund.microstructure.zone_card_bench import zone_card_bench


def test_zone_card_shape() -> None:
    out = zone_card_bench(horizon=1200, seed=7)
    assert out["schema"] == "zone_card.v1"
    assert out["research_only"] is True
    assert out["data_label"] == "MIXED"
    assert len(out["cells"]) == 6
    for c in out["cells"]:
        assert c["n_draws"] == 2
        m = c["mix_mean"]
        for key in ("sub", "delete", "exec"):
            assert m[key] is None or 0.0 <= m[key] <= 1.0
        for d in c["draws"]:
            assert d["n_fills"] >= 0
    assert set(out["claims"]) == {
        "cells_measured",
        "zone_improves_card",
        "life_on_tape_scale",
        "exec_share_on_card",
    }
    assert len(out["receipt_sha256"]) == 64
