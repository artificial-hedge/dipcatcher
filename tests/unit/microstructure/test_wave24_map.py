"""Tests for wave24_map."""

from quant_fund.microstructure.wave24_map import wave24_map


def test_wave24_map_shape() -> None:
    out = wave24_map()
    assert out["schema"] == "wave24_map.v1"
    assert out["research_only"] is True
    assert out["n_lanes"] == 10
    assert len(out["lanes"]) == 10
    assert out["claims"]["all_member_receipts_sealed"] is True
    assert out["claims"]["all_lanes_present"] is True
    assert set(out["claims"]) == {
        "all_member_receipts_sealed",
        "all_lanes_present",
        "band_opened",
        "floor_composes",
        "seven_pin_reachable",
        "stable_closure",
    }
    assert len(out["receipt_sha256"]) == 64
