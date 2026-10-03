"""Tests for zone_map."""

from quant_fund.microstructure.zone_map import zone_map


def test_zone_map_shape(tmp_path) -> None:
    import json as j
    import shutil
    from pathlib import Path

    src = Path("receipts")
    for name in (
        "wave24_map.json",
        "band_occupancy.json",
        "zone_embargo.json",
        "zone_stability.json",
    ):
        shutil.copy(src / name, tmp_path / name)
    out = zone_map(tmp_path)
    assert out["schema"] == "zone_map.v1"
    assert out["research_only"] is True
    assert out["n_lanes"] == 4
    assert all(lane["sealed"] for lane in out["lanes"])
    assert set(out["claims"]) == {
        "all_member_receipts_sealed",
        "all_lanes_present",
        "ambient_floor_falsified",
        "zone_reaches_occupancy",
        "closure_near_structural",
        "kernel_carried",
    }
    assert out["claims"]["closure_near_structural"] is True
    assert len(out["receipt_sha256"]) == 64
    j.dumps(out)
