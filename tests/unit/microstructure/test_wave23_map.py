"""wave23_map — sealed synthesis over the wave-23 lane receipts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from quant_fund.microstructure.wave23_map import WAVE23_MAP_SCHEMA, wave23_map
from quant_fund.research.receipt_v2 import verify_receipt_file

_RECEIPTS = Path(__file__).resolve().parents[3] / "receipts"


def test_wave23_map_over_committed_receipts(tmp_path: Path) -> None:
    payload = wave23_map(tmp_path if not _RECEIPTS.exists() else _RECEIPTS)
    assert payload["schema"] == WAVE23_MAP_SCHEMA
    assert payload["n_lanes"] == 19
    c = payload["claims"]
    if _RECEIPTS.exists() and any((_RECEIPTS / f).exists() for f in m_lanes()):
        # Committed corpus: member seals verify, both channels located.
        assert c["all_member_receipts_sealed"]
        assert c["instant_channel_located"]
        assert c["continuation_channel_located"]
        # Wave-23 ended falsified — no lane closed all five targets.
        assert c["closure_path_found"] is False
    out = tmp_path / "wave23_map_test.json"
    out.write_text(json.dumps(payload))
    assert verify_receipt_file(out)["valid"]


def m_lanes() -> list[str]:
    import quant_fund.microstructure.wave23_map as m

    return [f for f, _ in m._LANES]


def test_wave23_map_missing_receipts_fail_closed(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    payload = wave23_map(empty)
    assert payload["claims"]["all_member_receipts_sealed"] is False
    assert payload["claims"]["all_lanes_present"] is False
    assert payload["claims"]["closure_path_found"] is False


def test_wave23_map_partial_corpus(tmp_path: Path) -> None:
    """A subset of valid sealed receipts still synthesizes a map."""
    sub = tmp_path / "sub"
    sub.mkdir()
    src = _RECEIPTS / "release_chase_amzn.json"
    if src.exists():
        shutil.copy(src, sub / "release_chase_amzn.json")
    payload = wave23_map(sub)
    assert payload["claims"]["all_lanes_present"] is False
    if src.exists():
        rel = next(e for e in payload["lanes"] if e["receipt"] == "release_chase_amzn.json")
        assert rel["sealed"] is True
        assert rel["claims"]["joint_closure_exists"] is False
