"""Stale timestamp anchors report an age and are not a green claim."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from quant_fund.research import timestamp_anchor as ta
from quant_fund.research.repo_integrity import format_verify_repo_line, verify_repo_summary
from quant_fund.research.timestamp_anchor import extract_gen_time_unix, verify_timestamps
from quant_fund.utils.hashing import hash_bytes


def _tsr(imprint: bytes, gen_time: bytes) -> bytes:
    alg_id = ta._seq(ta.SHA256_ALG_OID + ta._tlv(0x05, b""))
    msg_imprint = ta._seq(alg_id + ta._tlv(0x04, imprint))
    tst_info = ta._seq(
        ta._tlv(0x02, b"\x01") + ta._tlv(0x06, b"\x2b\x06") + msg_imprint + ta._tlv(0x18, gen_time)
    )
    encap = ta._seq(ta._tlv(0x06, b"\x2b\x06") + ta._tlv(0xA0, ta._tlv(0x04, tst_info)))
    signed_data = ta._seq(ta._tlv(0x02, b"\x01") + ta._tlv(0x31, b"") + encap)
    content_info = ta._seq(ta._tlv(0x06, b"\x2b\x06") + ta._tlv(0xA0, signed_data))
    status = ta._seq(ta._tlv(0x02, b"\x00"))
    return ta._seq(status + content_info)


def test_stale_anchor_reports_eighteen_days(tmp_path: Path) -> None:
    target = tmp_path / "quality" / "epoch_heads.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"pinned state bytes")
    digest = hash_bytes(target.read_bytes())
    ts_dir = tmp_path / ta.TIMESTAMPS_DIR
    ts_dir.mkdir(parents=True)
    (ts_dir / "x.tsr").write_bytes(_tsr(bytes.fromhex(digest), b"20260929221601Z"))
    (ts_dir / ta.ANCHORS_MANIFEST).write_text(
        json.dumps(
            {
                "schema": "timestamp_anchors.v1",
                "anchors": {"x.tsr": {"target": "quality/epoch_heads.json", "sha256": digest}},
            }
        )
    )
    target.write_bytes(b"moved")
    now = int(datetime(2026, 10, 17, 22, 16, 1, tzinfo=UTC).timestamp())
    res = verify_timestamps(tmp_path, now=now)
    assert res["ok"] is True
    stale = res["staleness"]["quality/epoch_heads.json"]
    assert stale["stale"] is True
    assert stale["age_days"] == 18
    line = format_verify_repo_line(
        "timestamp_anchors",
        {"ok": True, "errors": [], "stale": True, "age_days": 18},
    )
    assert line == "verify-repo timestamp_anchors: stale age=18d"
    summary = verify_repo_summary({"gates": {"timestamp_anchors": {"stale": True, "age_days": 18}}})
    assert summary == "repo integrity: timestamp_anchors stale age=18d"
    assert "all gates intact" not in summary


def test_committed_epoch_heads_token_gen_time() -> None:
    token = Path("quality/timestamps/quality__epoch_heads.json.tsr").read_bytes()
    assert extract_gen_time_unix(token) == int(
        datetime(2026, 9, 29, 22, 16, 1, tzinfo=UTC).timestamp()
    )
