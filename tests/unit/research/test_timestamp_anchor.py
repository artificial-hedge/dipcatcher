"""Tests for timestamp_anchor — RFC 3161 anchors over the integrity pins."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.research import timestamp_anchor as ta
from quant_fund.research.timestamp_anchor import (
    ANCHORS_MANIFEST,
    TIMESTAMPS_DIR,
    _seq,
    _seq_children,
    _tlv,
    build_tsq,
    extract_imprint,
    verify_timestamps,
)
from quant_fund.utils.hashing import hash_bytes

DIGEST = "ab" * 32


def _fake_tsr(imprint: bytes) -> bytes:
    """Build a minimal DER TimeStampResp carrying ``imprint`` in TSTInfo."""
    alg_id = _seq(ta.SHA256_ALG_OID + _tlv(0x05, b""))
    msg_imprint = _seq(alg_id + _tlv(0x04, imprint))
    tst_info = _seq(_tlv(0x02, b"\x01") + _tlv(0x06, b"\x2b\x06") + msg_imprint)
    encap = _seq(_tlv(0x06, b"\x2b\x06") + _tlv(0xA0, _tlv(0x04, tst_info)))
    signed_data = _seq(_tlv(0x02, b"\x01") + _tlv(0x31, b"") + encap)
    content_info = _seq(_tlv(0x06, b"\x2b\x06") + _tlv(0xA0, signed_data))
    status = _seq(_tlv(0x02, b"\x00"))
    return _seq(status + content_info)


def _anchored_repo(tmp_path: Path) -> Path:
    target = tmp_path / "quality" / "epoch_heads.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"pinned state bytes")
    digest = hash_bytes(target.read_bytes())
    ts_dir = tmp_path / TIMESTAMPS_DIR
    ts_dir.mkdir(parents=True)
    (ts_dir / "x.tsr").write_bytes(_fake_tsr(bytes.fromhex(digest)))
    (ts_dir / ANCHORS_MANIFEST).write_text(
        json.dumps(
            {
                "schema": "timestamp_anchors.v1",
                "anchors": {"x.tsr": {"target": "quality/epoch_heads.json", "sha256": digest}},
            }
        )
    )
    return tmp_path


def test_build_tsq_is_wellformed_der() -> None:
    tsq = build_tsq(DIGEST)
    # outer SEQ { INT 1, SEQ MessageImprint, BOOLEAN TRUE }; len < 0x80
    assert tsq[0] == 0x30
    children = _seq_children(tsq[2 : 2 + tsq[1]])
    assert children[0] == (0x02, b"\x01")
    assert children[2] == (0x01, b"\xff")
    mi = _seq_children(children[1][1])
    assert mi[1] == (0x04, bytes.fromhex(DIGEST))


def test_extract_imprint_round_trip() -> None:
    tsr = _fake_tsr(bytes.fromhex(DIGEST))
    assert extract_imprint(tsr).hex() == DIGEST


def test_extract_imprint_rejects_garbage() -> None:
    with pytest.raises(ta.DerError):
        extract_imprint(b"not der")
    with pytest.raises(ta.DerError):
        extract_imprint(_seq(b"\x01\x00"))


def test_verify_neutral_when_unanchored(tmp_path: Path) -> None:
    res = verify_timestamps(tmp_path)
    assert res == {"ok": True, "anchored": False, "errors": []}


def test_verify_anchor_fresh_and_bound(tmp_path: Path) -> None:
    root = _anchored_repo(tmp_path)
    res = verify_timestamps(root)
    assert res["ok"] is True
    assert res["anchored"] is True
    assert res["fresh"] == {"quality/epoch_heads.json": True}
    # openssl may or may not exist on the test host — both states are honest.
    assert res["chain"]["quality/epoch_heads.json"] in {
        "chain_unchecked",
        "chain_failed",  # fake token has no real TSA signature
    }


def test_verify_stale_anchor_is_valid_not_fresh(tmp_path: Path) -> None:
    root = _anchored_repo(tmp_path)
    (root / "quality/epoch_heads.json").write_bytes(b"new bytes")
    res = verify_timestamps(root)
    assert res["ok"] is True  # stale anchor = proof of earlier state
    assert res["fresh"]["quality/epoch_heads.json"] is False


def test_verify_manifest_token_mismatch_fails(tmp_path: Path) -> None:
    root = _anchored_repo(tmp_path)
    manifest = json.loads((root / TIMESTAMPS_DIR / ANCHORS_MANIFEST).read_text())
    manifest["anchors"]["x.tsr"]["sha256"] = "ff" * 32
    (root / TIMESTAMPS_DIR / ANCHORS_MANIFEST).write_text(json.dumps(manifest))
    res = verify_timestamps(root)
    assert res["ok"] is False
    assert res["errors"] == ["imprint_mismatch:quality/epoch_heads.json"]


def test_verify_missing_target_fails(tmp_path: Path) -> None:
    root = _anchored_repo(tmp_path)
    (root / "quality/epoch_heads.json").unlink()
    res = verify_timestamps(root)
    assert res["ok"] is False
    assert res["errors"] == ["anchor_target_missing:quality/epoch_heads.json"]


def test_malformed_manifest_fails_closed(tmp_path: Path) -> None:
    ts_dir = tmp_path / TIMESTAMPS_DIR
    ts_dir.mkdir(parents=True)
    (ts_dir / ANCHORS_MANIFEST).write_text('{"schema": "other"}')
    res = verify_timestamps(tmp_path)
    assert res["ok"] is False
    assert res["errors"] == ["anchors_manifest_malformed"]


def test_anchors_manifest_is_chain_exempt(tmp_path: Path) -> None:
    """``timestamps/anchors.json`` is exempt — each anchor rewrites it, so
    membership would stale every anchor instantly; its integrity rides on
    the .tsr imprint binding + pinned TSA certs. The exemption is an exact
    rel-path, NOT a basename: a top-level ``anchors.json`` is a member —
    a real corpus file can never hide behind the exempt name."""
    from quant_fund.research.corpus_epoch import member_digests

    q = tmp_path / "quality"
    (q / "timestamps").mkdir(parents=True)
    (q / "x.json").write_text("{}")
    (q / "anchors.json").write_text("{}")  # basename alone does NOT exempt
    (q / "timestamps" / "anchors.json").write_text("{}")
    (q / "epoch_heads.json").write_text("{}")
    assert set(member_digests(q)) == {"anchors.json", "x.json"}


def test_real_committed_token_extracts(tmp_path: Path) -> None:
    """The committed repo token parses to a real imprint — parser proof."""
    repo_tsr = (
        Path(__file__).resolve().parents[3] / TIMESTAMPS_DIR / "quality__epoch_heads.json.tsr"
    )
    if not repo_tsr.is_file():
        pytest.skip("no committed token on this branch")
    imprint = extract_imprint(repo_tsr.read_bytes())
    assert len(imprint) == 32
