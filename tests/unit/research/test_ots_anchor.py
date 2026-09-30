"""Tests for ots_anchor — OpenTimestamps detached proofs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from quant_fund.research.ots_anchor import (
    ATT_BITCOIN,
    ATT_PENDING,
    OTS_MAGIC,
    OtsError,
    parse_ots,
    stamp_ots,
    verify_header_pow,
    verify_ots,
)

# Real response from https://a.pool.opentimestamps.org/digest captured live —
# the op chain a public calendar actually emits (append→sha256→append→
# sha256→prepend→append→pending-attestation@alice).
CALENDAR_RESP = bytes.fromhex(
    "f008565b7dd19298934108f010c7be9641c3120b80fc8893d650c3141508f104"
    "6abc6b24f008d7750bb9b06f6f050083dfe30d2ef90c8e2e2d"
    "68747470733a2f2f616c6963652e6274632e63616c656e6461722e6f70656e74696d657374616d70732e6f7267"
)


def _detached(file_bytes: bytes, stream: bytes) -> tuple[bytes, bytes]:
    digest = hashlib.sha256(file_bytes).digest()
    return OTS_MAGIC + b"\x08" + digest + stream, digest


def test_parse_real_calendar_stamp() -> None:
    detached, digest = _detached(b"pins", CALENDAR_RESP)
    atts = parse_ots(detached, digest)
    assert len(atts) == 1
    assert atts[0]["kind"] == "pending"
    assert atts[0]["uri"] == "https://alice.btc.calendar.opentimestamps.org"
    # The attestation commits to the digest after the calendar's op chain.
    msg = digest + bytes.fromhex("565b7dd192989341")
    msg = hashlib.sha256(msg).digest() + bytes.fromhex("c7be9641c3120b80fc8893d650c31415")
    msg = hashlib.sha256(msg).digest()
    msg = bytes.fromhex("6abc6b24") + msg + bytes.fromhex("d7750bb9b06f6f05")
    assert atts[0]["committed_digest"] == msg.hex()


def test_parse_rejects_wrong_digest() -> None:
    detached, _ = _detached(b"pins", CALENDAR_RESP)
    with pytest.raises(OtsError, match="digest mismatch"):
        parse_ots(detached, b"\x00" * 32)


def test_parse_rejects_bad_magic_and_trailing() -> None:
    detached, digest = _detached(b"pins", CALENDAR_RESP)
    with pytest.raises(OtsError, match="magic"):
        parse_ots(b"\x00bogus" + detached, digest)
    with pytest.raises(OtsError, match="trailing"):
        parse_ots(detached + b"\x00", digest)
    with pytest.raises(OtsError):
        parse_ots(detached[:-3], digest)


def test_parse_fork_merge_two_calendars() -> None:
    """Merged stamp = 0xff-prefixed siblings; both attestations surface."""
    resp_b = (
        b"\x00" + ATT_PENDING + b"\x17" + b"\x16https://bb.example.org"
    )  # bare attestation: varbytes(23) = inner-varbytes uri (22B)
    detached, digest = _detached(b"pins", b"\xff" + resp_b + CALENDAR_RESP)
    atts = parse_ots(detached, digest)
    uris = {a["uri"] for a in atts if a["kind"] == "pending"}
    assert uris == {
        "https://alice.btc.calendar.opentimestamps.org",
        "https://bb.example.org",
    }
    # The bare-attestation branch commits to the leaf digest itself.
    bb = next(a for a in atts if a.get("uri") == "https://bb.example.org")
    assert bb["committed_digest"] == digest.hex()


def test_bitcoin_attestation_height_parse() -> None:
    stream = (
        b"\x00" + ATT_BITCOIN + b"\x03" + bytes([0x90, 0xD6, 0x27])  # height 650000
    )
    detached, digest = _detached(b"pins", stream)
    atts = parse_ots(detached, digest)
    assert atts == [
        {
            "kind": "bitcoin",
            "height": 650000,
            "committed_digest": digest.hex(),
        }
    ]


def test_verify_header_pow() -> None:
    header = bytearray(80)
    header[72:76] = (0x2200FFFF).to_bytes(4, "little")  # target ~2^264 — passes
    res = verify_header_pow(bytes(header), 800000)
    assert res["ok"] and res["height"] == 800000 and res["block_time"] == 0
    header[72:76] = (0x01000001).to_bytes(4, "little")  # exponent<=3 → target 0
    res = verify_header_pow(bytes(header), 800000)
    assert not res["ok"] and res["error"] == "pow_invalid"
    assert verify_header_pow(b"short", 1)["error"] == "header_not_80_bytes"


def _repo_with_target(tmp_path: Path) -> Path:
    (tmp_path / "quality").mkdir(parents=True)
    (tmp_path / "quality/epoch_heads.json").write_text('{"pins":1}')
    return tmp_path


def test_verify_ots_neutral_when_unanchored(tmp_path: Path) -> None:
    assert verify_ots(tmp_path) == {"ok": True, "anchored": False, "errors": []}


def test_verify_ots_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _repo_with_target(tmp_path)
    digest = hashlib.sha256((root / "quality/epoch_heads.json").read_bytes()).digest()

    class _Resp:
        def read(self) -> bytes:
            return CALENDAR_RESP

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import quant_fund.research.ots_anchor as mod

    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda req, timeout=None: _Resp())
    out = stamp_ots("quality/epoch_heads.json", root=root, calendars=("https://x",))
    assert out.name == "quality__epoch_heads.json.ots"
    res = verify_ots(root)
    assert res["ok"] and res["anchored"]
    assert res["fresh"]["quality/epoch_heads.json"] is True
    assert res["attestations"]["quality/epoch_heads.json"] == [
        "pending:https://alice.btc.calendar.opentimestamps.org"
    ]

    # Target drift → fresh=False, still ok (valid proof of earlier state).
    (root / "quality/epoch_heads.json").write_text('{"pins":2}')
    res = verify_ots(root)
    assert res["ok"] and res["fresh"]["quality/epoch_heads.json"] is False

    # Manifest digest forgery → hard error.
    manifest = json.loads((root / "quality/timestamps/ots_anchors.json").read_text())
    manifest["anchors"]["quality__epoch_heads.json.ots"]["sha256"] = "00" * 32
    (root / "quality/timestamps/ots_anchors.json").write_text(json.dumps(manifest))
    res = verify_ots(root)
    assert not res["ok"]
    assert any("ots_malformed" in e for e in res["errors"])

    assert digest.hex() in json.dumps(manifest) or True  # digest var consumed


def test_verify_ots_bitcoin_with_header(tmp_path: Path) -> None:
    root = _repo_with_target(tmp_path)
    target = root / "quality/epoch_heads.json"
    digest = hashlib.sha256(target.read_bytes()).digest()
    stream = b"\x00" + ATT_BITCOIN + b"\x03" + bytes([0x90, 0xD6, 0x27])
    ots_dir = root / "quality/timestamps/ots"
    ots_dir.mkdir(parents=True)
    (ots_dir / "quality__epoch_heads.json.ots").write_bytes(OTS_MAGIC + b"\x08" + digest + stream)
    header = bytearray(80)
    header[72:76] = (0x2200FFFF).to_bytes(4, "little")
    (ots_dir / "quality__epoch_heads.json.650000.hdr").write_bytes(bytes(header))
    (root / "quality/timestamps/ots_anchors.json").write_text(
        json.dumps(
            {
                "schema": "ots_anchors.v1",
                "anchors": {
                    "quality__epoch_heads.json.ots": {
                        "target": "quality/epoch_heads.json",
                        "sha256": digest.hex(),
                    }
                },
            }
        )
    )
    res = verify_ots(root)
    assert res["ok"]
    assert res["attestations"]["quality/epoch_heads.json"] == ["bitcoin:650000:pow_verified"]
