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


def _pending_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, bytes]:
    """Repo with a stamped pending anchor; returns (root, digest)."""
    import quant_fund.research.ots_anchor as mod

    root = _repo_with_target(tmp_path)
    target_bytes = (root / "quality/epoch_heads.json").read_bytes()
    digest = hashlib.sha256(target_bytes).digest()

    class _Resp:
        def __init__(self, body: bytes) -> None:
            self._b = body

        def read(self) -> bytes:
            return self._b

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(
        mod.urllib.request, "urlopen", lambda req, timeout=None: _Resp(CALENDAR_RESP)
    )
    stamp_ots("quality/epoch_heads.json", root=root, calendars=("https://x",))
    return root, digest


def test_upgrade_ots_confirms_and_commits_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quant_fund.research.ots_anchor as mod

    root, digest = _pending_repo(tmp_path, monkeypatch)
    # Upgraded stream: append<x> → sha256 → bitcoin attestation at 650000.
    upgraded = (
        b"\xf0\x20"
        + b"\xab" * 32
        + b"\x08"
        + b"\x00"
        + ATT_BITCOIN
        + b"\x03"
        + bytes([0x90, 0xD6, 0x27])
    )
    header = bytearray(80)
    header[72:76] = (0x2200FFFF).to_bytes(4, "little")
    monkeypatch.setattr(mod, "_get", lambda url, timeout: upgraded)
    monkeypatch.setattr(mod, "_fetch_header", lambda h, *, explorer, timeout: bytes(header))
    res = mod.upgrade_ots(root=root)
    assert res["ok"]
    assert res["upgraded"] == ["quality__epoch_heads.json.ots"]
    assert res["anchors"]["quality/epoch_heads.json"] == ["upgraded:650000"]
    # The .ots now verifies as pow_verified against the committed header.
    out = verify_ots(root)
    assert out["attestations"]["quality/epoch_heads.json"] == ["bitcoin:650000:pow_verified"]
    # Re-upgrade is a no-op on a bitcoin-bearing proof.
    res2 = mod.upgrade_ots(root=root)
    assert res2["anchors"]["quality/epoch_heads.json"] == ["already_bitcoin"]


def test_upgrade_ots_still_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import quant_fund.research.ots_anchor as mod

    root, _ = _pending_repo(tmp_path, monkeypatch)
    # Calendar answers but hasn't confirmed — same pending stream back.
    monkeypatch.setattr(mod, "_get", lambda url, timeout: CALENDAR_RESP)
    res = mod.upgrade_ots(root=root)
    assert res["ok"] and res["upgraded"] == []
    assert res["anchors"]["quality/epoch_heads.json"] == ["no_bitcoin_attestation"]

    # Dead calendar → still_pending, no error.
    def _dead(url: str, timeout: float) -> bytes:
        raise OSError("offline")

    monkeypatch.setattr(mod, "_get", _dead)
    res = mod.upgrade_ots(root=root)
    assert res["anchors"]["quality/epoch_heads.json"] == ["still_pending"]
    # Malformed upgrade → recorded, token untouched.
    monkeypatch.setattr(mod, "_get", lambda url, timeout: b"\xff\xff\xff")
    res = mod.upgrade_ots(root=root)
    assert any(
        s.startswith("upgrade_malformed") for s in res["anchors"]["quality/epoch_heads.json"]
    )


def _fake_block(commitment: bytes) -> tuple[bytes, list[str], bytes]:
    """A synthetic 3-tx block: coinbase carries OP_RETURN <commitment>."""
    script_pub = b"\x6a\x20" + commitment
    coinbase = (
        b"\x01\x00\x00\x00"  # version
        + b"\x01"  # 1 vin
        + b"\x00" * 32
        + b"\xff\xff\xff\xff"  # prevhash + vout
        + b"\x04"  # scriptSig len
        + b"\x03\x10\xeb\x09"  # BIP34 push: height 650000 (0x09EB10) LE
        + b"\xff\xff\xff\xff"  # seq
        + b"\x02"  # 2 vout
        + (1000).to_bytes(8, "little")
        + b"\x19"
        + b"\x76\xa9\x14"
        + b"\x11" * 20
        + b"\x88\xac"
        + (0).to_bytes(8, "little")
        + bytes([len(script_pub)])
        + script_pub
        + b"\x00\x00\x00\x00"  # locktime
    )
    txid0 = hashlib.sha256(hashlib.sha256(coinbase).digest()).digest()
    txids_internal = [txid0, b"\x22" * 32, b"\x33" * 32]

    # merkle: h01 = d(t0+t1), h22 = d(t2+t2), root = d(h01+h22)
    def d(b: bytes) -> bytes:
        return hashlib.sha256(hashlib.sha256(b).digest()).digest()

    root = d(d(txids_internal[0] + txids_internal[1]) + d(txids_internal[2] + txids_internal[2]))
    header = bytearray(80)
    header[36:68] = root
    header[72:76] = (0x2200FFFF).to_bytes(4, "little")
    txids_display = [t[::-1].hex() for t in txids_internal]
    return bytes(header), txids_display, coinbase


def test_verify_block_inclusion_full(tmp_path: Path) -> None:
    from quant_fund.research.ots_anchor import (
        coinbase_commitments,
        verify_block_inclusion,
    )

    commitment = b"\x5a" * 32
    header, txids, coinbase = _fake_block(commitment)
    res = verify_block_inclusion(header, txids, coinbase, commitment)
    assert res["ok"] and res["n_tx"] == 3
    # each layer fails independently
    bad = verify_block_inclusion(header, txids, coinbase, b"\x99" * 32)
    assert bad["error"] == "commitment_absent"
    wrong_root = verify_block_inclusion(
        header[:36] + b"\x00" * 36 + header[72:], txids, coinbase, commitment
    )
    assert wrong_root["error"] == "block_merkle_mismatch"
    other_coinbase = coinbase.replace(b"\x6a\x20" + commitment, b"\x6a\x20" + b"\x77" * 32)
    res2 = verify_block_inclusion(header, txids, other_coinbase, commitment)
    assert res2["error"] == "coinbase_txid_mismatch"
    assert commitment in coinbase_commitments(coinbase)


def test_verify_block_inclusion_segwit_coinbase() -> None:
    """Segwit coinbase: txid must hash the witness-stripped form — hashing
    raw bytes yields the wtxid, which never equals txids[0] (the regression
    this test pins: every modern OTS-bearing block has a segwit coinbase)."""
    from quant_fund.research.ots_anchor import (
        coinbase_commitments,
        coinbase_height,
        verify_block_inclusion,
    )

    commitment = b"\x5a" * 32
    script_pub = b"\x6a\x20" + commitment
    base = (
        b"\x01\x00\x00\x00"  # version
        + b"\x01"  # 1 vin
        + b"\x00" * 32
        + b"\xff\xff\xff\xff"  # prevhash + vout
        + b"\x04"  # scriptSig len
        + b"\x03\xa0\xbb\x0d"  # BIP34 push: height 900000 (0x0DBBA0) LE
        + b"\xff\xff\xff\xff"  # seq
        + b"\x01"  # 1 vout
        + (0).to_bytes(8, "little")
        + bytes([len(script_pub)])
        + script_pub
        + b"\x00\x00\x00\x00"  # locktime
    )
    segwit = (
        base[:4]
        + b"\x00\x01"  # marker + flag
        + base[4:-4]
        + b"\x01"  # witness: 1 item on vin[0]
        + b"\x20"
        + b"\x99" * 32
        + base[-4:]
    )

    def d(b: bytes) -> bytes:
        return hashlib.sha256(hashlib.sha256(b).digest()).digest()

    txid0 = d(base)  # txid = hash of the stripped form
    assert d(segwit) != txid0  # raw hash is the wtxid — the trap
    header = bytearray(80)
    header[36:68] = txid0  # single-tx block: root is the coinbase txid
    res = verify_block_inclusion(bytes(header), [txid0[::-1].hex()], segwit, commitment)
    assert res["ok"], res
    assert commitment in coinbase_commitments(segwit)
    # BIP34: the scriptsig push declares height 900000 — the claimed-height
    # binding must accept it and reject any other.
    assert coinbase_height(segwit) == 900000
    ok_h = verify_block_inclusion(
        bytes(header), [txid0[::-1].hex()], segwit, commitment, claimed_height=900000
    )
    assert ok_h["ok"], ok_h
    bad_h = verify_block_inclusion(
        bytes(header), [txid0[::-1].hex()], segwit, commitment, claimed_height=900001
    )
    assert bad_h["error"] == "coinbase_height_mismatch"


def test_verify_ots_fully_verified(tmp_path: Path) -> None:
    """End-to-end: .blk witness file + hdr → fully_verified state."""
    root = _repo_with_target(tmp_path)
    target = root / "quality/epoch_heads.json"
    digest = hashlib.sha256(target.read_bytes()).digest()
    commitment = hashlib.sha256(b"calendar-merkle-root").digest()
    # attestation binds the commitment (the value att["committed_digest"] carries)
    stream = b"\x00" + ATT_BITCOIN + b"\x03" + bytes([0x90, 0xD6, 0x27])
    header, txids, coinbase = _fake_block(commitment)
    ots_dir = root / "quality/timestamps/ots"
    ots_dir.mkdir(parents=True)
    # craft: bare bitcoin attestation at node level — committed_digest == digest
    # itself (no ops), so we need the commitment to be the attestation's bound
    # digest: use a one-op chain prepend<pad> so committed_digest = pad+digest?
    # Simplest: commitment = what parse yields: att binds msg=digest (no ops).
    # So set commitment := digest by rebuilding the block with it.
    header, txids, coinbase = _fake_block(digest)
    (ots_dir / "quality__epoch_heads.json.ots").write_bytes(OTS_MAGIC + b"\x08" + digest + stream)
    (ots_dir / "quality__epoch_heads.json.650000.hdr").write_bytes(header)
    (ots_dir / "quality__epoch_heads.json.650000.blk").write_text(
        json.dumps({"txids": txids, "coinbase": coinbase.hex()})
    )
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
    assert res["attestations"]["quality/epoch_heads.json"] == ["bitcoin:650000:fully_verified"]
    # corrupt the witness → inclusion_invalid hard error
    (ots_dir / "quality__epoch_heads.json.650000.blk").write_text(
        json.dumps({"txids": txids, "coinbase": (b"\x00" + coinbase[1:]).hex()})
    )
    res = verify_ots(root)
    assert not res["ok"]
    assert any("inclusion_invalid" in e for e in res["errors"])


REPO_ROOT = Path(__file__).resolve().parents[3]
OTS_AUDITOR = REPO_ROOT / "scripts/verify_ots_auditor.py"


def _run_auditor(*args: str) -> tuple[int, str]:
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, str(OTS_AUDITOR), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc.returncode, proc.stdout + proc.stderr


@pytest.mark.skipif(not OTS_AUDITOR.is_file(), reason="auditor script absent")
def test_standalone_ots_auditor_agrees(tmp_path: Path) -> None:
    """Independent stdlib implementation must reach the same verdict as the
    library on the committed proof and on each tamper class."""
    root = _repo_with_target(tmp_path)
    target = root / "quality/epoch_heads.json"
    digest = hashlib.sha256(target.read_bytes()).digest()
    ots_dir = root / "quality/timestamps/ots"
    ots_dir.mkdir(parents=True)
    stream = b"\x00" + ATT_BITCOIN + b"\x03" + bytes([0x90, 0xD6, 0x27])
    ots_path = ots_dir / "quality__epoch_heads.json.ots"
    ots_path.write_bytes(OTS_MAGIC + b"\x08" + digest + stream)
    header, txids, coinbase = _fake_block(digest)
    hdr_path = ots_dir / "quality__epoch_heads.json.650000.hdr"
    blk_path = ots_dir / "quality__epoch_heads.json.650000.blk"
    hdr_path.write_bytes(header)
    blk_path.write_text(json.dumps({"txids": txids, "coinbase": coinbase.hex()}))
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

    lib = verify_ots(root)
    rc, out = _run_auditor(
        "--target",
        str(target),
        "--ots",
        str(ots_path),
        "--hdr",
        str(hdr_path),
        "--blk",
        str(blk_path),
    )
    assert lib["ok"] and rc == 0, out
    assert "fully_verified" in out and "fully_verified" in str(
        lib["attestations"]["quality/epoch_heads.json"]
    )

    # Drift: target byte differs post-stamp → both report STALE (the proof
    # remains a valid timestamp of the superseded bytes, not a failure)
    target.write_text('{"pins":2}')
    rc2, out2 = _run_auditor("--target", str(target), "--ots", str(ots_path))
    assert rc2 == 0 and "stale" in out2.lower()
    lib2 = verify_ots(root)
    assert lib2["ok"] and lib2["fresh"]["quality/epoch_heads.json"] is False

    # Tamper: corrupt coinbase → both flag inclusion
    target.write_text('{"pins":1}')
    blk_path.write_text(json.dumps({"txids": txids, "coinbase": (b"\x02" + coinbase[1:]).hex()}))
    rc3, out3 = _run_auditor(
        "--target",
        str(target),
        "--ots",
        str(ots_path),
        "--hdr",
        str(hdr_path),
        "--blk",
        str(blk_path),
    )
    assert rc3 == 1
    lib3 = verify_ots(root)
    assert not lib3["ok"] and any("inclusion_invalid" in e for e in lib3["errors"])

    # Tamper: header PoW fails → both flag
    blk_path.write_text(json.dumps({"txids": txids, "coinbase": coinbase.hex()}))
    bad_hdr = bytearray(header)
    bad_hdr[72:76] = (0x01000001).to_bytes(4, "little")
    hdr_path.write_bytes(bytes(bad_hdr))
    rc4, _ = _run_auditor(
        "--target",
        str(target),
        "--ots",
        str(ots_path),
        "--hdr",
        str(hdr_path),
        "--blk",
        str(blk_path),
    )
    assert rc4 == 1
    lib4 = verify_ots(root)
    assert not lib4["ok"] and any("pow_invalid" in e for e in lib4["errors"])

    # Committed real proof: standalone auditor parses it without the library
    real = REPO_ROOT / "quality/timestamps/ots/quality__epoch_heads.json.ots"
    if real.is_file():
        rc5, out5 = _run_auditor(
            "--target",
            str(REPO_ROOT / "quality/epoch_heads.json"),
            "--ots",
            str(real),
        )
        assert rc5 == 0 and "pending attestation" in out5


def _successor_headers(header: bytes, k: int) -> list[bytes]:
    """K headers each linking prev-hash, easy PoW."""
    out: list[bytes] = []
    prev = header
    for i in range(k):
        nxt = bytearray(80)
        nxt[4:36] = hashlib.sha256(hashlib.sha256(prev).digest()).digest()
        nxt[36:68] = bytes([0x40 + i]) * 32
        nxt[72:76] = (0x2200FFFF).to_bytes(4, "little")
        out.append(bytes(nxt))
        prev = bytes(nxt)
    return out


def test_verify_burial(tmp_path: Path) -> None:
    from quant_fund.research.ots_anchor import verify_burial

    header, _, _ = _fake_block(b"\x5a" * 32)
    succ = _successor_headers(header, 3)
    res = verify_burial(header, succ)
    # toy-easy nBits → tiny expected-work per block: log2 negative is honest
    assert res["ok"] and res["depth"] == 3 and isinstance(res["work_log2"], float)
    # chain break
    bad = list(succ)
    bad[1] = bad[1][:4] + b"\x00" * 32 + bad[1][36:]
    assert verify_burial(header, bad)["error"] == "chain_break:2"
    # bad successor PoW
    bad2 = list(succ)
    b = bytearray(bad2[0])
    b[72:76] = (0x01000001).to_bytes(4, "little")
    bad2[0] = bytes(b)
    assert verify_burial(header, bad2)["error"] == "succ_pow_invalid:1"


def test_verify_ots_buried(tmp_path: Path) -> None:
    root = _repo_with_target(tmp_path)
    target = root / "quality/epoch_heads.json"
    digest = hashlib.sha256(target.read_bytes()).digest()
    header, txids, coinbase = _fake_block(digest)
    succ = _successor_headers(header, 2)
    ots_dir = root / "quality/timestamps/ots"
    ots_dir.mkdir(parents=True)
    (ots_dir / "quality__epoch_heads.json.ots").write_bytes(
        OTS_MAGIC + b"\x08" + digest + b"\x00" + ATT_BITCOIN + b"\x03" + bytes([0x90, 0xD6, 0x27])
    )
    (ots_dir / "quality__epoch_heads.json.650000.hdr").write_bytes(header)
    (ots_dir / "quality__epoch_heads.json.650000.blk").write_text(
        json.dumps(
            {
                "txids": txids,
                "coinbase": coinbase.hex(),
                "succ_headers": [s.hex() for s in succ],
            }
        )
    )
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
    states = res["attestations"]["quality/epoch_heads.json"]
    assert "bitcoin:650000:fully_verified" in states
    assert any(s.startswith("buried:2:work_log2=") for s in states)
    # corrupt one successor → hard error
    (ots_dir / "quality__epoch_heads.json.650000.blk").write_text(
        json.dumps(
            {
                "txids": txids,
                "coinbase": coinbase.hex(),
                "succ_headers": [succ[0].hex(), succ[1].hex()[:70] + "ff" * 5],
            }
        )
    )
    res = verify_ots(root)
    assert not res["ok"] and any("burial_invalid" in e for e in res["errors"])
