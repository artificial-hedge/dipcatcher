"""Adversarial probes for tls_handshake: FIN must gate ESTABLISHED on a real
transcript-bound MAC verify — not auto-promote, not sign an empty transcript."""

from quant_fund.models.tls_handshake import (
    _ALERT,
    _ESTABLISHED,
    _FIN_RECV,
    Handshake,
    bench_tls_handshake,
)


def _drive(c: Handshake, s: Handshake) -> bytes:
    c.client_hello(["aes"])
    s.server_hello(c)
    s.encrypted_extensions(c)
    return s.finished(c)


def test_finished_leaves_peer_awaiting_verify() -> None:
    """finished() must NOT auto-promote to ESTABLISHED — verify gates it."""
    c, s = Handshake(b"k" * 32), Handshake(b"k" * 32)
    _drive(c, s)
    assert c.state == _FIN_RECV
    assert c.traffic_key == b""  # no key bound before verification


def test_recv_finished_verifies_and_binds_key() -> None:
    c, s = Handshake(b"k" * 32), Handshake(b"k" * 32)
    mac = _drive(c, s)
    assert c.recv_finished(mac)
    assert c.state == _ESTABLISHED
    assert len(c.traffic_key) == 32


def test_wrong_secret_rejects() -> None:
    """Differing secrets → MAC mismatch → ALERT (was vacuous: never verified)."""
    c, s = Handshake(b"c" * 32), Handshake(b"s" * 32)
    mac = _drive(c, s)
    assert not c.recv_finished(mac)
    assert c.state == _ALERT
    assert c.traffic_key == b""


def test_forged_mac_rejected() -> None:
    c, s = Handshake(b"k" * 32), Handshake(b"k" * 32)
    _drive(c, s)
    assert not c.recv_finished(b"\x00" * 16)
    assert c.state == _ALERT


def test_mac_binds_real_transcript() -> None:
    """The MAC must cover the peer's CH|SH|EE transcript — the old code signed
    the sender's *empty* transcript, so a one-byte tamper goes unnoticed."""
    import hashlib
    import hmac as _hmac

    c, s = Handshake(b"k" * 32), Handshake(b"k" * 32)
    mac = _drive(c, s)
    expected = _hmac.new(b"k" * 32, b"CH|aesSH|tls13EE", hashlib.sha256).digest()[:16]
    assert mac == expected  # fails if the MAC was over b""


def test_bench() -> None:
    out = bench_tls_handshake()
    assert out["synthetic_establish"] == 1.0
    assert out["synthetic_tamper_rejected"] == 1.0
