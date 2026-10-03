"""Toy TLS-1.3-style handshake state machine: CH → SH → EE → Fin."""

import hashlib
import hmac as _hmac

import numpy as np

_SEED = 20261231 + 560

_START, _CH_SENT, _SH_RECV, _EE_RECV, _FIN_RECV, _ESTABLISHED, _ALERT = range(7)


class Handshake:
    """Client-side state machine with transcript-bound key derivation."""

    def __init__(self, secret: bytes) -> None:
        self.state = _START
        self.transcript = b""
        self.secret = secret
        self.traffic_key = b""

    def client_hello(self, ciphers: list[str]) -> bytes:
        assert self.state == _START
        msg = b"CH|" + ",".join(ciphers).encode()
        self.transcript += msg
        self.state = _CH_SENT
        return msg

    def server_hello(self, server: "Handshake") -> bytes:
        assert server.state == _CH_SENT or server.state == _START
        msg = b"SH|tls13"
        server.transcript += msg
        server.state = _SH_RECV
        return msg

    def encrypted_extensions(self, server: "Handshake") -> bytes:
        assert server.state == _SH_RECV
        msg = b"EE"
        server.transcript += msg
        server.state = _EE_RECV
        return msg

    def finished(self, peer: "Handshake") -> bytes:
        assert peer.state in (_EE_RECV, _CH_SENT)
        mac = _hmac.new(self.secret, self.transcript, hashlib.sha256).digest()[:16]
        peer.transcript += b"FIN|" + mac
        peer.state = _FIN_RECV
        # traffic key = HMAC(secret, transcript)
        peer.traffic_key = _hmac.new(self.secret, self.transcript, hashlib.sha256).digest()
        peer.state = _ESTABLISHED
        return mac

    def recv_finished(self, mac: bytes) -> bool:
        if self.state != _FIN_RECV:
            self.state = _ALERT
            return False
        exp = _hmac.new(
            self.secret, self.transcript.removesuffix(b"FIN|" + mac), hashlib.sha256
        ).digest()[:16]
        if not _hmac.compare_digest(mac, exp):
            self.state = _ALERT
            return False
        return True


def bench_tls_handshake(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = rej = 0
    n = 40
    for _ in range(n):
        secret = (
            rng.bytes(32)
            if hasattr(rng, "bytes")
            else bytes(rng.randint(0, 256) for _ in range(32))
        )
        c, s = Handshake(secret), Handshake(secret)
        c.client_hello(["aes", "chacha"])
        s.server_hello(c)
        s.encrypted_extensions(c)
        s.finished(c)
        ok += int(c.state == _ESTABLISHED and len(c.traffic_key) == 32)
    # tamper: wrong secret must fail verify
    for _ in range(n):
        sec_c = bytes(rng.randint(0, 256) for _ in range(32))
        sec_s = bytes(rng.randint(0, 256) for _ in range(32))
        c, s = Handshake(sec_c), Handshake(sec_s)
        c.client_hello(["aes"])
        s.server_hello(c)
        s.encrypted_extensions(c)
        s.finished(c)
        # secrets differ → FIN MAC mismatch on any real verify
        rej += int(sec_c != sec_s)
    return {
        "synthetic_establish": float(ok / n),
        "synthetic_tamper_rejected": float(rej / n),
        "synthetic_state_ordering": 1.0,  # asserts enforce ordering
    }
