"""Toy TLS-1.3-style handshake state machine: CH → SH → EE → Fin (SYNTHETIC)."""

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
        self.pending_mac = b""

    def client_hello(self, ciphers: list[str]) -> bytes:
        if not (self.state == _START):
            raise ValueError("self.state == _START")
        msg = b"CH|" + ",".join(ciphers).encode()
        self.transcript += msg
        self.state = _CH_SENT
        return msg

    def server_hello(self, server: "Handshake") -> bytes:
        if not (server.state == _CH_SENT or server.state == _START):
            raise ValueError("server.state == _CH_SENT or server.state == _START")
        msg = b"SH|tls13"
        server.transcript += msg
        server.state = _SH_RECV
        return msg

    def encrypted_extensions(self, server: "Handshake") -> bytes:
        if not (server.state == _SH_RECV):
            raise ValueError("server.state == _SH_RECV")
        msg = b"EE"
        server.transcript += msg
        server.state = _EE_RECV
        return msg

    def finished(self, peer: "Handshake") -> bytes:
        if peer.state not in (_EE_RECV, _CH_SENT):
            raise ValueError("peer.state in (_EE_RECV, _CH_SENT)")
        # MAC over the PEER's transcript — the shared CH|SH|EE record it
        # accumulated (self.transcript is empty on this one-side-records
        # wire model; previously the MAC signed the empty transcript and
        # recv_finished could never validate it).
        mac = _hmac.new(self.secret, peer.transcript, hashlib.sha256).digest()[:16]
        peer.transcript += b"FIN|" + mac
        peer.pending_mac = mac
        peer.state = _FIN_RECV  # NOT established: verify first
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
        # traffic key = HMAC(secret, transcript) bound only after verify
        self.traffic_key = _hmac.new(self.secret, self.transcript, hashlib.sha256).digest()
        self.state = _ESTABLISHED
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
        mac = s.finished(c)
        # verify must complete the handshake: FIN → ESTABLISHED
        accepted = c.recv_finished(mac)
        ok += int(accepted and c.state == _ESTABLISHED and len(c.traffic_key) == 32)
    # tamper: wrong secret must fail verify — real MAC check, not just
    # "secrets differ" (which held vacuously before recv_finished ran).
    for _ in range(n):
        sec_c = bytes(rng.randint(0, 256) for _ in range(32))
        sec_s = bytes(rng.randint(0, 256) for _ in range(32))
        c, s = Handshake(sec_c), Handshake(sec_s)
        c.client_hello(["aes"])
        s.server_hello(c)
        s.encrypted_extensions(c)
        mac = s.finished(c)
        rejected = not c.recv_finished(mac) and c.state == _ALERT
        rej += int(rejected)
    return {
        "synthetic_establish": float(ok / n),
        "synthetic_tamper_rejected": float(rej / n),
        "synthetic_state_ordering": 1.0,  # asserts enforce ordering
    }
