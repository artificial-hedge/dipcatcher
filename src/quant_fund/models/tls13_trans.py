"""TLS 1.3 transcript + HKDF key schedule (wave 292).

Transcript-hash driven key schedule: early → handshake → master
secrets via toy HKDF-Extract/Expand over the transcript digest; server
and client derive identical keys and Finished MACs verify.
"""

import hashlib
import hmac as _hmac

_SEED = 20261231 + 837


def _hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    return _hmac.new(salt, ikm, hashlib.sha256).digest()


def _hkdf_expand(prk: bytes, info: bytes, n: int = 32) -> bytes:
    out, t, i = b"", b"", 1
    while len(out) < n:
        t = _hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        out += t
        i += 1
    return out[:n]


def handshake_messages() -> tuple[bytes, bytes, bytes]:
    ch = hashlib.sha256(b"client_hello:random").digest()
    sh = hashlib.sha256(b"server_hello:random:share").digest()
    fin = hashlib.sha256(b"encrypted_extensions:cert:cv").digest()
    return ch, sh, fin


def derive(ch: bytes, sh: bytes, fin: bytes, shared: bytes) -> tuple[bytes, bytes]:
    early = _hkdf_extract(b"\x00" * 32, b"\x00" * 32)
    hs_secret = _hkdf_extract(early, shared)
    c_hs = _hkdf_expand(hs_secret, hashlib.sha256(ch + sh).digest() + b"c hs traffic")
    s_hs = _hkdf_expand(hs_secret, hashlib.sha256(ch + sh).digest() + b"s hs traffic")
    master = _hkdf_extract(hs_secret, b"\x00" * 32)
    c_ap = _hkdf_expand(master, hashlib.sha256(ch + sh + fin).digest() + b"c ap traffic")
    _s_ap = _hkdf_expand(master, hashlib.sha256(ch + sh + fin).digest() + b"s ap traffic")
    return c_hs + s_hs, c_ap + _s_ap


def finished_mac(hs_key: bytes, transcript: bytes) -> bytes:
    return _hmac.new(_hkdf_expand(hs_key, b"finished"), transcript, hashlib.sha256).digest()


def bench_tls13_trans(seed: int = _SEED) -> dict[str, float]:
    ch, sh, fin = handshake_messages()
    shared = hashlib.sha256(b"ecdhe_shared_secret").digest()
    hs_c, ap_c = derive(ch, sh, fin, shared)
    hs_s, ap_s = derive(ch, sh, fin, shared)
    ok = int(hs_c == hs_s and ap_c == ap_s and hs_c != ap_c)
    transcript = ch + sh + fin
    mac = finished_mac(hs_s[:32], transcript)
    ok += int(mac == finished_mac(hs_c[:32], transcript))
    # tampered transcript changes MAC
    ok += int(finished_mac(hs_c[:32], transcript + b"!") != mac)
    return {"synthetic_tls13": float(ok == 3)}
