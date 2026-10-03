"""Noise-IK-lite handshake (wave 292).

Toy WireGuard Noise_IK: ECDH-on-toy-group chaining —
ck = H(ck, dh); initiator/responder derive identical session keys; a
tampered ephemeral share desynchronizes them.
"""

import hashlib

_SEED = 20261231 + 839

_P = 2**255 - 19


def _dh(priv: int, pub: int) -> int:
    return pow(pub, priv, _P)


def _h(*parts: bytes) -> bytes:
    hh = hashlib.blake2s()
    for p in parts:
        hh.update(p)
    return hh.digest()


def ik_handshake() -> tuple[bytes, bytes]:
    g = 9
    init_s, resp_s = 0xA5, 0x5A
    init_e, resp_e = 0x11, 0x22
    pub_is, pub_rs = _dh(init_s, g), _dh(resp_s, g)
    pub_ie, pub_re = _dh(init_e, g), _dh(resp_e, g)
    ck = _h(b"Noise_IK")
    # initiator: mix hash, DH(e, rs), DH(s, rs) on receive of re
    ck_i = _h(ck, pub_ie.to_bytes(32, "little"))
    ck_i = _h(ck_i, _dh(init_e, pub_rs).to_bytes(32, "little"))
    ck_i = _h(ck_i, pub_re.to_bytes(32, "little"))
    ck_i = _h(ck_i, _dh(init_e, pub_re).to_bytes(32, "little"))
    ck_i = _h(ck_i, _dh(init_s, pub_re).to_bytes(32, "little"))
    # responder
    # responder mirrors: DH(e_i, s_r), then DH(re, ie), then DH(re, s_i)
    ck_r = _h(ck, pub_ie.to_bytes(32, "little"))
    ck_r = _h(ck_r, _dh(resp_s, pub_ie).to_bytes(32, "little"))
    ck_r = _h(ck_r, pub_re.to_bytes(32, "little"))
    ck_r = _h(ck_r, _dh(resp_e, pub_ie).to_bytes(32, "little"))
    ck_r = _h(ck_r, _dh(resp_e, pub_is).to_bytes(32, "little"))
    return ck_i, ck_r


def bench_wg_ik(seed: int = _SEED) -> dict[str, float]:
    ki, kr = ik_handshake()
    return {"synthetic_wg_ik": float(ki == kr and len(ki) == 32)}
