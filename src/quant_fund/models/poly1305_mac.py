"""Poly1305 one-time authenticator mod 2^130-5 (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 697

_P130 = (1 << 130) - 5


def poly1305(msg: bytes, r: int, s: int) -> int:
    acc = 0
    for i in range(0, len(msg), 16):
        chunk = msg[i : i + 16]
        n = int.from_bytes(chunk, "little") + (1 << (8 * len(chunk)))
        acc = ((acc + n) * r) % _P130
    return (acc + s) % (1 << 128)


def bench_poly1305_mac(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 30
    for _ in range(trials):
        msg = bytes(rng.randint(0, 256, int(rng.randint(1, 64))))
        r = int(rng.randint(1, (1 << 60) - 1)) | (int(rng.randint(1, (1 << 60))) << 62)
        s = int(rng.randint(1, (1 << 60) - 1)) | (int(rng.randint(1, (1 << 60))) << 62)
        t1 = poly1305(msg, r, s)
        ok += float(t1 == poly1305(msg, r, s))
        # tampered message should give different tag (w.h.p.)
        if len(msg) > 0:
            bad = bytearray(msg)
            bad[0] ^= 1
            ok += float(poly1305(bytes(bad), r, s) != t1)
    denom = trials * 2
    return {"synthetic_poly_det_and_detect": ok / denom}
