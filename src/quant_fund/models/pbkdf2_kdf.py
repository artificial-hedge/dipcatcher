"""PBKDF2-HMAC-SHA256 (toy) vs hashlib oracle + domain separation."""

import hashlib
import hmac as _hmac

import numpy as np

_SEED = 20261231 + 565


def pbkdf2(pw: bytes, salt: bytes, iters: int, dklen: int = 32) -> bytes:
    out = b""
    i = 1
    while len(out) < dklen:
        u = _hmac.new(pw, salt + i.to_bytes(4, "big"), hashlib.sha256).digest()
        t = u
        for _ in range(iters - 1):
            u = _hmac.new(pw, u, hashlib.sha256).digest()
            t = bytes(a ^ b for a, b in zip(t, u, strict=True))
        out += t
        i += 1
    return out[:dklen]


def bench_pbkdf2_kdf(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    n = 40
    match = diff = 0
    for _ in range(n):
        pw = bytes(rng.randint(0, 256) for _ in range(rng.randint(4, 16)))
        salt = bytes(rng.randint(0, 256) for _ in range(16))
        it = rng.randint(1, 50)
        mine = pbkdf2(pw, salt, it)
        ref = hashlib.pbkdf2_hmac("sha256", pw, salt, it)
        match += int(mine == ref)
        salt2 = bytes([salt[0] ^ 1]) + salt[1:]
        diff += int(pbkdf2(pw, salt2, it) != mine)
    return {
        "synthetic_matches_stdlib": float(match / n),
        "synthetic_salt_domain_sep": float(diff / n),
    }
