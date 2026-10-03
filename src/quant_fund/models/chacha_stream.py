"""ChaCha20 core function: quarter-round diffusion + keystream determinism."""

import numpy as np

_SEED = 20261231 + 696

_MASK = 0xFFFFFFFF


def _qr(s: np.ndarray, a: int, b: int, c: int, d: int) -> None:
    s[a] = (s[a] + s[b]) & _MASK
    s[d] ^= s[a]
    s[d] = ((s[d] << 16) | (s[d] >> 16)) & _MASK
    s[c] = (s[c] + s[d]) & _MASK
    s[b] ^= s[c]
    s[b] = ((s[b] << 12) | (s[b] >> 20)) & _MASK
    s[a] = (s[a] + s[b]) & _MASK
    s[d] ^= s[a]
    s[d] = ((s[d] << 8) | (s[d] >> 24)) & _MASK
    s[c] = (s[c] + s[d]) & _MASK
    s[b] ^= s[c]
    s[b] = ((s[b] << 7) | (s[b] >> 25)) & _MASK


def block(key: np.ndarray, counter: int, nonce: int) -> np.ndarray:
    """One 64-byte block via 20 double-rounds."""
    consts = np.array([0x61707865, 0x3320646E, 0x79622D32, 0x6B206574], dtype=np.uint32)
    state = np.concatenate([consts, key, np.array([counter, nonce, 0, 0], dtype=np.uint32)]).copy()
    work = state.astype(np.int64) & _MASK
    for _ in range(10):
        _qr(work, 0, 4, 8, 12)
        _qr(work, 1, 5, 9, 13)
        _qr(work, 2, 6, 10, 14)
        _qr(work, 3, 7, 11, 15)
        _qr(work, 0, 5, 10, 15)
        _qr(work, 1, 6, 11, 12)
        _qr(work, 2, 7, 8, 13)
        _qr(work, 3, 4, 9, 14)
    out = (work + (state.astype(np.int64) & _MASK)) & _MASK
    return out.astype(np.uint32)


def bench_chacha_stream(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 30
    for _ in range(trials):
        key = rng.randint(0, 2**32 - 1, 8).astype(np.uint32)
        b1 = block(key, 0, 7)
        b2 = block(key, 0, 7)
        b3 = block(key, 1, 7)
        ok += float(np.array_equal(b1, b2) and not np.array_equal(b1, b3))
    return {"synthetic_chacha_deterministic": ok / trials}
