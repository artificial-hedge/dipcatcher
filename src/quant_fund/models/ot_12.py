"""1-out-of-2 oblivious transfer (toy Even-Goldreich-Lempel over prime group) (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 695

_P = 786433
_G = 10


def ot_12(m0: int, m1: int, b: int, rng: np.random.RandomState) -> tuple[int, bool]:
    """Return (received_message, sender_learned_b)."""
    # receiver keypair
    x = int(rng.randint(2, _P - 2))
    pk_b = pow(_G, x, _P)
    pk_notb = pow(_G, int(rng.randint(2, _P - 2)), _P)  # receiver doesn't know dlog
    pks = [pk_b, pk_notb] if b == 0 else [pk_notb, pk_b]
    # sender encrypts both under the two pks
    r0, r1 = int(rng.randint(2, _P - 2)), int(rng.randint(2, _P - 2))
    e0 = (pow(_G, r0, _P), (m0 * pow(pks[0], r0, _P)) % _P)
    e1 = (pow(_G, r1, _P), (m1 * pow(pks[1], r1, _P)) % _P)
    # receiver can only decrypt its chosen pk
    c = e0 if b == 0 else e1
    got = (c[1] * pow(pow(c[0], x, _P), -1, _P)) % _P
    return got, False


def bench_ot_12(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 30
    for _ in range(trials):
        m0 = int(rng.randint(2, 1000))
        m1 = int(rng.randint(2, 1000))
        b = int(rng.randint(0, 2))
        got, leaked = ot_12(m0, m1, b, rng)
        ok += float(got == (m0 if b == 0 else m1) and not leaked)
    return {"synthetic_ot_correct": ok / trials}
