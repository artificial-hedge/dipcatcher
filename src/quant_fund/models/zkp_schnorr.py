"""SYNTHETIC Schnorr identification / Sigma protocol.

Group: subgroup of Z*_p with generator g of order q (toy safe-prime).
Completeness: honest prover always accepted.
Special soundness: two transcripts (t, c1, s1), (t, c2, s2) extract w.
Honest-verifier ZK: simulator output is indistinguishable by transcript
equation (simulated (t, c, s) passes verification without knowing w).
"""

from __future__ import annotations

import random


def _params() -> tuple[int, int, int]:
    # toy safe prime: p = 2q+1 with q prime; pick small primes
    for q in (1401223, 122063, 2999779, 7919, 104729):
        p = 2 * q + 1
        if all(p % d for d in range(3, int(p**0.5) + 1, 2)):
            # find generator of order-q subgroup: g = h^2 mod p for h != 1
            for h in range(2, 30):
                g = h * h % p
                if g != 1 and pow(g, q, p) == 1:
                    return p, q, g
    return 251, 5, 6  # unreachable for listed q


P, Q, G = _params()


def bench_zkp_schnorr(seed: int = 20261231 + 424) -> dict[str, float]:
    rng = random.Random(seed)
    complete = sound = zk = 0
    trials = 25
    for _ in range(trials):
        w = rng.randrange(1, Q)
        pub = pow(G, w, P)
        # honest run: t = g^r, c, s = r + c*w
        r = rng.randrange(1, Q)
        t = pow(G, r, P)
        c = rng.randrange(1, Q)
        s = (r + c * w) % Q
        complete += int(pow(G, s, P) == t * pow(pub, c, P) % P)
        # soundness: second challenge on same commitment t
        c2 = (c + rng.randrange(1, Q - 1)) % Q or 1
        if c2 != c:
            s2 = (r + c2 * w) % Q
            w_ext = (s - s2) * pow(c - c2, -1, Q) % Q
            sound += int(pow(G, w_ext, P) == pub)
        else:
            sound += 1
        # honest-verifier simulation: pick s,c then t = g^s * pub^-c — no w needed
        c3 = rng.randrange(1, Q)
        s3 = rng.randrange(1, Q)
        t3 = pow(G, s3, P) * pow(pub, -c3, P) % P
        zk += int(pow(G, s3, P) == t3 * pow(pub, c3, P) % P)
    return {
        "synthetic_completeness": float(complete / trials),
        "synthetic_special_soundness": float(sound / trials),
        "synthetic_simulator_verifies": float(zk / trials),
    }
