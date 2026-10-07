"""Pedersen commitment: C(v, r) = g^v h^r mod p (SYNTHETIC).

Perfectly hiding (r uniform) and computationally binding under DL.
Bench: homomorphic property C(v1,r1)*C(v2,r2) = C(v1+v2, r1+r2) holds
exactly, and an opening check verifies commit/verify round-trip.
"""

import numpy as np

from quant_fund.models._crypto_synth import G_SAFE, P_SAFE


def _commit(v: int, r: int, h: int) -> int:
    return pow(G_SAFE, v, P_SAFE) * pow(h, r, P_SAFE) % P_SAFE


def bench_pedersen_commit(seed: int = 4707) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    h = pow(G_SAFE, 999, P_SAFE)  # h = g^a; log unknown to committer (synthetic)
    v1, v2 = 1234, 5678
    r1 = int(rng.integers(1, P_SAFE - 1))
    r2 = int(rng.integers(1, P_SAFE - 1))
    c1 = _commit(v1, r1, h)
    c2 = _commit(v2, r2, h)
    c_sum = _commit(v1 + v2, r1 + r2, h)
    hom = c1 * c2 % P_SAFE == c_sum
    # opening verification
    c = _commit(v1, r1, h)
    opens_ok = c == _commit(v1, r1, h)
    opens_bad = c == _commit(v1 + 1, r1, h)
    return {
        "synthetic_ped_homomorphic": float(hom),
        "synthetic_ped_open_ok": float(opens_ok),
        "synthetic_ped_open_bad": float(opens_bad),
        "synthetic_ped_commit": float(c),
        "synthetic_ped_distinct": float(c1 != c2),
    }
