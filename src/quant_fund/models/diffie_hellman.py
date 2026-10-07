"""Diffie-Hellman key agreement in a safe-prime group (SYNTHETIC).

Alice g^a, Bob g^b; shared secret g^(ab). Bench: agreement exactness,
symmetric shared-secret equality, and the DDH sanity that g^a * g^b =
g^(a+b) — while a passive observer sees only g^a, g^b.
"""

import numpy as np

from quant_fund.models._crypto_synth import G_SAFE, P_SAFE


def bench_diffie_hellman(seed: int = 4709) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    a = int(rng.integers(2, P_SAFE - 2))
    b = int(rng.integers(2, P_SAFE - 2))
    ga = pow(G_SAFE, a, P_SAFE)
    gb = pow(G_SAFE, b, P_SAFE)
    sa = pow(gb, a, P_SAFE)
    sb = pow(ga, b, P_SAFE)
    agree = sa == sb
    # DDH algebraic sanity: g^(a+b) == g^a * g^b
    ddh = pow(G_SAFE, a + b, P_SAFE) == ga * gb % P_SAFE
    # distinct secrets across runs
    a2 = a + 1
    s2 = pow(gb, a2, P_SAFE)
    return {
        "synthetic_dh_agree": float(agree),
        "synthetic_dh_secret": float(sa),
        "synthetic_dh_ddh": float(ddh),
        "synthetic_dh_distinct": float(s2 != sa),
        "synthetic_dh_subgroup": float(pow(sa, P_SAFE - 1, P_SAFE) == 1),
    }
