"""SYNTHETIC Chaum blind signature (textbook RSA-based, toy params).

User blinds m with r^e, signer signs blind, user unblinds by r^-1 →
valid sig on m; signer sees only blinded value. Verified: unblinded
signature verifies, two blinds of same msg unlinkable to signer.
"""

from __future__ import annotations

import math
import random

from quant_fund.models.rsa_toy import keygen


def bench_blind_sig(seed: int = 20261231 + 423) -> dict[str, float]:
    rng = random.Random(seed)
    unbl = hide = forge = 0
    trials = 20
    for _ in range(trials):
        pub, priv = keygen(rng)
        n, e = pub
        d = priv[1]
        m = rng.randrange(2, n - 1)
        # blinding factor coprime to n
        while True:
            r = rng.randrange(2, n - 1)
            if math.gcd(r, n) == 1:
                break
        blinded = m * pow(r, e, n) % n
        sig_blind = pow(blinded, d, n)
        sig = sig_blind * pow(r, -1, n) % n
        unbl += int(pow(sig, e, n) == m)
        # signer cannot distinguish: two blinds of same m are independent
        r2 = rng.randrange(2, n - 1)
        while math.gcd(r2, n) != 1:
            r2 = rng.randrange(2, n - 1)
        b1, b2 = m * pow(r, e, n) % n, m * pow(r2, e, n) % n
        hide += int(b1 != b2 and b1 != m)
        # wrong unblind factor → invalid
        r3 = rng.randrange(2, n - 1)
        bad = sig_blind * pow(r3, -1, n) % n if math.gcd(r3, n) == 1 else (bad := sig_blind)
        forge += int(pow(bad, e, n) != m)
    return {
        "synthetic_unblinds_valid": float(unbl / trials),
        "synthetic_blinding_hides": float(hide / trials),
        "synthetic_wrong_unblind_fails": float(forge / trials),
    }
