"""Fisher-Rao geometry on the univariate-Gaussian model manifold.

The Fisher metric maps (mu, sigma) to the hyperbolic half-plane
(mu/sqrt(2), sigma): d_FR = sqrt(2) * acosh(1 + dz^2/(2 s1 s2)) with
dz^2 = ((mu1-mu2)^2 + 2(s1-s2)^2)/2. Bench: triangle inequality on three
planted points and monotonicity of distance in the mean gap (vs Euclidean
coordinate distance as the honest-negative comparison).
"""

import numpy as np

from quant_fund.models._ig_synth import fr_gauss_pair


def _fr(p: tuple[float, float], q: tuple[float, float]) -> float:
    m1, s1 = p
    m2, s2 = q
    z = ((m1 - m2) ** 2 + 2.0 * (s1 - s2) ** 2) / 2.0
    return float(np.sqrt(2.0) * np.arccosh(1.0 + z / (2.0 * s1 * s2)))


def bench_fisher_rao(seed: int = 4301) -> dict[str, float]:
    del seed
    p, q = fr_gauss_pair()
    mid = ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)
    d_pq = _fr(p, q)
    tri_gap = d_pq - (_fr(p, mid) + _fr(mid, q)) + 1e-12
    # monotonicity: shift mean of q further away
    q_far = (q[0] + 2.0, q[1])
    d_far = _fr(p, q_far)
    euc_pq = float(np.hypot(p[0] - q[0], p[1] - q[1]))
    euc_far = float(np.hypot(p[0] - q_far[0], p[1] - q_far[1]))
    return {
        "synthetic_fr_dist": d_pq,
        "synthetic_fr_tri_slack": float(tri_gap),
        "synthetic_fr_far_dist": d_far,
        "synthetic_fr_monotone": float(d_far > d_pq),
        "synthetic_fr_euc_ratio": d_pq / euc_pq,
        "synthetic_fr_euc_far_ratio": d_far / euc_far,
    }
