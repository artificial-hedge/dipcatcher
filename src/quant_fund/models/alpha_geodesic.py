"""Alpha-connection geodesic interpolation on the multinomial manifold (SYNTHETIC).

The alpha-family geodesic between p and q at parameter t is
g(t) propto (p^{(1-a)/2})^{1-t} (q^{(1-a)/2})^t projected back (the
alpha=1 exponential mixture gives the e-geodesic, alpha=-1 the m-geodesic).
Bench: endpoint exactness and the midpoint's KL asymmetry vs the plain
Euclidean mixture (geometric vs arithmetic interpolation).
"""

import numpy as np

from quant_fund.models._ig_synth import multinomial_pair


def _geo(p: np.ndarray, q: np.ndarray, t: float, alpha: float = 1.0) -> np.ndarray:
    b = (1.0 - alpha) / 2.0
    if abs(b) < 1e-12:
        r = p ** (1.0 - t) * q**t
    else:
        w = p**b * (1.0 - t) + q**b * t
        r = w ** (1.0 / b)
    return r / r.sum()


def _kl(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(a * np.log(a / b)))


def bench_alpha_geodesic(seed: int = 4309) -> dict[str, float]:
    del seed
    p, q = multinomial_pair()
    g0, g1 = _geo(p, q, 0.0), _geo(p, q, 1.0)
    g_mid = _geo(p, q, 0.5)
    m_mid = 0.5 * (p + q)
    return {
        "synthetic_alpha_end0_err": float(np.max(np.abs(g0 - p))),
        "synthetic_alpha_end1_err": float(np.max(np.abs(g1 - q))),
        "synthetic_alpha_mid_kl_pq": _kl(p, g_mid),
        "synthetic_alpha_mid_kl_qp": _kl(q, g_mid),
        "synthetic_alpha_mix_kl_pq": _kl(p, m_mid),
        "synthetic_alpha_mix_kl_qp": _kl(q, m_mid),
    }
