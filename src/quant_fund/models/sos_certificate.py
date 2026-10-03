"""Sum-of-squares certificate for a nonnegative univariate polynomial.

For p >= 0 on R, complex roots come in conjugate pairs; taking one
root from each pair gives q with p = |q|^2 = Re(q)^2 + Im(q)^2 — an
explicit SOS decomposition. Bench builds the certificate for a degree-6
p, verifies p - (u^2 + v^2) vanishes, and reports the residual norm.
"""

import numpy as np


def _sos(coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """coeffs: leading-first np.poly1d style. Returns (u, v, residual)."""
    roots = np.roots(coeffs)
    # take roots with positive imaginary part + real roots come in
    # even multiplicity -> split them evenly
    lead = coeffs[0]
    upper = [r for r in roots if r.imag > 1e-9]
    real_roots = sorted(r.real for r in roots if abs(r.imag) <= 1e-9)
    # each real root should appear with even multiplicity
    kept = []
    i = 0
    while i < len(real_roots):
        j = i
        while j + 1 < len(real_roots) and abs(real_roots[j + 1] - real_roots[i]) < 1e-6:
            j += 1
        mult = j - i + 1
        kept.extend([real_roots[i]] * (mult // 2))
        i = j + 1
    q_roots = np.array(upper + [complex(r, 0.0) for r in kept])
    q = np.poly(q_roots) * np.sqrt(lead)
    # q has complex coeffs; u = Re q, v = Im q (descending order)
    u = np.real(q)
    v = np.imag(q)
    rec = np.convolve(u, u) + np.convolve(v, v)
    rec = np.trim_zeros(rec, "f")
    p_desc = np.trim_zeros(coeffs, "f")
    d = max(len(rec), len(p_desc))
    rec = np.pad(rec, (d - len(rec), 0))
    p_desc = np.pad(p_desc, (d - len(p_desc), 0))
    resid = float(np.max(np.abs(p_desc - rec)))
    return np.asarray(u), np.asarray(v), resid


def bench_sos_certificate(seed: int = 5305) -> dict[str, float]:
    # p(x) = x^6 - 2x^4 + 3x^2 + 2  (strictly positive)
    p = np.array([1.0, 0.0, -2.0, 0.0, 3.0, 0.0, 2.0])
    u, v, resid = _sos(p)
    xs = np.linspace(-2, 2, 41)
    pmin = float(np.min(np.polyval(p, xs)))
    sos_min = float(np.min(np.polyval(u, xs) ** 2 + np.polyval(v, xs) ** 2))
    return {
        "synthetic_sos_residual": resid,
        "synthetic_sos_p_min": pmin,
        "synthetic_sos_rec_min": sos_min,
        "synthetic_sos_deg_u": float(len(u) - 1),
    }
