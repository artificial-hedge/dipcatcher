"""Lo-MacKinlay variance-ratio test for random walks.

Under an uncorrelated-increment null, Var(r_t + ... + r_{t+q-1})
= q·Var(r_t). The variance-ratio VR(q) = σ̂²_q/σ̂²_1 detects
positive or negative autocorrelation that single-lag tests
miss; the heteroskedasticity-robust z*(q) keeps the test valid
under changing volatility.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure random-walk deviation
detection on generated series — never market evidence.

References:
- Lo, A. W., MacKinlay, A. C. (1988). Stock market prices do
  not follow random walks: evidence from a simple specification
  test. *Review of Financial Studies* 1, 41-66 — VR statistic,
  homoskedastic z and heteroskedastic-robust z*.
- Lo, A. W., MacKinlay, A. C. (1989). The size and power of the
  variance ratio test. *J. Econometrics* 40.
- Chow, K. V., Denning, K. C. (1993). A simple multiple variance
  ratio test. *J. Econometrics* 58 — joint VR over horizons.
- Wright, J. H. (2000). Alternative variance-ratio tests using
  ranks and signs. *J. Business & Economic Statistics* 18.

Composition: pure numpy + scipy — overlapping q-period returns,
θ(q) bias correction, delta-method SEs; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def variance_ratio(
    r: FloatArray,
    q: int = 8,
) -> dict[str, float]:
    """Lo-MacKinlay VR(q) on a log-return series.

    Returns the ratio, both z-statistics, and per-lag
    autocorrelation diagnostics."""
    rr = np.asarray(r, dtype=np.float64).ravel()
    t = rr.size
    if t < 100:
        raise ValueError("T>=100")
    if not (2 <= q <= min(32, t // 5)):
        raise ValueError("q in 2..T/5")
    if not np.all(np.isfinite(rr)):
        raise ValueError("finite inputs required")
    if np.std(rr) < 1e-12:
        raise ValueError("returns must vary")

    mu = float(np.mean(rr))
    x = rr - mu
    var1 = float(np.sum(x**2) / (t - 1))
    if var1 < 1e-300:
        raise ValueError("degenerate variance")
    # q-period overlapping sums
    nq = t - q + 1
    rq = np.convolve(x, np.ones(q))[q - 1 : q - 1 + nq]
    m = q * (t - q + 1) * (1 - q / t)
    varq = float(np.sum(rq**2) / m)
    vr = varq / var1

    # homoskedastic asymptotic variance of VR−1 (Lo-MacK 1988)
    phi = 2 * (2 * q - 1) * (q - 1) / (3 * q * t)
    z = (vr - 1.0) / np.sqrt(phi)

    # heteroskedastic-robust: delta_j weights on squared x's
    num = 0.0
    for j in range(1, q):
        wj = (1 - j / q) ** 2
        aj = x[j:] * x[:-j]
        delta_j = float(np.sum(aj**2)) / (var1**2 * (t - j))
        num += wj * delta_j
    var_rob = 4 * num / t if num > 0 else phi
    z_star = (vr - 1.0) / np.sqrt(max(var_rob, 1e-300))

    rho1 = float(np.corrcoef(rr[1:], rr[:-1])[0, 1])
    return {
        "t": float(t),
        "q": float(q),
        "vr": vr,
        "z_homosk": float(z),
        "p_homosk": float(2 * (1 - stats.norm.cdf(abs(z)))),
        "z_heterosk": float(z_star),
        "p_heterosk": float(2 * (1 - stats.norm.cdf(abs(z_star)))),
        "rho1": rho1,
    }


def synth_vr(
    t: int = 800,
    kind: str = "iid",
    ar: float = 0.25,
    seed: int = 0,
) -> FloatArray:
    """Return DGPs: iid (VR≈1), AR(1)+ (VR>1), MA(1)− (VR<1)."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, 1.0, t)
    if kind == "iid":
        return e
    if kind == "ar_pos":
        y = np.zeros(t)
        for i in range(1, t):
            y[i] = ar * y[i - 1] + e[i]
        return y
    # ma_neg: mean-reverting returns r_t = e_t − θ e_{t−1}
    return e - ar * np.concatenate([[0.0], e[:-1]])


def bench_variance_ratio(seed: int = 20261231 + 239) -> dict[str, float]:
    """Variance-ratio self-check: AR+ shows VR>1 and rejects,
    MA− shows VR<1, iid keeps VR≈1 and fails to reject.
    All ``synthetic_*``."""
    rp = synth_vr(kind="ar_pos", ar=0.3, seed=seed)
    outp = variance_ratio(rp, q=8)
    rn = synth_vr(kind="ma_neg", ar=0.4, seed=seed + 1)
    outn = variance_ratio(rn, q=8)
    ri = synth_vr(kind="iid", seed=seed + 2)
    outi = variance_ratio(ri, q=8)
    outp_b = variance_ratio(rp, q=8)

    vr_p = float(outp["vr"])
    return {
        "synthetic_vr_pos": vr_p,
        "synthetic_p_pos": float(outp["p_heterosk"]),
        "synthetic_vr_neg": float(outn["vr"]),
        "synthetic_p_neg": float(outn["p_heterosk"]),
        "synthetic_vr_iid": float(outi["vr"]),
        "synthetic_p_iid": float(outi["p_heterosk"]),
        "synthetic_rho1_pos": float(outp["rho1"]),
        "synthetic_detects": float(
            vr_p > 1.1
            and float(outn["vr"]) < 0.9
            and float(outp["p_heterosk"]) < 0.05
            and float(outi["p_heterosk"]) > 0.05
            and abs(float(outi["vr"]) - 1.0) < 0.2
        ),
        "synthetic_determinism": float(vr_p == float(outp_b["vr"])),
    }
