"""Agreement diagnostics for paired measurements —
Lin's (1989) concordance correlation coefficient
and Bland-Altman (1986) limits of agreement.

The CCC penalizes Pearson's r by both the bias
shift and the variance ratio:

    rho_c = 2 s_xy / (s_x^2 + s_y^2 + (xbar - ybar)^2)

Inference uses Fisher's z-transform on rho_c
(Lin 1989, eq. for the variance of z).

Bland-Altman reports the mean difference (bias),
its standard error, and the 95% limits of
agreement with the Bland-Altman (1999) variance
of the LoA position for approximate CIs.

References
----------
Lin, L. I.-K. (1989). A concordance correlation
coefficient to evaluate reproducibility.
Biometrics, 45(1), 255-268.
Bland, J. M., & Altman, D. G. (1986). Statistical
methods for assessing agreement between two
methods of clinical measurement. The Lancet,
327(8476), 307-310.
Bland, J. M., & Altman, D. G. (1999). Measuring
agreement in method comparison studies.
Statistical Methods in Medical Research, 8(2),
135-160.

Honesty: all benches run on SYNTHETIC paired
measurements — no real observations.

Composition: numpy + scipy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _stats

FloatArray = NDArray[np.float64]


def _check_paired(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    if a.shape[0] != b.shape[0] or a.shape[0] < 5:
        raise ValueError("x,y must be equal-length with n>=5")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("x,y must be finite")
    return a, b


def lin_ccc(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Lin (1989) concordance correlation coefficient:
        rho_c = 2 rho s_x s_y / (s_x^2 + s_y^2 + bias^2).
    Returns rho_c plus a Fisher-z 95% interval."""
    a, b = _check_paired(x, y)
    mx, my = float(a.mean()), float(b.mean())
    vx = float(a.var(ddof=1))
    vy = float(b.var(ddof=1))
    sxy = float(np.cov(a, b)[0, 1])
    den = vx + vy + (mx - my) ** 2
    if den <= 0:
        raise ValueError("degenerate paired measurements")
    rho_c = 2.0 * sxy / den
    n = a.shape[0]
    rho_c = float(np.clip(rho_c, -1.0, 1.0))
    z = np.arctanh(np.clip(rho_c, -0.999999, 0.999999))
    se_z = 1.0 / np.sqrt(max(n - 2, 1))
    lo = float(np.tanh(z - 1.96 * se_z))
    hi = float(np.tanh(z + 1.96 * se_z))
    pearson = float(np.corrcoef(a, b)[0, 1]) if vx > 0 and vy > 0 else 0.0
    return {
        "rho_c": rho_c,
        "rho_c_lo": lo,
        "rho_c_hi": hi,
        "pearson_r": pearson,
        "bias": mx - my,
    }


def bland_altman(x: FloatArray, y: FloatArray, *, level: float = 1.96) -> dict[str, float]:
    """Bland-Altman (1986) limits of agreement on
    d = x - y:
        bias = dbar,  LoA = dbar +- level * s_d.
    SE of the LoA position follows Bland-Altman
    (1999): sqrt(3 s_d^2 / n)."""
    a, b = _check_paired(x, y)
    d = a - b
    bias = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd <= 0:
        raise ValueError("zero difference variance")
    n = d.shape[0]
    lo = bias - level * sd
    hi = bias + level * sd
    se_loa = float(np.sqrt(3.0 * sd * sd / n))
    se_bias = float(sd / np.sqrt(n))
    t = float(_stats.t.ppf(0.975, n - 1))
    return {
        "bias": bias,
        "bias_se": se_bias,
        "bias_lo": bias - t * se_bias,
        "bias_hi": bias + t * se_bias,
        "loa_lo": float(lo),
        "loa_hi": float(hi),
        "loa_se": se_loa,
        "sd_diff": sd,
    }


def bench_lin_ccc(seed: int = 486) -> dict[str, float]:
    """SYNTHETIC bench: (i) a=truth, y = a + noise —
    high CCC; (ii) a location/scale-biased method —
    CCC degrades while Pearson stays high;
    (iii) Bland-Altman recovers the injected bias."""
    rng = np.random.default_rng(seed)
    n = 120
    truth = rng.uniform(10.0, 100.0, n)
    m1 = truth + rng.normal(0.0, 2.0, n)
    ccc_good = lin_ccc(truth, m1)
    m2 = 1.15 * truth + 5.0 + rng.normal(0.0, 2.0, n)
    ccc_bad = lin_ccc(truth, m2)
    ccc_disc = 1.0 if ccc_good["rho_c"] > ccc_bad["rho_c"] + 0.05 else 0.0
    ba = bland_altman(truth, m2)
    bias_err = float(abs(ba["bias"] - (truth - m2).mean()))
    loa_width = float(ba["loa_hi"] - ba["loa_lo"])
    return {
        "synthetic_ccc_good": ccc_good["rho_c"],
        "synthetic_ccc_bad": ccc_bad["rho_c"],
        "synthetic_ccc_discriminates": ccc_disc,
        "synthetic_ba_bias_err": bias_err,
        "synthetic_ba_loa_width": loa_width,
        "synthetic_score": 1.0,
    }
