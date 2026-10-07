"""Equivalence testing via two one-sided tests (TOST).

Schuirmann (1987) two one-sided tests for mean
equivalence: ``H0: |delta| >= theta`` rejected when both
one-sided p-values fall below alpha — equivalently the
(1-2alpha) CI lies inside (-theta, theta). Welch and
paired variants plus a Fisher-z correlation variant.

Honesty: the bench simulates equivalent and shifted
normal draws and requires the TOST p-value to separate
them — a pure frequentist-operating-characteristic check,
not an empirical claim.

References
----------
* Schuirmann, D.J. (1987) "A comparison of the two
  one-sided tests procedure...", J. Pharmacokinet.
  Biopharm. 15, 657-680.
* Lakens, D. (2017) "Equivalence tests: a practical
  primer", Social Psych. & Personality Sci. 8, 355-362.
* Goertzen & Cribbie (2010) correlation equivalence via
  Fisher z, Br. J. Math. Stat. Psychol. 63.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def tost_means(
    x: FloatArray,
    y: FloatArray | None = None,
    theta: float = 0.5,
    alpha: float = 0.05,
    paired: bool = False,
) -> dict[str, float]:
    """TOST for mean equivalence (Welch or paired).

    ``x`` alone => one-sample vs 0; ``x, y`` => two-sample.
    ``theta`` is the equivalence bound in raw units.
    Returns ``p_tost``, the 90% CI half-width, and the
    equivalence decision bit.
    """
    xx = np.asarray(x, dtype=np.float64).ravel()
    if paired:
        if y is None:
            raise ValueError("paired requires y")
        yy = np.asarray(y, dtype=np.float64).ravel()
        if yy.size != xx.size or xx.size < 3:
            raise ValueError("bad paired shapes")
        d = xx - yy
        n = d.size
        dm = float(d.mean())
        sd = float(d.std(ddof=1))
        se = sd / math.sqrt(n)
        df = n - 1.0
        t1 = (dm + theta) / se
        t2 = (dm - theta) / se
        p1 = 1.0 - stats.t.cdf(t1, df)  # H0: delta <= -theta
        p2 = float(stats.t.cdf(t2, df))  # H0: delta >= +theta
        ci_hw = float(stats.t.ppf(1 - alpha, df)) * se
        delta = dm
    else:
        n1 = xx.size
        if n1 < 3:
            raise ValueError("need n>=3")
        m1 = float(xx.mean())
        v1 = float(xx.var(ddof=1))
        if y is None:
            se = math.sqrt(v1 / n1)
            df = n1 - 1.0
            delta = m1
        else:
            yy = np.asarray(y, dtype=np.float64).ravel()
            n2 = yy.size
            if n2 < 3:
                raise ValueError("need n>=3")
            m2 = float(yy.mean())
            v2 = float(yy.var(ddof=1))
            se = math.sqrt(v1 / n1 + v2 / n2)
            df = (v1 / n1 + v2 / n2) ** 2 / (
                v1 * v1 / (n1 * n1 * (n1 - 1)) + v2 * v2 / (n2 * n2 * (n2 - 1))
            )
            delta = m1 - m2
        if se <= 0:
            raise ValueError("zero SE")
        t1 = (delta + theta) / se
        t2 = (delta - theta) / se
        p1 = 1.0 - stats.t.cdf(t1, df)
        p2 = float(stats.t.cdf(t2, df))
        ci_hw = float(stats.t.ppf(1 - alpha, df)) * se
    p_tost = float(max(p1, p2))
    lo, hi = delta - ci_hw, delta + ci_hw
    equiv = 1.0 if (lo > -theta and hi < theta) else 0.0
    return {
        "p_tost": p_tost,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "equivalent": equiv,
    }


def tost_correlation(
    r: float,
    n: int,
    theta: float = 0.3,
    alpha: float = 0.05,
) -> dict[str, float]:
    """TOST on Fisher-z transformed correlation.

    Tests |z(r)| inside the z(theta) bound with se = 1/sqrt(n-3).
    """
    if not (-0.999 < r < 0.999 and n > 4 and theta > 0):
        raise ValueError("bad inputs")
    z = math.atanh(r)
    zb = math.atanh(theta)
    se = 1.0 / math.sqrt(n - 3)
    z1 = (z + zb) / se
    z2 = (z - zb) / se
    p1 = 1.0 - stats.norm.cdf(z1)
    p2 = float(stats.norm.cdf(z2))
    p_tost = float(max(p1, p2))
    hw = float(stats.norm.ppf(1 - alpha)) * se
    lo, hi = math.tanh(z - hw), math.tanh(z + hw)
    return {
        "p_tost": p_tost,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "equivalent": 1.0 if (lo > -theta and hi < theta) else 0.0,
    }


def tost_ratio(
    x: FloatArray,
    y: FloatArray,
    theta: float = 0.25,
    alpha: float = 0.05,
) -> dict[str, float]:
    """TOST on log-ratio of means (bioequivalence style).

    Bounds on log(x_bar/y_bar) inside (-theta, theta).
    """
    xx = np.asarray(x, dtype=np.float64).ravel()
    yy = np.asarray(y, dtype=np.float64).ravel()
    if xx.size < 3 or yy.size < 3 or (xx <= 0).any() or (yy <= 0).any():
        raise ValueError("need positive samples")
    lx, ly = np.log(xx), np.log(yy)
    return tost_means(lx, ly, theta=theta, alpha=alpha)


def bench_tost(seed: int = 20261231 + 464) -> dict[str, float]:
    """SYNTHETIC check — equivalent draws accepted, shifted rejected."""
    rng = np.random.default_rng(seed)
    n = 200
    theta = 0.4
    # equivalent pair (tiny true diff)
    a = rng.normal(size=n)
    b = rng.normal(loc=0.05, size=n)
    eq = tost_means(a, b, theta=theta)
    # clearly non-equivalent pair
    c = rng.normal(loc=1.5, size=n)
    ne = tost_means(a, c, theta=theta)
    if not (eq["p_tost"] < 0.05 and eq["equivalent"] == 1.0):
        raise ValueError(f"tost eq off: {eq}")
    if not (ne["p_tost"] > 0.2 and ne["equivalent"] == 0.0):
        raise ValueError(f"tost neq off: {ne}")
    cr = tost_correlation(0.02, 400, theta=0.25)
    if cr["equivalent"] != 1.0:
        raise ValueError(f"tost corr off: {cr}")
    return {
        "synthetic_eq_p": float(eq["p_tost"]),
        "synthetic_neq_p": float(ne["p_tost"]),
        "synthetic_score": 1.0,
    }
