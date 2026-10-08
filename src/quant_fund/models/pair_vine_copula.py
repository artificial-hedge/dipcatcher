"""Vine copulas (R-vine, C-vine, D-vine) and GAS dynamic copulas (SYNTHETIC).

Regular vine (R-vine) pair-copula constructions for high-dimensional
dependence modelling, plus the Generalized Autoregressive Score (GAS)
dynamic copula for time-varying bivariate dependence.

References
----------
Dissmann, Brechmann, Czado & Kurowicka (2013). "Selecting and estimating
    regular vine copulae and application to financial returns."
    Computational Statistics & Data Analysis 59.
Bedford & Cooke (2002). "Vines — a new graphical model for dependent
    random variables." Annals of Statistics 30.
Aas, Czado, Frigessi & Bakken (2009). "Pair-copula constructions of
    multiple dependence." Insurance: Mathematics & Economics 44.
Creal, Koopman & Lucas (2013). "Generalized autoregressive score models
    with applications." Journal of Applied Econometrics 28.
Patton (2006). "Modelling asymmetric exchange rate dependence."
    International Economic Review 47.

Complexity
----------
vine_fit is O(d^2 * n_families * n) where d is dimension, n is sample
size.  The greedy MST-based structure selector avoids O(d! * d!)
exhaustive enumeration.  max_dim is enforced (default 30).  GAS recursion
is O(T) per pair.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as opt
from scipy import stats as sstats
from scipy.special import gammaln

if TYPE_CHECKING:
    from quant_fund.models.rvine import RVineSpec

Array = NDArray[np.float64]

# ------------------------------------------------------------------ constants

_MAX_DIM = 30

_FAMILY_PALETTE: tuple[str, ...] = (
    "gaussian",
    "t",
    "clayton",
    "gumbel",
    "frank",
    "joe",
)

_ARCHIMEDEAN_SET: frozenset[str] = frozenset({"clayton", "gumbel", "frank", "joe"})
_ELLIPTICAL_SET: frozenset[str] = frozenset({"gaussian", "t"})

_EPS = 1e-10

# ----------------------------------------------------------------- helpers


def _as_finite_matrix(x: Array, *, min_rows: int = 10, min_cols: int = 2) -> Array:
    m = np.asarray(x, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] < min_rows or m.shape[1] < min_cols:
        raise ValueError(f"expected (n>={min_rows}, d>={min_cols}) array, got {m.shape}")
    if not np.isfinite(m).all():
        raise ValueError("input contains non-finite values")
    return m


def _clip(u: Array) -> Array:
    return np.asarray(np.clip(np.asarray(u, dtype=np.float64), _EPS, 1.0 - _EPS), dtype=np.float64)


def _pseudo(x: Array) -> Array:
    m = _as_finite_matrix(x)
    n = m.shape[0]
    return np.column_stack(
        [
            np.asarray(sstats.rankdata(m[:, j]) / (n + 1.0), dtype=np.float64)
            for j in range(m.shape[1])
        ]
    )


def _kendall_tau_pair(u: Array, v: Array) -> float:
    tau = float(sstats.kendalltau(np.asarray(u, dtype=float), np.asarray(v, dtype=float)).statistic)
    if not np.isfinite(tau):
        raise ValueError("Kendall tau is undefined")
    return tau


def _aic(loglik: float, n_params: int) -> float:
    return -2.0 * loglik + 2.0 * n_params


def _bic(loglik: float, n_params: int, n_obs: int) -> float:
    return -2.0 * loglik + float(n_params) * math.log(max(n_obs, 2))


def _rho_from_tau(tau: float) -> float:
    """Elliptical correlation from Kendall tau: rho = sin(pi/2 * tau)."""
    return float(np.clip(math.sin(math.pi * 0.5 * tau), -0.999, 0.999))


def _tau_from_rho(rho: float) -> float:
    return float(2.0 / math.pi * math.asin(rho))


def _finite_param(
    val: float, *, lo: float = -np.inf, hi: float = np.inf, name: str = "param"
) -> float:
    v = float(val)
    if not np.isfinite(v) or v <= lo or v >= hi:
        raise ValueError(f"{name} must be finite and in ({lo}, {hi})")
    return v


# --------------------------------------------------------------- VineMatrix


@dataclass
class VineMatrix:
    """R-vine matrix encoding (Dißmann et al. 2013, §2.1).

    ``matrix`` is a lower-triangular (d × d) array.  ``diagonal[j]``
    names variable j in the conditioning order.  For j < i, ``matrix[i, j]``
    is the condition set variable for the pair-copula at tree level (i-j).

    ``families`` maps ``(tree, edge)`` → family name (one of
    ``_FAMILY_PALETTE``).  ``params`` maps ``(tree, edge)`` → parameter
    dict.

    ``tree_edges`` records edges per tree: ``[(i, j, cond_set), ...]``.
    """

    matrix: Array
    families: dict[tuple[int, int], str]
    params: dict[tuple[int, int], dict[str, float]]
    aic: float = field(default=np.inf)
    bic: float = field(default=np.inf)
    loglik: float = field(default=-np.inf)
    n_params: int = field(default=0)
    tree_edges: list[list[tuple[int, int, tuple[int, ...]]]] = field(default_factory=list)
    # ``ordering[k]`` is the caller's column index that occupies vine position
    # ``k``; None means identity.  ``structure`` is "cvine", "dvine", or
    # "rvine" and tells vine_logpdf/vine_sample how positional ladder columns
    # map onto tree_edges.
    ordering: Array | None = field(default=None)
    structure: str = field(default="rvine")
    # Internal handle produced by ``vine_fit(structure="rvine")``: the fitted
    # edge list + sampling arrays that the generic R-vine paths consume.
    # Hand-built matrices leave it None and keep the legacy warn/degrade
    # behaviour.
    rvine_spec: RVineSpec | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=np.float64)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError("Vine matrix must be square")
        dim = m.shape[0]
        if dim < 2 or dim > _MAX_DIM:
            raise ValueError(f"Vine dimension must be in [2, {_MAX_DIM}]")
        self.matrix = m
        if self.ordering is not None:
            o = np.asarray(self.ordering, dtype=int).ravel()
            if o.shape[0] != dim or sorted(o.tolist()) != list(range(dim)):
                raise ValueError("ordering must be a permutation of 0..dim-1")
            self.ordering = o
        if self.structure not in ("cvine", "dvine", "rvine"):
            raise ValueError(f"unknown vine structure {self.structure}")

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_trees(self) -> int:
        return self.dim - 1

    def pair_copula(self, tree: int, edge: int) -> tuple[str, dict[str, float]]:
        """Return ``(family, param_dict)`` for pair-copula ``(tree, edge)``."""
        key = (tree, edge)
        if key not in self.families:
            raise KeyError(f"no pair-copula at tree={tree}, edge={edge}")
        return self.families[key], self.params[key]


# ---------------------------------------------------- C-vine / D-vine factories


def cvine_structure(dim: int) -> VineMatrix:
    """Build a C-vine (canonical vine) structure matrix.

    In a C-vine, each tree has a root node connected to all others.
    The lower-triangular matrix encodes the non-symmetric conditioning:
    column ``j`` is identified with the root variable of tree ``j``;
    below-diagonal entries repeat the column's root.
    """
    if dim < 2 or dim > _MAX_DIM:
        raise ValueError(f"dim must be in [2, {_MAX_DIM}]")
    matrix = np.zeros((dim, dim), dtype=np.float64)
    for i in range(dim):
        matrix[i, i] = float(dim - i)
        for j in range(i):
            matrix[i, j] = float(dim - j)
    return VineMatrix(
        matrix=matrix,
        families={},
        params={},
        tree_edges=_cvine_edges(dim),
        structure="cvine",
    )


def _cvine_edges(dim: int) -> list[list[tuple[int, int, tuple[int, ...]]]]:
    """Pre-compute C-vine edge list: root = tree_idx, leaves = tree_idx+1..dim-1."""
    edges: list[list[tuple[int, int, tuple[int, ...]]]] = []
    for tree in range(dim - 1):
        level: list[tuple[int, int, tuple[int, ...]]] = []
        root = tree
        cond = tuple(range(tree)) if tree > 0 else ()
        for leaf in range(tree + 1, dim):
            level.append((root, leaf, cond))
        edges.append(level)
    return edges


def dvine_structure(dim: int) -> VineMatrix:
    """Build a D-vine (drawable vine) structure matrix.

    In a D-vine, edges form a path: 1-2-3-...-d.
    """
    if dim < 2 or dim > _MAX_DIM:
        raise ValueError(f"dim must be in [2, {_MAX_DIM}]")
    matrix = np.zeros((dim, dim), dtype=np.float64)
    for i in range(dim):
        matrix[i, i] = float(i + 1)
        for j in range(i):
            matrix[i, j] = float(j + 1)
    return VineMatrix(
        matrix=matrix,
        families={},
        params={},
        tree_edges=_dvine_edges(dim),
        structure="dvine",
    )


def _dvine_edges(dim: int) -> list[list[tuple[int, int, tuple[int, ...]]]]:
    edges: list[list[tuple[int, int, tuple[int, ...]]]] = []
    for tree in range(dim - 1):
        level: list[tuple[int, int, tuple[int, ...]]] = []
        for start in range(dim - tree - 1):
            a = start
            b = start + tree + 1
            # D-vine tree-t edge (a, b) conditions on the nodes between them
            cond = tuple(range(a + 1, b))
            level.append((a, b, cond))
        edges.append(level)
    return edges


# ======================================================= Pair-copula toolkit
# Each family provides: h_function, h_inverse, logpdf, fit.
# We reparametrise here (do NOT import from copula.py / archimedean_extra.py
# for the fit routines — use the math directly to keep the module
# self-contained for the pair-copula layer).


# ------------------------------------------------------------------- Gaussian


def _gaussian_h(u: Array, v: Array, rho: float) -> Array:
    x = sstats.norm.ppf(_clip(u))
    y = sstats.norm.ppf(_clip(v))
    return np.asarray(sstats.norm.cdf((x - rho * y) / math.sqrt(1.0 - rho * rho)), dtype=np.float64)


def _gaussian_hinv(w: Array, v: Array, rho: float) -> Array:
    y = sstats.norm.ppf(_clip(v))
    z = sstats.norm.ppf(_clip(w))
    scale = math.sqrt(1.0 - rho * rho)
    return np.asarray(sstats.norm.cdf(rho * y + z * scale), dtype=np.float64)


def _gaussian_logpdf(u: Array, v: Array, rho: float) -> float:
    uu = _clip(u)
    vv = _clip(v)
    x = sstats.norm.ppf(uu)
    y = sstats.norm.ppf(vv)
    n = x.shape[0]
    r2 = rho * rho
    # Gaussian copula log-density = joint Gaussian log-density - marginal
    # log-densities (the +0.5*(x² + y²) terms restore the marginals)
    q = (x * x - 2.0 * rho * x * y + y * y) / (1.0 - r2)
    return float(-0.5 * n * math.log(1.0 - r2) - 0.5 * np.sum(q) + 0.5 * np.sum(x * x + y * y))


def _gaussian_fit(u: Array, v: Array) -> dict[str, float]:
    tau = _kendall_tau_pair(u, v)
    rho_init = _rho_from_tau(tau)

    def nll(r: float) -> float:
        if not (-0.999 < r < 0.999):
            return 1e12
        return -_gaussian_logpdf(u, v, r)

    _ = nll(rho_init)  # fast path; bail if degeneracy surfaces early
    res = opt.minimize_scalar(nll, bounds=(-0.998, 0.998), method="bounded")
    rho_hat = float(res.x)
    ll = -float(res.fun)
    return {"rho": rho_hat, "loglik": ll, "aic": _aic(ll, 1), "bic": _bic(ll, 1, u.shape[0])}


# ------------------------------------------------------------------------- t


def _t_h(u: Array, v: Array, rho: float, nu: float) -> Array:
    x = sstats.t.ppf(_clip(u), df=nu)
    y = sstats.t.ppf(_clip(v), df=nu)
    scale = np.sqrt((nu + y * y) * (1.0 - rho * rho) / (nu + 1.0))
    return np.asarray(sstats.t.cdf((x - rho * y) / scale, df=nu + 1.0), dtype=np.float64)


def _t_hinv(w: Array, v: Array, rho: float, nu: float) -> Array:
    y = sstats.t.ppf(_clip(v), df=nu)
    z = sstats.t.ppf(_clip(w), df=nu + 1.0)
    scale = np.sqrt((nu + y * y) * (1.0 - rho * rho) / (nu + 1.0))
    return np.asarray(sstats.t.cdf(rho * y + z * scale, df=nu), dtype=np.float64)


def _t_logpdf(u: Array, v: Array, rho: float, nu: float) -> float:
    uu = _clip(u)
    vv = _clip(v)
    x = sstats.t.ppf(uu, df=nu)
    y = sstats.t.ppf(vv, df=nu)
    n = x.shape[0]
    r2 = rho * rho
    # Joint bivariate t log-density
    d = (x * x + y * y - 2.0 * rho * x * y) / ((1.0 - r2) * nu)
    joint = n * (
        gammaln((nu + 2.0) / 2.0)
        - gammaln(nu / 2.0)
        - math.log(nu * math.pi)
        - 0.5 * math.log(1.0 - r2)
    ) - (nu + 2.0) / 2.0 * float(np.sum(np.log1p(d)))
    # Subtract marginal t log-densities for the copula correction
    marg_const = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * math.log(nu * math.pi)
    ll_marg_x = n * marg_const - (nu + 1.0) / 2.0 * float(np.sum(np.log1p(x * x / nu)))
    ll_marg_y = n * marg_const - (nu + 1.0) / 2.0 * float(np.sum(np.log1p(y * y / nu)))
    return float(joint - ll_marg_x - ll_marg_y)


def _t_fit(u: Array, v: Array) -> dict[str, float]:
    tau = _kendall_tau_pair(u, v)
    rho_init = _rho_from_tau(tau)
    n = u.shape[0]
    best = (-np.inf, rho_init, 8.0)

    for nu_val in np.geomspace(3.0, 30.0, 15):
        nu_f = float(nu_val)

        def nll(r: float, nu_fix: float = nu_f) -> float:
            if not (-0.998 < r < 0.998):
                return 1e12
            return -_t_logpdf(u, v, r, nu_fix)

        res = opt.minimize_scalar(nll, bounds=(-0.997, 0.997), method="bounded")
        if -res.fun > best[0]:
            best = (-res.fun, float(res.x), nu_f)

    ll = best[0]
    return {
        "rho": best[1],
        "nu": best[2],
        "loglik": ll,
        "aic": _aic(ll, 2),
        "bic": _bic(ll, 2, n),
    }


# ------------------------------------------------------------------ Clayton


def _clayton_cdf_val(u: float, v: float, theta: float) -> float:
    return float((u ** (-theta) + v ** (-theta) - 1.0) ** (-1.0 / theta))


def _clayton_h(u: Array, v: Array, theta: float) -> Array:
    uu = _clip(u)
    vv = _clip(v)
    c = (uu ** (-theta) + vv ** (-theta) - 1.0) ** (-1.0 / theta)
    return np.asarray(c ** (theta + 1.0) * vv ** (-theta - 1.0), dtype=np.float64)


def _clayton_hinv(w: Array, v: Array, theta: float) -> Array:
    """Inverse h-function for Clayton via root-finding."""
    ww = _clip(w)
    vv = _clip(v)
    out = np.empty(vv.shape[0], dtype=np.float64)
    for i in range(vv.shape[0]):
        wi = float(ww[i])
        vi = float(vv[i])

        def _f(u: float, vi_fix: float = vi, wi_fix: float = wi) -> float:
            return float(_clayton_h(np.array([u]), np.array([vi_fix]), theta)[0] - wi_fix)

        try:
            lo, hi = _EPS, 1.0 - _EPS
            if _f(lo) * _f(hi) >= 0:
                # fallback: clamp
                out[i] = lo if _f(lo) < 0 else hi
                continue
            out[i] = float(opt.brentq(_f, lo, hi, xtol=1e-10))
        except (ValueError, RuntimeError, ArithmeticError):
            out[i] = wi
    return out


def _clayton_logpdf(u: Array, v: Array, theta: float) -> float:
    uu = _clip(u)
    vv = _clip(v)
    n = uu.shape[0]
    # Clayton copula density: c(u,v) = (1+θ)(uv)^{-θ-1}(u^{-θ}+v^{-θ}-1)^{-2-1/θ}
    # log c = log(1+θ) - (θ+1)log(uv) - (2+1/θ)log(u^{-θ}+v^{-θ}-1)
    t1 = math.log(1.0 + theta)
    t2 = (theta + 1.0) * np.log(uu * vv)  # = (θ+1)log(uv) — negative since uv<1
    t3 = (2.0 + 1.0 / theta) * np.log(uu ** (-theta) + vv ** (-theta) - 1.0)
    return float(n * t1 - np.sum(t2) - np.sum(t3))


def _clayton_fit(u: Array, v: Array) -> dict[str, float]:
    tau = _kendall_tau_pair(u, v)
    if tau <= 0.0:
        # fallback: use a very small theta
        pass
    else:
        float(np.clip(2.0 * tau / (1.0 - tau), 0.02, 20.0))
    n = u.shape[0]

    def nll(th: float) -> float:
        if th <= 0.005:
            return 1e12
        val = _clayton_logpdf(u, v, th)
        if not np.isfinite(val):
            return 1e12
        return -val

    res = opt.minimize_scalar(nll, bounds=(0.005, 30.0), method="bounded")
    ll = -float(res.fun)
    return {
        "theta": float(res.x),
        "loglik": ll,
        "aic": _aic(ll, 1),
        "bic": _bic(ll, 1, n),
    }


# ------------------------------------------------------------------- Gumbel


def _gumbel_h(u: Array, v: Array, alpha: float) -> Array:
    uu = _clip(u)
    vv = _clip(v)
    a = (-np.log(uu)) ** alpha
    b = (-np.log(vv)) ** alpha
    s = a + b
    cap = np.exp(-(s ** (1.0 / alpha)))
    return np.asarray(cap * (s ** (1.0 / alpha - 1.0)) * b / (-np.log(vv)) / vv, dtype=np.float64)


def _gumbel_hinv(w: Array, v: Array, alpha: float) -> Array:
    ww = _clip(w)
    vv = _clip(v)
    out = np.empty(vv.shape[0], dtype=np.float64)
    for i in range(vv.shape[0]):
        wi = float(ww[i])
        vi = float(vv[i])

        def _f(u: float, vi_fix: float = vi, wi_fix: float = wi) -> float:
            return float(_gumbel_h(np.array([u]), np.array([vi_fix]), alpha)[0] - wi_fix)

        try:
            lo, hi = _EPS, 1.0 - _EPS
            if _f(lo) * _f(hi) >= 0:
                out[i] = lo if _f(lo) < 0 else hi
                continue
            out[i] = float(opt.brentq(_f, lo, hi, xtol=1e-10))
        except (ValueError, RuntimeError, ArithmeticError):
            out[i] = wi
    return out


def _gumbel_logpdf(u: Array, v: Array, alpha: float) -> float:
    if alpha < 1.0001:
        return 0.0
    uu = _clip(u)
    vv = _clip(v)
    a = (-np.log(uu)) ** alpha
    b = (-np.log(vv)) ** alpha
    s = a + b
    cap = s ** (1.0 / alpha)
    # The Gumbel copula density formula (Schepsmeier & Stöber 2014 eq. 4):
    # c(u,v) = C(u,v) * (u v)^(-1) * ((-log u)^alpha + (-log v)^alpha)^{(2-2alpha)/alpha}
    #          * [((-log u)^alpha + (-log v)^alpha)^{1/alpha} + alpha - 1] * ((-log u)(-log v))^{alpha-1}
    # Taking logs:
    ll = float(
        np.sum(-cap)
        + (1.0 / alpha - 2.0) * np.sum(np.log(s))
        + np.sum(np.log(cap + alpha - 1.0))
        + (alpha - 1.0) * np.sum(np.log(np.log(1.0 / uu)) + np.log(np.log(1.0 / vv)))
        + np.sum(np.log(1.0 / uu) + np.log(1.0 / vv))
    )
    return ll


def _gumbel_fit(u: Array, v: Array) -> dict[str, float]:
    tau = _kendall_tau_pair(u, v)
    if tau <= 0.0:
        pass
    else:
        float(np.clip(1.0 / (1.0 - tau), 1.05, 12.0))
    n = u.shape[0]

    def nll(a: float) -> float:
        if a < 1.001:
            return 1e12
        val = _gumbel_logpdf(u, v, a)
        if not np.isfinite(val):
            return 1e12
        return -val

    res = opt.minimize_scalar(nll, bounds=(1.001, 25.0), method="bounded")
    ll = -float(res.fun)
    return {
        "alpha": float(res.x),
        "loglik": ll,
        "aic": _aic(ll, 1),
        "bic": _bic(ll, 1, n),
    }


# -------------------------------------------------------------------- Frank


def _frank_h(u: Array, v: Array, theta: float) -> Array:
    uu = _clip(u)
    vv = _clip(v)
    if abs(theta) < 1e-8:
        return uu.copy()
    e_mt = np.expm1(-theta)
    gu = np.expm1(-theta * uu)
    gv = np.expm1(-theta * vv)
    cap = -1.0 / theta * np.log1p(gu * gv / e_mt)
    # h(u|v) = ∂C/∂v = φ'(v) / φ'(C)
    # φ'(t) = θ e^{-θt} / (e^{-θt} - 1)
    # Ratio (θ cancels): [e^{-θv}/(e^{-θv}-1)] / [e^{-θC}/(e^{-θC}-1)]
    phi_v = np.exp(-theta * vv) / np.expm1(-theta * vv)
    phi_c = np.exp(-theta * cap) / np.expm1(-theta * cap)
    return np.asarray(phi_v / phi_c, dtype=np.float64)


def _frank_hinv(w: Array, v: Array, theta: float) -> Array:
    ww = _clip(w)
    vv = _clip(v)
    out = np.empty(vv.shape[0], dtype=np.float64)
    for i in range(vv.shape[0]):
        wi = float(ww[i])
        vi = float(vv[i])

        def _f(u: float, vi_fix: float = vi, wi_fix: float = wi) -> float:
            return float(_frank_h(np.array([u]), np.array([vi_fix]), theta)[0] - wi_fix)

        try:
            lo, hi = _EPS, 1.0 - _EPS
            if _f(lo) * _f(hi) >= 0:
                out[i] = lo if _f(lo) < 0 else hi
                continue
            out[i] = float(opt.brentq(_f, lo, hi, xtol=1e-10))
        except (ValueError, RuntimeError, ArithmeticError):
            out[i] = wi
    return out


def _frank_logpdf(u: Array, v: Array, theta: float) -> float:
    uu = _clip(u)
    vv = _clip(v)
    n = uu.shape[0]
    if abs(theta) < 1e-8:
        return 0.0
    # Frank copula density (Schepsmeier & Stöber 2014, eq. 16):
    # c(u,v) = θ * (1-e^{-θ}) * e^{-θ(u+v)} / [(1-e^{-θ}) - (1-e^{-θu})(1-e^{-θv})]²
    a = -np.expm1(-theta)  # = 1 - e^{-θ}  (> 0 for θ > 0)
    bu = -np.expm1(-theta * uu)  # = 1 - e^{-θu}
    bv = -np.expm1(-theta * vv)  # = 1 - e^{-θv}
    denom = a - bu * bv  # the denominator of the copula density
    # denom may be positive or negative; we need denom², log uses abs(denom).
    ll = float(
        n * math.log(abs(theta) * abs(a))
        - theta * np.sum(uu + vv)
        - 2.0 * np.sum(np.log(np.maximum(np.abs(denom), 1e-300)))
    )
    return ll


def _frank_fit(u: Array, v: Array) -> dict[str, float]:
    tau = _kendall_tau_pair(u, v)
    if abs(tau) < 1e-4:
        return {"theta": 0.0, "loglik": 0.0, "aic": 0.0, "bic": 0.0}
    theta0 = float(np.clip(6.0 * tau, -30.0, 30.0))
    if abs(theta0) < 0.01:
        theta0 = 0.5 if tau > 0 else -0.5
    n = u.shape[0]

    def nll(th: float) -> float:
        if abs(th) > 32.0 or abs(th) < 0.005:
            return 1e12
        val = _frank_logpdf(u, v, th)
        if not np.isfinite(val):
            return 1e12
        return -val

    res = opt.minimize_scalar(nll, bounds=(-32.0, 32.0), method="bounded")
    ll = -float(res.fun)
    return {
        "theta": float(res.x),
        "loglik": ll,
        "aic": _aic(ll, 1),
        "bic": _bic(ll, 1, n),
    }


# ---------------------------------------------------------------------- Joe


def _joe_h(u: Array, v: Array, theta: float) -> Array:
    """h(u|v) = ∂C/∂v for the Joe copula.

    With ``ũ = 1-u``, ``ṽ = 1-v``, ``s = ũ^θ + ṽ^θ - ũ^θ·ṽ^θ`` and
    ``C = 1 - s^{1/θ}``:

        ∂C/∂v = s^{1/θ - 1} · ṽ^{θ-1} · (1 - ũ^θ)

    (the mirror form ``s^{1/θ-1}·ũ^{θ-1}·(1-ṽ^θ)`` is ∂C/∂u — the swapped
    conditional).
    """
    uu = _clip(u)
    vv = _clip(v)
    a = (1.0 - uu) ** theta
    b = (1.0 - vv) ** theta
    s = a + b - a * b
    return np.asarray(
        ((1.0 - vv) ** (theta - 1.0) * (1.0 - a) * s ** (1.0 / theta - 1.0)),
        dtype=np.float64,
    )


def _joe_hinv(w: Array, v: Array, theta: float) -> Array:
    ww = _clip(w)
    vv = _clip(v)
    out = np.empty(vv.shape[0], dtype=np.float64)
    for i in range(vv.shape[0]):
        wi = float(ww[i])
        vi = float(vv[i])

        def _f(u: float, vi_fix: float = vi, wi_fix: float = wi) -> float:
            return float(_joe_h(np.array([u]), np.array([vi_fix]), theta)[0] - wi_fix)

        try:
            lo, hi = _EPS, 1.0 - _EPS
            if _f(lo) * _f(hi) >= 0:
                out[i] = lo if _f(lo) < 0 else hi
                continue
            out[i] = float(opt.brentq(_f, lo, hi, xtol=1e-10))
        except (ValueError, RuntimeError, ArithmeticError):
            out[i] = wi
    return out


def _joe_logpdf(u: Array, v: Array, theta: float) -> float:
    uu = _clip(u)
    vv = _clip(v)
    a = (1.0 - uu) ** theta
    b = (1.0 - vv) ** theta
    s_vals = a + b - a * b
    ll = float(
        np.sum(
            (1.0 / theta - 2.0) * np.log(s_vals)
            + (theta - 1.0) * np.log((1.0 - uu) * (1.0 - vv))
            + np.log(theta - 1.0 + s_vals)
        )
    )
    return ll


def _joe_fit(u: Array, v: Array) -> dict[str, float]:
    n = u.shape[0]

    def nll(th: float) -> float:
        if th < 1.001:
            return 1e12
        val = _joe_logpdf(u, v, th)
        if not np.isfinite(val):
            return 1e12
        return -val

    res = opt.minimize_scalar(nll, bounds=(1.001, 25.0), method="bounded")
    ll = -float(res.fun)
    return {
        "theta": float(res.x),
        "loglik": ll,
        "aic": _aic(ll, 1),
        "bic": _bic(ll, 1, n),
    }


# ------------------------------------------------------ Family dispatch map

_H_FN: dict[str, Callable[..., Array]] = {
    "gaussian": _gaussian_h,
    "t": _t_h,
    "clayton": _clayton_h,
    "gumbel": _gumbel_h,
    "frank": _frank_h,
    "joe": _joe_h,
}

_HINV_FN: dict[str, Callable[..., Array]] = {
    "gaussian": _gaussian_hinv,
    "t": _t_hinv,
    "clayton": _clayton_hinv,
    "gumbel": _gumbel_hinv,
    "frank": _frank_hinv,
    "joe": _joe_hinv,
}

_LOGLIK_FN: dict[str, Callable[..., float]] = {
    "gaussian": _gaussian_logpdf,
    "t": _t_logpdf,
    "clayton": _clayton_logpdf,
    "gumbel": _gumbel_logpdf,
    "frank": _frank_logpdf,
    "joe": _joe_logpdf,
}

_FIT_FN: dict[str, Callable[..., dict[str, float]]] = {
    "gaussian": _gaussian_fit,
    "t": _t_fit,
    "clayton": _clayton_fit,
    "gumbel": _gumbel_fit,
    "frank": _frank_fit,
    "joe": _joe_fit,
}


def _select_family(
    u: Array,
    v: Array,
    families: Sequence[str] = _FAMILY_PALETTE,
    criterion: str = "aic",
) -> tuple[str, dict[str, float], float]:
    """Fit every candidate family and return the best by AIC/BIC.

    Returns ``(family_name, param_dict, criterion_value)``.
    """
    best_fam = ""
    best_params: dict[str, float] = {}
    best_crit = np.inf
    n = u.shape[0]

    for fam in families:
        try:
            fit = _FIT_FN[fam](u, v)
        except (ValueError, RuntimeError, ArithmeticError):
            continue
        if not np.isfinite(fit.get("loglik", -np.inf)):
            continue
        n_params = 2 if fam == "t" else 1
        crit = (
            _aic(fit["loglik"], n_params)
            if criterion == "aic"
            else _bic(fit["loglik"], n_params, n)
        )
        if crit < best_crit:
            best_crit = crit
            best_fam = fam
            best_params = fit

    if best_fam == "":
        # fallback: independence (Gaussian with rho ~ 0)
        best_fam = "gaussian"
        best_params = {"rho": 0.0, "loglik": 0.0, "aic": 0.0, "bic": 0.0}
        best_crit = 0.0

    return best_fam, best_params, best_crit


def _h_eval(u: Array, v: Array, family: str, params: dict[str, float]) -> Array:
    fn = _H_FN[family]
    if family == "t":
        return fn(u, v, float(params["rho"]), float(params["nu"]))
    elif family == "gaussian":
        return fn(u, v, float(params["rho"]))
    elif family == "clayton":
        return fn(u, v, float(params["theta"]))
    elif family == "gumbel":
        return fn(u, v, float(params["alpha"]))
    elif family == "frank" or family == "joe":
        return fn(u, v, float(params["theta"]))
    raise ValueError(f"unknown family {family}")


def _hinv_eval(w: Array, v: Array, family: str, params: dict[str, float]) -> Array:
    fn = _HINV_FN[family]
    if family == "t":
        return fn(w, v, float(params["rho"]), float(params["nu"]))
    elif family == "gaussian":
        return fn(w, v, float(params["rho"]))
    elif family == "clayton":
        return fn(w, v, float(params["theta"]))
    elif family == "gumbel":
        return fn(w, v, float(params["alpha"]))
    elif family == "frank" or family == "joe":
        return fn(w, v, float(params["theta"]))
    raise ValueError(f"unknown family {family}")


# =========================================================== Vine operations


def vine_sample(
    vm: VineMatrix,
    n: int,
    seed: int | None = None,
) -> Array:
    """Sample ``n`` observations from a fitted vine copula.

    Uses the inverse Rosenblatt transform (Aas et al. 2009, Algorithm 1):
    draw ``w ~ U(0,1)^d``, then sequentially invert the h-functions
    through the vine structure, bottom-to-top.
    """
    dim = vm.dim
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rng = np.random.default_rng(seed)
    w = rng.random((n, dim))
    uniforms = np.empty((n, dim), dtype=np.float64)

    # Column-assignment map: variable index -> column in output
    # For the standard R-vine matrix, sampling proceeds through
    # columns of the matrix.  We use the implicit ordering implied
    # by the edge list.
    vm_edges = vm.tree_edges
    if not vm_edges:
        # If no edges fitted, just return independent uniforms
        return w

    # Determine variable ordering from tree edges.
    # We need a DAG structure.  For C/D-vines, variable 0 is the
    # root of the first tree.
    # Implement the Aas et al. (2009) general sampling algorithm (Algorithm 2).
    v_direct = np.zeros((dim, dim), dtype=np.float64)  # direct variables
    v_indirect = np.zeros((dim, dim), dtype=np.float64)  # indirect variables

    # Build the sampling arrays from tree_edges
    for tree, edges in enumerate(vm_edges):
        for edge_idx, (a, b, _cond) in enumerate(edges):
            v_direct[tree, edge_idx] = float(a)
            v_indirect[tree, edge_idx] = float(b)

    uniforms[:, 0] = w[:, 0]

    # For the sampling, we need to know for each edge where the
    # h-inverse gets its conditioning variable.  This is implicit
    # in the vine structure.
    # A simpler approach: implement the sampling as:
    # for i in range(1, dim):
    #   cur = w[:, i]
    #   for tree from (dim-2) down to 0:
    #     ... apply h-inverse using the appropriate pair
    # This requires knowing the pair-copula at each (tree, edge).
    #
    # Since we have vm.tree_edges, we can use the column-based
    # sampling algorithm for C-vines and D-vines.
    # For a general R-vine, we need the full matrix inversion.
    #
    # Let's implement C-vine and D-vine sampling explicitly first,
    # then generalize.

    # Dispatch on the recorded structure (vine_fit sets it; the factories
    # tag themselves).  The diagonal heuristic remains as a fallback for
    # hand-built VineMatrix objects that predate the ``structure`` field.
    structure = getattr(vm, "structure", None) or "rvine"
    diag = np.array([vm.matrix[j, j] for j in range(dim)])
    dvals = np.sort(diag)

    if structure == "rvine" and vm.rvine_spec is not None:
        from quant_fund.models.rvine import rvine_sample as _rvine_sim

        return _rvine_sim(n, vm.rvine_spec, rng)

    if structure == "cvine" or (
        structure == "rvine" and np.allclose(dvals, np.arange(1, dim + 1, dtype=float)[::-1])
    ):
        out = _cvine_sample_inner(vm, n, rng)
    elif structure == "dvine" or (
        structure == "rvine" and np.allclose(dvals, np.arange(1, dim + 1, dtype=float))
    ):
        out = _dvine_sample_inner(vm, n, rng)
    else:
        out = _rvine_sample_inner(vm, n, w, rng)

    # Samples are produced in vine positions; map back to the caller's
    # original column order when the fit permuted variables.
    if vm.ordering is not None:
        out = out[:, np.argsort(vm.ordering)]
    return out


def _cvine_sample_inner(vm: VineMatrix, n: int, rng: np.random.Generator) -> Array:
    """C-vine sampling: root at tree 0 and leaves connected to root.

    Inverse Rosenblatt recursion: to sample variable ``i``, apply the
    h-inverses for trees ``i-1 .. 0``, conditioning at tree ``t`` on
    ``F(x_t | x_0..t-1)`` — which by construction equals the input uniform
    ``w_t``, NOT the marginal sample ``x_t``.
    """
    dim = vm.dim
    independent = rng.random((n, dim))
    uniforms = np.zeros((n, dim), dtype=np.float64)
    uniforms[:, 0] = independent[:, 0]

    for i in range(1, dim):
        val = independent[:, i].copy()
        for j in range(i - 1, -1, -1):
            tree = j
            edge_idx = i - j - 1
            key = (tree, edge_idx)
            if key in vm.families:
                fam, par = vm.pair_copula(tree, edge_idx)
                val = _hinv_eval(val, independent[:, j], fam, par)
        uniforms[:, i] = val

    return uniforms


def _dvine_sample_inner(vm: VineMatrix, n: int, rng: np.random.Generator) -> Array:
    """D-vine sampling: path structure 1-2-3-...-d.

    Inverse Rosenblatt recursion mirroring the fit's double array:
    ``v_left[t][s] = F(x_s | x_{s+1..s+t})`` and
    ``v_right[t][s] = F(x_{s+t} | x_{s..s+t-1})`` over *sampled* values.
    To sample var ``i`` at tree ``t`` (edge ``(t, i-t-1)``), the
    conditioning value is ``v_left[t][i-t-1]`` — a Rosenblatt transform
    of the window's left endpoint, NOT the marginal ``x_{i-t-1}`` or the
    input uniform.
    """
    dim = vm.dim
    independent = rng.random((n, dim))
    uniforms = np.zeros((n, dim), dtype=np.float64)
    uniforms[:, 0] = independent[:, 0]

    v_left: dict[tuple[int, int], Array] = {(0, 0): independent[:, 0].copy()}
    v_right: dict[tuple[int, int], Array] = {(0, 0): independent[:, 0].copy()}

    for i in range(1, dim):
        val = independent[:, i].copy()
        for t in range(i - 1, -1, -1):
            s = i - t - 1
            if (t, s) in vm.families:
                fam, par = vm.pair_copula(t, s)
                val = _hinv_eval(val, v_left[(t, s)], fam, par)
        uniforms[:, i] = val
        v_left[(0, i)] = uniforms[:, i].copy()
        v_right[(0, i)] = uniforms[:, i].copy()

        # Fill transforms for every window [s..i] that now has both ends.
        for t in range(1, i + 1):
            s = i - t
            key = (t - 1, s)
            if key in vm.families:
                fam, par = vm.pair_copula(t - 1, s)
                v_left[(t, s)] = _h_eval(v_left[(t - 1, s)], v_right[(t - 1, s + 1)], fam, par)
                v_right[(t, s)] = _h_eval(v_right[(t - 1, s + 1)], v_left[(t - 1, s)], fam, par)
            else:
                v_left[(t, s)] = v_left[(t - 1, s)]
                v_right[(t, s)] = v_right[(t - 1, s + 1)]

    return uniforms


def _rvine_sample_inner(vm: VineMatrix, n: int, w: Array, rng: np.random.Generator) -> Array:
    """Fallback for hand-built ``VineMatrix`` objects without ``structure``.

    Re-detects the C/D-vine diagonal patterns; anything else warns and
    returns independent uniforms — a correct general R-vine sampler needs
    per-edge Rosenblatt transforms that ``tree_edges`` alone does not carry.
    """
    dim = vm.dim
    diag_vals = np.array([int(vm.matrix[i, i]) for i in range(dim)])
    dvals = np.arange(1, dim + 1)
    diag_sorted = np.sort(diag_vals)
    if np.allclose(diag_sorted, dvals[::-1]):
        return _cvine_sample_inner(vm, n, rng)

    if np.allclose(diag_sorted, dvals):
        return _dvine_sample_inner(vm, n, rng)

    # Generic R-vine sampling needs per-edge Rosenblatt transforms of the
    # conditioning variables — metadata that ``tree_edges`` alone does not
    # carry.  Warn honestly instead of returning wrong-conditioning values.
    if vm.tree_edges and vm.families:
        warnings.warn(
            "vine_sample: generic R-vine sampling is not implemented; "
            "returning independent uniforms",
            stacklevel=2,
        )
    else:
        warnings.warn(
            "vine_sample: unrecognized vine structure, returning independent", stacklevel=2
        )
    return w


def _edge_loglik(fam: str, par: dict[str, float], u1: Array, u2: Array) -> float:
    fn = _LOGLIK_FN[fam]
    if fam == "t":
        return fn(u1, u2, float(par["rho"]), float(par["nu"]))
    if fam == "gaussian":
        return fn(u1, u2, float(par["rho"]))
    if fam == "clayton":
        return fn(u1, u2, float(par["theta"]))
    if fam == "gumbel":
        return fn(u1, u2, float(par["alpha"]))
    return fn(u1, u2, float(par["theta"]))  # frank / joe


def vine_logpdf(vm: VineMatrix, u: Array) -> float:
    """Evaluate the vine copula log-density at observations ``u``.

    ``u`` is an ``(n, d)`` matrix of pseudo-observations in the caller's
    ORIGINAL column order.  Columns are re-ordered to vine positions via
    ``vm.ordering``, then every pair-copula is evaluated on its
    *Rosenblatt-transformed* arguments — the forward h-ladder matching
    ``vine_fit``.  Evaluating tree-t edges on raw marginals is a
    wrong-conditioning bug (the pair densities live on conditioned
    uniforms).
    """
    return float(_vine_logpdf_rows(vm, u).sum())


def vine_logpdf_rows(vm: VineMatrix, u: Array) -> Array:
    """Per-observation vine log-density, shape (n,); sums to ``vine_logpdf``.

    Needed by lanes that score *individual* held-out observations (e.g.
    confidence sequences on log-loss differentials) rather than totals.
    """
    return _vine_logpdf_rows(vm, u)


def _edge_logpdf_rows(fam: str, par: dict[str, float], u1: Array, u2: Array) -> Array:
    """Per-observation log pair-copula density (vectorized, no row sum)."""
    from quant_fund.models.rvine import _vec_logpdf

    return _vec_logpdf(fam, par, u1, u2)


def _vine_logpdf_rows(vm: VineMatrix, u: Array) -> Array:
    """Per-observation vine log-density — the row-wise spine of vine_logpdf."""
    m = _as_finite_matrix(u)
    if m.shape[1] != vm.dim:
        raise ValueError(f"u has {m.shape[1]} columns but vine has dim={vm.dim}")
    if vm.ordering is not None:
        m = m[:, vm.ordering]
    d = vm.dim
    structure = vm.structure

    row_ll = np.zeros(m.shape[0], dtype=np.float64)

    if structure == "cvine":
        # h_data[t][c] = F(x_{t+c} | x_0..t-1); col 0 is the tree root.
        h_data: list[Array] = [m]
        for tree in range(d - 1):
            cur = h_data[tree]
            next_cols: list[Array] = []
            for leaf_var in range(tree + 1, d):
                edge_idx = leaf_var - tree - 1
                key = (tree, edge_idx)
                if key not in vm.families:
                    continue
                fam, par = vm.pair_copula(tree, edge_idx)
                row_ll += _edge_logpdf_rows(fam, par, cur[:, 0], cur[:, leaf_var - tree])
            if tree < d - 2 and cur.shape[1] > 1:
                # Push: next root col + remaining leaf transforms.
                first: Array
                if (tree, 0) in vm.families:
                    fam, par = vm.pair_copula(tree, 0)
                    first = _h_eval(cur[:, 1], cur[:, 0], fam, par)
                else:
                    first = cur[:, 1]
                next_cols = [first]
                for leaf_var in range(tree + 2, d):
                    key = (tree, leaf_var - tree - 1)
                    if key in vm.families:
                        fam, par = vm.pair_copula(tree, leaf_var - tree - 1)
                        next_cols.append(_h_eval(cur[:, leaf_var - tree], cur[:, 0], fam, par))
                    else:
                        next_cols.append(cur[:, leaf_var - tree])
                h_data.append(np.column_stack([np.asarray(c, dtype=np.float64) for c in next_cols]))
        return row_ll

    if structure == "dvine":
        # Double-array ladder (mirrors vine_fit):
        #   v_left[t][s]  = F(x_s   | x_{s+1..s+t})
        #   v_right[t][s] = F(x_{s+t} | x_{s..s+t-1})
        # edge (t, s) pairs (v_left[t][s], v_right[t][s+1]).
        v_left = [m[:, c].copy() for c in range(d)]
        v_right = [m[:, c].copy() for c in range(d)]
        for tree in range(d - 1):
            next_left: list[Array] = []
            next_right: list[Array] = []
            for s in range(d - tree - 1):
                key = (tree, s)
                if key in vm.families:
                    fam, par = vm.pair_copula(tree, s)
                    row_ll += _edge_logpdf_rows(fam, par, v_left[s], v_right[s + 1])
                    next_left.append(_h_eval(v_left[s], v_right[s + 1], fam, par))
                    next_right.append(_h_eval(v_right[s + 1], v_left[s], fam, par))
                else:
                    next_left.append(v_left[s])
                    next_right.append(v_right[s + 1])
            v_left = next_left
            v_right = next_right
            if not v_left:
                break
        return row_ll

    if structure == "rvine" and vm.rvine_spec is not None:
        from quant_fund.models.rvine import rvine_logpdf_edges

        return rvine_logpdf_edges(m, vm.rvine_spec)

    # Generic R-vine without a fitted spec: edge endpoints are variable
    # labels in vine space, and the correct arguments are conditioned
    # transforms that ``tree_edges`` alone does not describe — refuse rather
    # than silently evaluate wrong-conditioning densities.
    raise ValueError(
        "vine_logpdf requires structure='cvine' or 'dvine' "
        "(generic R-vine evaluation is unsupported)"
    )


# --------------------------------------------------------------- vine_fit


def _mst_edges(
    tau_mat: Array, used_pairs: set[tuple[int, int]]
) -> list[tuple[int, int, tuple[int, ...]]]:
    """Maximum spanning tree via Prim's algorithm on |Kendall-tau|.

    ``used_pairs`` are edges already in the vine (lower trees).  Returns
    edges in the order they enter the MST, with empty condition sets
    (conditions are added by the caller).
    """
    d = tau_mat.shape[0]
    in_tree = np.zeros(d, dtype=bool)
    parent = np.full(d, -1, dtype=int)
    weight = np.full(d, -np.inf)
    weight[0] = 0.0
    edges_in_order: list[tuple[int, int]] = []

    for _ in range(d):
        # find the unvisited node with max weight
        best = -1
        best_w = -np.inf
        for v in range(d):
            if not in_tree[v] and weight[v] > best_w:
                best_w = weight[v]
                best = v
        if best < 0:
            break
        in_tree[best] = True
        if parent[best] >= 0:
            a, b = int(parent[best]), int(best)
            pair = (min(a, b), max(a, b))
            if pair not in used_pairs:
                edges_in_order.append(pair)
                used_pairs.add(pair)

        # update weights for unvisited neighbors
        for v in range(d):
            if not in_tree[v]:
                w = abs(tau_mat[best, v])
                if w > weight[v]:
                    weight[v] = w
                    parent[v] = best

    # Build edge list with empty condition sets (caller fills these in)
    result: list[tuple[int, int, tuple[int, ...]]] = []
    for a, b in edges_in_order:
        result.append((a, b, ()))
    return result


def vine_fit(
    u: Array,
    families: Sequence[str] = _FAMILY_PALETTE,
    criterion: str = "aic",
    max_dim: int = _MAX_DIM,
    tau_threshold: float = 0.02,
    structure: str = "cvine",
) -> VineMatrix:
    """Fit a vine copula to pseudo-observations ``u`` (n × d).

    Implements sequential pair-copula fitting within a fixed vine structure
    (C-vine or D-vine).  The variable ordering is determined by a greedy
    top-tree Kendall-tau maximisation (MST on |τ|).

    Complexity
    ----------
    O(d² × |families| × n): each tree fits O(d) pair-copulas, each
    requiring O(|families| × n) for family selection + MLE.

    Parameters
    ----------
    u : (n, d) array of pseudo-observations in (0, 1).
    families : sequence of family names to try (default all 6).
    criterion : "aic" or "bic" for family selection.
    max_dim : maximum dimension guard (default 30).
    tau_threshold : skip edges with |Kendall-tau| below this value.
    structure : "cvine" (default) or "dvine".

    Returns
    -------
    VineMatrix with fitted families, params, and edge structure.
    """
    m = _as_finite_matrix(u)
    n_obs, d = m.shape
    if d > max_dim:
        raise ValueError(f"dimension {d} exceeds max_dim={max_dim}")
    if structure not in ("cvine", "dvine", "rvine"):
        raise ValueError("structure must be 'cvine', 'dvine' or 'rvine'")

    if structure == "rvine":
        from quant_fund.models.rvine import rvine_fit

        spec = rvine_fit(m, families=families, criterion=criterion, tau_threshold=tau_threshold)
        tree_edges_r = [[(e.x, e.y, tuple(e.cond)) for e in tree_list] for tree_list in spec.edges]
        fam_r: dict[tuple[int, int], str] = {}
        par_r: dict[tuple[int, int], dict[str, float]] = {}
        n_par = 0
        for t, tree_list in enumerate(spec.edges):
            for eidx, e in enumerate(tree_list):
                fam_r[(t, eidx)] = e.family
                par_r[(t, eidx)] = dict(e.params)
                n_par += 2 if e.family == "t" else 1
        vm_r = VineMatrix(
            matrix=np.asarray(spec.matrix, dtype=np.float64),
            families=fam_r,
            params=par_r,
            loglik=float(spec.loglik),
            tree_edges=tree_edges_r,
            ordering=None,
            structure="rvine",
            rvine_spec=spec,
        )
        vm_r.n_params = n_par
        vm_r.aic = _aic(spec.loglik, n_par)
        vm_r.bic = _bic(spec.loglik, n_par, n_obs)
        return vm_r

    # Determine variable ordering via MST on |tau|
    tau_full = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(i + 1, d):
            tau_full[i, j] = _kendall_tau_pair(m[:, i], m[:, j])
            tau_full[j, i] = tau_full[i, j]

    ordering = _mst_ordering(tau_full)
    m_ordered = m[:, ordering]

    # Build the target vine structure
    if structure == "cvine":
        vm = cvine_structure(d)
    else:
        vm = dvine_structure(d)

    total_loglik = 0.0
    total_n_params = 0

    # Maintain a matrix of pseudo-observations:
    # h_data[tree][var_idx] = pseudo-observation for variable var_idx
    #                         conditioned on all variables up to tree-1
    # Tree 0 uses original (reordered) data
    h_data: list[Array] = [m_ordered.copy()]

    for tree in range(d - 1):
        next_cols: list[Array] = []
        if structure == "cvine":
            col_root = 0
            for leaf_var in range(tree + 1, d):
                edge_idx = leaf_var - tree - 1
                col_leaf = leaf_var - tree
                u_root = h_data[tree][:, col_root]
                u_leaf = h_data[tree][:, col_leaf]
                tau_ab = abs(_kendall_tau_pair(u_root, u_leaf))
                if tau_ab < tau_threshold:
                    fam = "gaussian"
                    par = {"rho": 0.0, "loglik": 0.0, "aic": 0.0, "bic": 0.0}
                else:
                    fam, par, _ = _select_family(
                        u_root, u_leaf, families=tuple(families), criterion=criterion
                    )
                key = (tree, edge_idx)
                vm.families[key] = fam
                vm.params[key] = par
                total_loglik += float(par.get("loglik", 0.0))
                total_n_params += 2 if fam == "t" else 1

            if tree < d - 2:
                next_cols = []
                # First column: h_{next_root | 0..tree}
                # where next_root = tree + 1, using edge (tree, tree+1) from this tree
                next_root_edge = 0  # first edge: (root, root+1)
                if (tree, next_root_edge) in vm.families:
                    fam = vm.families[(tree, next_root_edge)]
                    par = vm.params[(tree, next_root_edge)]
                    u_leaf = h_data[tree][:, 1]  # col 1 = leaf tree+1
                    u_root = h_data[tree][:, 0]  # col 0 = root
                    h_root_next = _h_eval(u_leaf, u_root, fam, par)
                    next_cols.append(h_root_next)
                else:
                    next_cols.append(h_data[tree][:, 1])

                # Remaining columns: h_{leaf | 0..tree} for leaf > tree+1
                for leaf_var in range(tree + 2, d):
                    edge_idx = leaf_var - tree - 1
                    key_edge = (tree, edge_idx)
                    col_leaf = leaf_var - tree
                    if key_edge not in vm.families:
                        next_cols.append(h_data[tree][:, col_leaf])
                        continue
                    fam = vm.families[key_edge]
                    par = vm.params[key_edge]
                    h_val = _h_eval(h_data[tree][:, col_leaf], h_data[tree][:, col_root], fam, par)
                    next_cols.append(h_val)
                if next_cols:
                    h_data.append(
                        np.column_stack([np.asarray(c, dtype=np.float64) for c in next_cols])
                    )

        else:
            # D-vine: path structure — the ladder needs TWO transforms per
            # window (Aas et al. 2009, §5): at tree t, edge (t, start) pairs
            #   left  = F(x_start     | x_{start+1..start+t})   = v_dir[t][start]
            #   right = F(x_{start+t+1} | x_{start+1..start+t}) = v_ind[t][start+1]
            # and both sides propagate:
            #   v_dir[t+1][i] = h(v_dir[t][i]   | v_ind[t][i+1], edge(t,i))
            #   v_ind[t+1][i] = h(v_ind[t][i+1] | v_dir[t][i],   edge(t,i))
            if tree == 0:
                v_dir = [h_data[0][:, c].copy() for c in range(d)]
                v_ind = [h_data[0][:, c].copy() for c in range(d)]
            for start in range(d - tree - 1):
                edge_idx = start
                u_a = v_dir[start]
                u_b = v_ind[start + 1]
                tau_ab = abs(_kendall_tau_pair(u_a, u_b))
                if tau_ab < tau_threshold:
                    fam = "gaussian"
                    par = {"rho": 0.0, "loglik": 0.0, "aic": 0.0, "bic": 0.0}
                else:
                    fam, par, _ = _select_family(
                        u_a, u_b, families=tuple(families), criterion=criterion
                    )
                key = (tree, edge_idx)
                vm.families[key] = fam
                vm.params[key] = par
                total_loglik += float(par.get("loglik", 0.0))
                total_n_params += 2 if fam == "t" else 1

            if tree < d - 2:
                next_dir = []
                next_ind = []
                for start in range(d - tree - 1):
                    key_edge = (tree, start)
                    if key_edge in vm.families:
                        fam = vm.families[key_edge]
                        par = vm.params[key_edge]
                        next_dir.append(_h_eval(v_dir[start], v_ind[start + 1], fam, par))
                        next_ind.append(_h_eval(v_ind[start + 1], v_dir[start], fam, par))
                    else:
                        next_dir.append(v_dir[start])
                        next_ind.append(v_ind[start + 1])
                v_dir = next_dir
                v_ind = next_ind

    vm.ordering = np.asarray(ordering, dtype=int)
    vm.structure = structure

    vm.loglik = total_loglik
    vm.n_params = total_n_params
    vm.aic = _aic(total_loglik, total_n_params)
    vm.bic = _bic(total_loglik, total_n_params, n_obs)

    return vm


def _mst_ordering(tau_mat: Array) -> list[int]:
    """Determine variable ordering via Prim's MST on |Kendall-tau|.

    Returns a permutation of ``0..d-1`` where sequentially connected
    nodes are adjacent in the ordering (used for D-vine ordering).
    """
    d = tau_mat.shape[0]
    in_tree = np.zeros(d, dtype=bool)
    parent = np.full(d, -1, dtype=int)
    weight = np.full(d, -np.inf)
    weight[0] = 0.0
    edges: list[tuple[int, int]] = []

    for _ in range(d):
        best = -1
        best_w = -np.inf
        for v in range(d):
            if not in_tree[v] and weight[v] > best_w:
                best_w = weight[v]
                best = v
        if best < 0:
            break
        in_tree[best] = True
        if parent[best] >= 0:
            edges.append((int(parent[best]), int(best)))
        for v in range(d):
            if not in_tree[v]:
                w = abs(tau_mat[best, v])
                if w > weight[v]:
                    weight[v] = w
                    parent[v] = best

    # Build ordering from edges: DFS traversal of the MST
    adj: dict[int, list[int]] = {i: [] for i in range(d)}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    visited: set[int] = set()
    ordering: list[int] = []

    def dfs(node: int) -> None:
        visited.add(node)
        ordering.append(node)
        for nb in adj[node]:
            if nb not in visited:
                dfs(nb)

    dfs(0)
    # Add any isolated nodes
    for v in range(d):
        if v not in visited:
            ordering.append(v)

    return ordering


# ============================================================ GAS copula


def gas_copula_score_gaussian(u: Array, v: Array, rho: float) -> Array:
    """Score of the Gaussian copula log-density w.r.t. the unconstrained
    parameter ``kappa = atanh(rho)``.

    For each observation pair ``(u_i, v_i)`` and current ``rho``, returns
    the score vector ``s_i``.

    The analytic score is:

        ∂log c / ∂κ = ρ - ρ·(x²+y²)/(1-ρ²) + x·y·(1+ρ²)/(1-ρ²)

    where x = Φ⁻¹(u), y = Φ⁻¹(v).  Score is clipped to prevent
    numerical blow-up near ρ ≈ ±1.
    """
    uu = _clip(u)
    vv = _clip(v)
    x = sstats.norm.ppf(uu)
    y = sstats.norm.ppf(vv)
    r = float(np.clip(rho, -0.998, 0.998))
    r2 = r * r
    one_m_r2 = max(1.0 - r2, 1e-6)
    score = r - r * (x * x + y * y) / one_m_r2 + x * y * (1.0 + r2) / one_m_r2
    return np.asarray(np.clip(score, -20.0, 20.0), dtype=np.float64)


def gas_copula_logpdf(u: Array, v: Array, rho: float) -> float:
    """Gaussian copula log-density at parameter ``rho``."""
    return _gaussian_logpdf(u, v, rho)


def gas_copula_filter(
    u: Array,
    omega: float,
    alpha: float,
    beta: float,
    *,
    kappa0: float | None = None,
) -> tuple[Array, Array, Array]:
    """Filter a GAS(1,1) dynamic Gaussian copula.

    Recursion (Creal et al. 2013, §4):

        κ_{t+1} = ω + α·s_t + β·κ_t
        ρ_t = tanh(κ_t),  clipped to [-0.998, 0.998]
        s_t = score of copula log-density w.r.t. κ_t (clipped to [-20, 20])

    Returns ``(rho_path, kappa_path, score_path)``, each length ``T``.

    Parameters
    ----------
    u : (T, 2) array of pseudo-observations.
    omega, alpha, beta : GAS(1,1) parameters.  Requires |β| < 1.
    kappa0 : initial κ value.  If None, use atanh(sample correlation).
    """
    m = _as_finite_matrix(u, min_cols=2)
    if m.shape[1] != 2:
        m = m[:, :2]
    T = m.shape[0]
    rho = np.empty(T, dtype=np.float64)
    kappa = np.empty(T, dtype=np.float64)
    scores = np.empty(T, dtype=np.float64)

    if kappa0 is None:
        corr = float(
            np.corrcoef(sstats.norm.ppf(_clip(m[:, 0])), sstats.norm.ppf(_clip(m[:, 1])))[0, 1]
        )
        corr = float(np.clip(corr, -0.98, 0.98))
        k0 = math.atanh(corr) if abs(corr) < 0.999 else math.copysign(3.0, corr)
    else:
        k0 = float(np.clip(kappa0, -4.0, 4.0))

    k_prev = float(k0)
    for t in range(T):
        kappa[t] = k_prev
        rho[t] = float(np.clip(np.tanh(k_prev), -0.998, 0.998))
        s_t = float(gas_copula_score_gaussian(m[t : t + 1, 0], m[t : t + 1, 1], float(rho[t]))[0])
        scores[t] = s_t
        k_next = omega + alpha * s_t + beta * k_prev
        # Clip kappa to prevent runaway
        k_next = float(np.clip(k_next, -10.0, 10.0))
        if not np.isfinite(k_next):
            raise ValueError(f"GAS recursion diverged at t={t}")
        k_prev = k_next

    return rho, kappa, scores


def gas_copula_fit(
    u: Array,
    *,
    omega_bounds: tuple[float, float] = (-2.0, 2.0),
    alpha_bounds: tuple[float, float] = (0.0, 1.5),
    beta_bounds: tuple[float, float] = (-0.97, 0.97),
) -> dict[str, Array | float]:
    """Fit GAS(1,1) dynamic Gaussian copula by MLE.

    Estimates ``(omega, alpha, beta)`` that drive the time-varying
    dependence parameter ``ρ_t = tanh(κ_t)`` via:

        κ_{t+1} = ω + α·s_t + β·κ_t

    Returns a dict with keys: ``omega``, ``alpha``, ``beta``, ``rho``,
    ``kappa``, ``scores``, ``loglik``, ``converged``.
    """
    m = _as_finite_matrix(u, min_cols=2)
    if m.shape[1] != 2:
        m = m[:, :2]
    T = m.shape[0]
    if T < 30:
        raise ValueError("need at least 30 observations for GAS fit")

    # Initial kappa from sample correlation
    corr = float(
        np.corrcoef(sstats.norm.ppf(_clip(m[:, 0])), sstats.norm.ppf(_clip(m[:, 1])))[0, 1]
    )
    corr = float(np.clip(corr, -0.98, 0.98))
    kappa_init = math.atanh(corr) if abs(corr) < 0.999 else math.copysign(3.0, corr)

    def nll(theta: Array) -> float:
        om = float(theta[0])
        al = float(theta[1])
        be = float(theta[2])
        if abs(be) >= 0.97:
            return 1e12
        try:
            rho_p, _, _ = gas_copula_filter(m, om, al, be, kappa0=kappa_init)
        except ValueError:
            return 1e12
        ll = 0.0
        for t in range(T):
            rt = float(np.clip(rho_p[t], -0.998, 0.998))
            ll_t = gas_copula_logpdf(m[t : t + 1, 0], m[t : t + 1, 1], rt)
            if not np.isfinite(ll_t):
                return 1e12
            ll += ll_t
        if not np.isfinite(ll):
            return 1e12
        return -ll

    # Multiple starting values to avoid local minima
    candidates = [
        np.array([0.0, 0.2, 0.7]),
        np.array([0.0, 0.05, 0.95]),
        np.array([0.05, 0.1, 0.8]),
    ]
    bounds = [omega_bounds, alpha_bounds, beta_bounds]
    best_ll = np.inf
    best_x = candidates[0].copy()

    for theta0 in candidates:
        try:
            res = opt.minimize(nll, theta0, method="L-BFGS-B", bounds=bounds)
            # res.fun == 1e12 is the penalty sentinel, never a valid optimum
            if np.isfinite(res.fun) and res.fun < 1e11 and res.fun < best_ll:
                best_ll = float(res.fun)
                best_x = res.x.copy()
                best_success = float(res.success)
        except (ValueError, RuntimeError, ArithmeticError):
            continue

    omega_hat, alpha_hat, beta_hat = float(best_x[0]), float(best_x[1]), float(best_x[2])

    try:
        rho_p, kap_p, sc_p = gas_copula_filter(m, omega_hat, alpha_hat, beta_hat, kappa0=kappa_init)
    except ValueError:
        rho_p = np.full(T, float(np.clip(corr, -0.99, 0.99)))
        kap_p = np.full(T, math.atanh(float(np.clip(corr, -0.98, 0.98))))
        sc_p = np.zeros(T)

    return {
        "omega": omega_hat,
        "alpha": alpha_hat,
        "beta": beta_hat,
        "rho": rho_p,
        "kappa": kap_p,
        "scores": sc_p,
        "loglik": float(-best_ll) if np.isfinite(best_ll) else 0.0,
        "converged": float(best_success) if np.isfinite(best_ll) else 0.0,
    }


# ===================================================== Utility: tail dep


def _empirical_tail_dep_pair(u: Array, v: Array, k: int | None = None) -> tuple[float, float]:
    """Empirical lower/upper tail dependence for a pair."""
    uu = np.asarray(u, dtype=float).ravel()
    vv = np.asarray(v, dtype=float).ravel()
    n = uu.shape[0]
    kk = int(max(2, math.sqrt(n))) if k is None else int(k)
    if kk < 1 or kk >= n // 2:
        raise ValueError("k out of range")
    s = np.sort(uu)
    p = s[kk]
    q = s[n - 1 - kk]
    lower = float(np.mean(vv[uu <= p] <= p))
    upper = float(np.mean(vv[uu >= q] >= q))
    return lower, upper


def vine_tail_dependence(
    vm: VineMatrix, n_sim: int = 20000, seed: int | None = None
) -> dict[str, dict[str, float]]:
    """Estimate all pair-wise tail-dependence coefficients from a fitted vine.

    Returns a dict ``{(i, j): {"lower": λ_L, "upper": λ_U}}``.
    """
    samples = vine_sample(vm, n_sim, seed=seed)
    d = vm.dim
    result: dict[str, dict[str, float]] = {}
    for i in range(d):
        for j in range(i + 1, d):
            lo, hi = _empirical_tail_dep_pair(samples[:, i], samples[:, j])
            result[f"{i},{j}"] = {"lower": lo, "upper": hi}
    return result
