"""Numerical QuEST spectrum estimation and nonlinear covariance shrinkage.

Ledoit & Wolf (2015) "Spectrum estimation: a unified framework for
covariance matrix estimation and PCA in large dimensions" (JMVA 139) and
Ledoit & Wolf (2017) "Numerical implementation of the QuEST function"
(Computational Statistics & Data Analysis 115) define the QuEST map: a
discretized population eigenvalue spectrum ``tau`` (``p`` nonnegative
masses) plus effective sample size ``n`` determine the asymptotic sample
eigenvalue distribution through the discretized Marcenko-Pastur equation.
Inverting that map against observed sample eigenvalues yields a
consistent estimate of the population spectrum, and the Oracle shrinkage
formula on the QuEST internals yields the nonlinear-shrunk covariance.

This module is a faithful numerical port of the reference implementation
(the authors' MATLAB code as packaged in the R ``nlshrink`` package:
``QuEST`` / ``get_lambda_J`` / ``tau_estimate`` / ``nlshrink_est`` /
``linshrink``). This is the *numerical inversion* estimator — distinct
from the analytical LW-2020 shrinker in ``models/covariance.py``, which
solves the same problem via kernel/Hilbert transforms without recovering
the population spectrum itself.

Fail-closed contract: degenerate inputs (non-finite, all-zero spectrum,
empty support, non-convergent inversion, non-finite Jacobian) raise
``ValueError``; nothing silently degrades to a cruder estimator. All
solvers are deterministic — same inputs produce identical outputs.

Notation follows the reference: ``u``-space is the population-spectrum
coordinate, ``x``-space the sample-spectrum coordinate; ``m_LF`` is the
modified (companion) Stieltjes transform of the population spectral
distribution.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar

Array = NDArray[np.float64]

QUEST_POWER = 4  # exponent ``a`` of the zeta power transform of the abscissa
QUEST_MIN_GRID = 100  # minimum grid points per support interval before scaling
QUEST_BISECTION_STEPS = 200  # bisection depth for the MP imaginary-root solve
QUEST_LBFGSB_MAXITER = 2000


@dataclass
class QuestForward:
    """Discretized QuEST solution for one (tau, n) pair.

    Carries the internals the Jacobian and the nonlinear shrinkage map
    need, in the same layout as the reference implementation.
    """

    p: int
    n: int
    c: float
    tau: Array  # sorted population spectrum (length p, may include zeros)
    t: Array  # unique positive population eigenvalues
    pw: Array  # multiplicity of each entry of ``t`` (float for arithmetic)
    pzw: int  # number of zero eigenvalues in ``tau``
    numint: int
    u_fbar: Array  # (numint, 2) support intervals in u-space
    endpoints: Array  # (numint, 2) support endpoints in x-space
    m_lf_u_fbar: Array  # (numint, 2) companion Stieltjes at u_fbar
    numeig: Array  # (numint,) int — sample eigenvalues per interval
    f0: float
    fstart: Array
    fend: Array
    xi_list: list[Array] = field(default_factory=list)
    zxi_list: list[NDArray[np.complex128]] = field(default_factory=list)
    m_lf_zxi_list: list[NDArray[np.complex128]] = field(default_factory=list)
    x_list: list[Array] = field(default_factory=list)
    f_list: list[Array] = field(default_factory=list)
    zeta_list: list[Array] = field(default_factory=list)
    g_list: list[Array] = field(default_factory=list)
    dis_x_list: list[Array] = field(default_factory=list)
    dis_zeta_list: list[Array] = field(default_factory=list)
    dis_m_lf_list: list[NDArray[np.complex128]] = field(default_factory=list)
    dis_g_list: list[Array] = field(default_factory=list)
    dis_g_raw: list[Array] = field(default_factory=list)
    dis_g_list_cum: list[Array] = field(default_factory=list)  # dis_G_list
    f_vals_list: list[Array] = field(default_factory=list)  # unique dis_G
    f_idx_list: list[NDArray[np.intp]] = field(default_factory=list)  # rows in dis arrays
    x_f_list: list[Array] = field(default_factory=list)
    x_f_mean_list: list[Array] = field(default_factory=list)
    x_f_diff_list: list[Array] = field(default_factory=list)
    f_diff_list: list[Array] = field(default_factory=list)
    quant_list: list[Array] = field(default_factory=list)
    bins_list: list[NDArray[np.intp]] = field(default_factory=list)
    integral_indic_list: list[Array] = field(default_factory=list)
    lam: Array = field(default_factory=lambda: np.zeros(0))


def _brent_min_squared(fn: Callable[[float], float], lo: float, hi: float) -> float:
    """Bounded Brent minimization matching R ``optimize(tol=1e-20)``."""
    out = minimize_scalar(fn, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
    return float(out.x)


def _spectral_support(
    tau: Array, t: Array, pw: Array, pzw: int, n: int, p: int
) -> tuple[Array, Array, Array]:
    """Support intervals of the sample spectrum in u-space.

    Ports ``sup_fn``: a gap between population clusters is certified where
    ``phi(u) = sum_k pw_k t_k^2 / (t_k - u)^2`` dips below ``n``. Interior
    separation boundaries are the two ``phi = n`` roots bracketing the
    reference point ``x_fn``; global outer boundaries extend the same
    root-find beyond the extreme eigenvalues.
    """
    k_count = t.size
    pwt2 = pw * t * t

    def x_fn(i: int) -> float:
        return float(
            (t[i] * t[i + 1]) ** (2.0 / 3.0)
            * ((pw[i] * t[i + 1]) ** (1.0 / 3.0) + (pw[i + 1] * t[i]) ** (1.0 / 3.0))
            / (pwt2[i] ** (1.0 / 3.0) + pwt2[i + 1] ** (1.0 / 3.0))
        )

    def p_theta(u: float, i: int) -> float:
        with np.errstate(divide="ignore"):
            return float(pwt2[i] / (t[i] - u) ** 2 + pwt2[i + 1] / (t[i + 1] - u) ** 2)

    def p_phi_l(u: float, i: int) -> float:
        if i == 0:
            return 0.0
        return float(np.sum(pwt2[:i] / (t[:i] - u) ** 2))

    def p_phi_r(u: float, i: int) -> float:
        if i >= k_count - 2:
            return 0.0
        return float(np.sum(pwt2[i + 2 :] / (t[i + 2 :] - u) ** 2))

    def p_phi(u: float) -> float:
        return float(np.sum(pwt2 / (t - u) ** 2))

    def p_phi_diff(u: float) -> float:
        return float(2.0 * np.sum(pwt2 / (t - u) ** 3))

    if k_count == 1:
        boundint = np.empty((0, 4))
    else:
        sep_rows: list[list[float]] = []
        for i in range(k_count - 1):
            # Necessary condition first (cheap screen).
            if not (p_theta(x_fn(i), i) + p_phi_l(t[i + 1], i) + p_phi_r(t[i], i) < n):
                continue
            x_mid = x_fn(i)
            d = p_phi_diff(x_mid)
            if d == 0.0:
                gap_root = x_mid
            elif d > 0.0:
                gap_root = _brent_min_squared(lambda u: p_phi_diff(u) ** 2, t[i], x_mid)
                if p_phi(gap_root) >= n:
                    continue
            else:
                gap_root = _brent_min_squared(lambda u: p_phi_diff(u) ** 2, x_mid, t[i + 1])
                if p_phi(gap_root) >= n:
                    continue
            lb = _brent_min_squared(lambda u: (p_phi(u) - n) ** 2, t[i], x_mid)
            ub = _brent_min_squared(lambda u: (p_phi(u) - n) ** 2, x_mid, t[i + 1])
            pw1 = pzw + int(np.sum(pw[: i + 1]))
            pw2 = p - pw1
            sep_rows.append([lb, ub, float(pw1), float(pw2)])
        boundint = np.asarray(sep_rows, dtype=float).reshape(-1, 4)

    spread = float(np.sqrt(np.sum(pwt2) / n)) + 1.0
    x_t1 = _brent_min_squared(lambda u: (p_phi(u) - n) ** 2, t[0] - spread, t[0])
    x_tk = _brent_min_squared(
        lambda u: (p_phi(u) - n) ** 2, t[k_count - 1], t[k_count - 1] + spread
    )

    if boundint.size == 0:
        supp = np.array([[x_t1, x_tk]])
        omega = np.array([p])
    else:
        flat = np.concatenate([[x_t1], boundint[:, 0:2].ravel(), [x_tk]])
        supp = flat.reshape(-1, 2)
        omega = np.array(
            [int(np.sum(pw[(t > supp[i, 0]) & (t < supp[i, 1])])) for i in range(supp.shape[0])]
        )
    return supp, omega, boundint


def _min_sq_dist(t: Array, xi: Array) -> Array:
    diff = t.reshape(1, -1) - xi.reshape(-1, 1)
    return np.min(diff * diff, axis=1)


def _gamma(tau: Array, xi: Array, y: Array, n: int) -> Array:
    """``Gamma(y, xi) = sum_j tau_j^2 / ((tau_j - xi)^2 + y^2) - n``.

    Strictly decreasing in ``y >= 0`` per grid point, so vectorized
    bisection below returns the exact root (or the boundary when none
    exists — matching the reference's ``optimize(Gamma^2)`` limit).
    """
    d = (tau.reshape(1, -1) - xi.reshape(-1, 1)) ** 2 + y.reshape(-1, 1) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = (tau * tau).reshape(1, -1) / d
    terms = np.nan_to_num(terms, nan=np.inf)
    return np.sum(terms, axis=1) - float(n)


def quest_forward(tau: Array, n: int) -> QuestForward:
    """Discretized QuEST map: population spectrum -> sample eigenvalue law.

    ``tau`` is the ``p``-vector of population eigenvalues (nonnegative),
    ``n`` the effective sample size. Returns the forward solution with
    ``lam`` = the predicted sample eigenvalues (length ``p``, sorted).
    """
    n = int(n)
    if n <= 0:
        raise ValueError("quest requires a positive integer sample size")
    tau = np.asarray(tau, dtype=float)
    if tau.ndim != 1 or tau.size == 0 or not np.isfinite(tau).all():
        raise ValueError("quest population spectrum must be a finite 1D array")
    tau = np.sort(tau)
    tau[(tau < 0) & (np.abs(tau) < 1e-8)] = 0.0
    tau[(tau > 0) & (np.abs(tau) < 1e-12)] = 0.0
    if np.any(tau < 0):
        raise ValueError("quest population eigenvalues must be non-negative")
    if np.all(tau == 0):
        raise ValueError("quest population spectrum is entirely zero")
    p = int(tau.size)
    c = p / float(n)

    tau_nonzero = tau[tau > 0]
    t, pw_counts = np.unique(tau_nonzero, return_counts=True)
    pw: Array = pw_counts.astype(np.float64)
    pzw = p - int(np.sum(pw))
    pwt2 = pw * t * t

    supp, _omega, boundint = _spectral_support(tau, t, pw, pzw, n, p)
    u_fbar = np.asarray(supp, dtype=float)
    numint = u_fbar.shape[0]

    # Modified Stieltjes transform of the population law at support edges.
    m_lf_u_fbar = (1.0 / p) * np.sum(
        (pw * t).reshape(1, -1) / (t.reshape(1, -1) - u_fbar.reshape(-1, 1)),
        axis=1,
    ).reshape(numint, 2)
    endpoints = np.maximum(0.0, u_fbar - c * u_fbar * m_lf_u_fbar)
    if p == n:
        endpoints[0, 0] = 0.0

    # Sample eigenvalue counts per interval.
    numeig = np.empty(numint, dtype=int)
    if numint == 1:
        numeig[0] = p - max(pzw, p - n)
    else:
        pw1 = boundint[:, 2]
        numeig[0] = int(pw1[0]) - max(pzw, p - n)
        for i in range(1, numint - 1):
            numeig[i] = int(pw1[i] - pw1[i - 1])
        numeig[numint - 1] = int(boundint[numint - 2, 3])

    # Grid points (sine-squared spacing) per interval.
    nxi0 = max(QUEST_MIN_GRID, min(int(np.sum(pw)), n))
    mult = int(np.ceil(nxi0 / min(int(np.sum(pw)), n)))
    xi_list: list[Array] = []
    for i in range(numint):
        temp = int(numeig[i]) * mult
        frac = np.sin(np.pi / (2.0 * (temp + 1)) * np.arange(1, temp + 1)) ** 2
        xi_list.append(u_fbar[i, 0] + (u_fbar[i, 1] - u_fbar[i, 0]) * frac)

    # Solve the MP equation for the imaginary part at each grid point.
    zxi_list: list[NDArray[np.complex128]] = []
    for i in range(numint):
        xi = xi_list[i]
        y = np.zeros(xi.size)
        gamma0 = _gamma(tau, xi, y, n)
        y_hi = np.sqrt(np.maximum(np.sum(pwt2) / n - _min_sq_dist(t, xi), 0.0)) + 1.0
        lo = np.zeros(xi.size)
        hi = np.asarray(y_hi, dtype=float)
        active = gamma0 > 0.0
        for _ in range(QUEST_BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            g = _gamma(tau, xi, mid, n)
            up = g > 0.0
            lo = np.where(up, mid, lo)
            hi = np.where(up, hi, mid)
        y = np.where(active, 0.5 * (lo + hi), 0.0)
        zxi_list.append(xi.astype(complex) + 1j * y)

    # Sample density in x-space via the Silverstein map.
    a = float(QUEST_POWER)
    x_list: list[Array] = []
    f_list: list[Array] = []
    zeta_list: list[Array] = []
    g_list: list[Array] = []
    m_lf_zxi_list: list[NDArray[np.complex128]] = []
    for i in range(numint):
        z = zxi_list[i]
        denom = t.reshape(1, -1) - z.reshape(-1, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            m_lf = (1.0 / p) * np.sum((pw * t).reshape(1, -1) / denom, axis=1)
        m_lf_zxi_list.append(np.asarray(m_lf, dtype=np.complex128))
        x = np.real(z * (1.0 - c * m_lf))
        x = np.maximum(x, 0.0)
        f = np.imag(z) / (np.pi * c * np.abs(z) ** 2)
        zeta = x ** (1.0 / a)
        g = a * zeta ** (a - 1.0) * f
        x_list.append(x)
        f_list.append(np.asarray(f, dtype=float))
        zeta_list.append(np.asarray(zeta, dtype=float))
        g_list.append(np.asarray(np.nan_to_num(g, nan=0.0), dtype=float))

    # Discretized sample cdf: trapezoid over the zeta axis per interval.
    f0 = (1.0 / p) * max(pzw, p - n)
    fstart = f0 + np.concatenate([[0.0], np.cumsum(numeig[:-1])]) / p
    fend = f0 + np.cumsum(numeig) / p
    dis_x_list: list[Array] = []
    dis_zeta_list: list[Array] = []
    dis_m_lf_list: list[NDArray[np.complex128]] = []
    dis_g_list: list[Array] = []
    dis_g_raw: list[Array] = []
    dis_g_list_cum: list[Array] = []
    for i in range(numint):
        dis_zeta_list.append(
            np.concatenate(
                [
                    [endpoints[i, 0] ** (1.0 / a)],
                    zeta_list[i],
                    [endpoints[i, 1] ** (1.0 / a)],
                ]
            )
        )
        dis_x_list.append(np.concatenate([[endpoints[i, 0]], x_list[i], [endpoints[i, 1]]]))
        dis_m_lf_list.append(
            np.concatenate([[m_lf_u_fbar[i, 0]], m_lf_zxi_list[i], [m_lf_u_fbar[i, 1]]])
        )
        dis_g_list.append(np.concatenate([[0.0], g_list[i], [0.0]]))
        dzeta = np.diff(dis_zeta_list[i])
        g_mean = 0.5 * (dis_g_list[i][1:] + dis_g_list[i][:-1])
        g_raw = np.concatenate([[0.0], np.cumsum(dzeta * g_mean)])
        total = g_raw[-1]
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError(
                "quest spectral support interval has no sample mass; "
                "population spectrum is degenerate"
            )
        dis_g_raw.append(g_raw)
        dis_g_list_cum.append(fstart[i] + (fend[i] - fstart[i]) * g_raw / total)

    # Recover the sample eigenvalue quantiles by integrating x over dF.
    f_vals_list: list[Array] = []
    f_idx_list: list[NDArray[np.intp]] = []
    x_f_list: list[Array] = []
    x_f_mean_list: list[Array] = []
    x_f_diff_list: list[Array] = []
    f_diff_list: list[Array] = []
    quant_list: list[Array] = []
    bins_list: list[NDArray[np.intp]] = []
    integral_indic_list: list[Array] = []
    lam = np.zeros(p)
    for i in range(numint):
        f_vals, f_idx = np.unique(dis_g_list_cum[i], return_index=True)
        nidx = f_vals.size
        x_f = dis_zeta_list[i][f_idx] ** a
        x_f_mean = 0.5 * (x_f[1:] + x_f[:-1])
        x_f_diff = np.diff(x_f)
        f_diff = np.diff(f_vals)
        nquant = int(numeig[i]) + 1
        quant = np.linspace(f_vals[0], f_vals[-1], nquant)
        bins = np.clip(np.searchsorted(f_vals, quant, side="right") - 1, 0, nidx - 2)
        if bins[0] != 0 or bins[-1] != nidx - 2:
            raise ValueError("quest quantile bins fell outside the support grid")
        # Full segments below each quantile in F-position coordinates.
        indic = (np.arange(1, nidx).reshape(1, -1) <= bins.reshape(-1, 1)).astype(float)
        x_integral = f_diff * x_f_mean
        kappa = (quant - f_vals[bins]) * (
            x_f[bins] + 0.5 * (quant - f_vals[bins]) * x_f_diff[bins] / f_diff[bins]
        )
        x_cum = indic @ x_integral + kappa
        lam_lo = int(round(f_vals[0] * p))
        lam_hi = int(round(f_vals[-1] * p))
        lam[lam_lo:lam_hi] = np.diff(x_cum) * p
        f_vals_list.append(f_vals)
        f_idx_list.append(np.asarray(f_idx, dtype=np.intp))
        x_f_list.append(x_f)
        x_f_mean_list.append(x_f_mean)
        x_f_diff_list.append(x_f_diff)
        f_diff_list.append(f_diff)
        quant_list.append(quant)
        bins_list.append(np.asarray(bins, dtype=np.intp))
        integral_indic_list.append(indic)

    if not np.isfinite(lam).all() or np.any(lam < 0):
        raise ValueError("quest produced non-finite or negative sample eigenvalues")

    return QuestForward(
        p=p,
        n=n,
        c=c,
        tau=tau,
        t=t,
        pw=pw,
        pzw=pzw,
        numint=numint,
        u_fbar=u_fbar,
        endpoints=endpoints,
        m_lf_u_fbar=m_lf_u_fbar,
        numeig=numeig,
        f0=f0,
        fstart=fstart,
        fend=fend,
        xi_list=xi_list,
        zxi_list=zxi_list,
        m_lf_zxi_list=m_lf_zxi_list,
        x_list=x_list,
        f_list=f_list,
        zeta_list=zeta_list,
        g_list=g_list,
        dis_x_list=dis_x_list,
        dis_zeta_list=dis_zeta_list,
        dis_m_lf_list=dis_m_lf_list,
        dis_g_list=dis_g_list,
        dis_g_raw=dis_g_raw,
        dis_g_list_cum=dis_g_list_cum,
        f_vals_list=f_vals_list,
        f_idx_list=f_idx_list,
        x_f_list=x_f_list,
        x_f_mean_list=x_f_mean_list,
        x_f_diff_list=x_f_diff_list,
        f_diff_list=f_diff_list,
        quant_list=quant_list,
        bins_list=bins_list,
        integral_indic_list=integral_indic_list,
        lam=lam,
    )


def quest_lambda_jacobian(q: QuestForward) -> Array:
    """Analytic Jacobian of ``quest_forward(tau, n).lam`` w.r.t. sorted ``tau``.

    Port of the reference ``get_lambda_J``: the chain rule through the
    support endpoints, grid points, the MP imaginary-root map
    (implicit-function theorem on ``Gamma``), the Silverstein x-map, the
    zeta-transformed density, the trapezoid cdf, and the quantile-bucket
    integrals. Output shape is ``(p, p)``; row ``i`` is the derivative of
    the ``i``-th predicted sample eigenvalue.
    """
    p, n, c, a = q.p, q.n, q.c, float(QUEST_POWER)

    # Jacobian of support endpoints (u-space then x-space).
    u_vec = q.u_fbar  # (numint, 2)
    dr = np.sum(
        (q.tau**2).reshape(1, 1, -1) / (q.tau.reshape(1, 1, -1) - u_vec.reshape(-1, 2, 1)) ** 3,
        axis=2,
    )  # (numint, 2)
    temp = q.tau.reshape(1, 1, p) - u_vec.reshape(-1, 2, 1)  # tau_j - u_ib
    with np.errstate(divide="ignore", invalid="ignore"):
        sup_j = (u_vec.reshape(-1, 2, 1) * q.tau.reshape(1, 1, p)) / (
            temp**3 * dr.reshape(-1, 2, 1)
        )
    sup_j = np.nan_to_num(sup_j, nan=0.0, posinf=0.0, neginf=0.0)
    endpoints_j = (1.0 / n) * (u_vec.reshape(-1, 2, 1) ** 2) / temp**2
    if p == n:
        endpoints_j[0, 0, :] = 0.0

    xi_j_list: list[Array] = []
    for i in range(q.numint):
        m = q.xi_list[i].size
        temp_sin = np.sin(np.pi / (2.0 * (m + 1)) * np.arange(1, m + 1)) ** 2
        xi_j_list.append(
            np.outer(1.0 - temp_sin, sup_j[i, 0, :]) + np.outer(temp_sin, sup_j[i, 1, :])
        )

    zxi_j_list: list[NDArray[np.complex128]] = []
    f_j_list: list[Array] = []
    m_lf_j_list: list[NDArray[np.complex128]] = []
    x_j_list: list[Array] = []
    zeta_j_list: list[Array] = []
    g_j_list: list[Array] = []
    for i in range(q.numint):
        z = q.zxi_list[i]
        m_pts = z.size
        xi = np.real(z)
        y = np.imag(z)
        tau_m = np.tile(q.tau.reshape(1, -1), (m_pts, 1))
        tau2_m = tau_m * tau_m
        zxi_m = np.tile(z.reshape(-1, 1), (1, p))
        xi_m = np.tile(xi.reshape(-1, 1), (1, p))
        y2_m = np.tile((y * y).reshape(-1, 1), (1, p))
        tau_xi = tau_m - xi_m
        norm = tau_xi * tau_xi + y2_m
        norm2 = norm * norm
        with np.errstate(divide="ignore", invalid="ignore"):
            nr = np.sum(tau2_m * tau_xi / norm2, axis=1)
            dr_y = np.sum(tau2_m * np.tile(y.reshape(-1, 1), (1, p)) / norm2, axis=1)
            yxi_j = (
                tau_m / norm - tau2_m * tau_xi / norm2 + np.outer(nr, np.ones(p)) * xi_j_list[i]
            ) / np.outer(dr_y, np.ones(p))
        yxi_j = np.nan_to_num(yxi_j, nan=0.0, posinf=0.0, neginf=0.0)
        zxi_j = xi_j_list[i] + 1j * yxi_j
        zxi_j_list.append(zxi_j)

        f_j = (1.0 / (np.pi * c)) * np.imag(zxi_j * np.tile((1.0 / (z * z)).reshape(-1, 1), (1, p)))
        f_j_list.append(np.asarray(f_j, dtype=float))

        tau_zxi2 = (tau_m - zxi_m) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            m_lf_j = (1.0 / p) * (
                -zxi_m / tau_zxi2
                + zxi_j * np.tile(np.sum(tau_m / tau_zxi2, axis=1).reshape(-1, 1), (1, p))
            )
        m_lf_j_list.append(np.asarray(np.nan_to_num(m_lf_j), dtype=np.complex128))

        x_j = np.real(
            zxi_j * np.tile((1.0 - c * q.m_lf_zxi_list[i]).reshape(-1, 1), (1, p))
            - c * zxi_m * m_lf_j
        )
        x_j_list.append(np.asarray(x_j, dtype=float))
        zeta_j = (1.0 / a) * x_j * np.tile((q.x_list[i] ** (1.0 / a - 1.0)).reshape(-1, 1), (1, p))
        zeta_j_list.append(np.asarray(zeta_j, dtype=float))
        g_j = a * (
            (a - 1.0)
            * zeta_j
            * np.tile((q.f_list[i] * q.zeta_list[i] ** (a - 2.0)).reshape(-1, 1), (1, p))
            + np.tile((q.zeta_list[i] ** (a - 1.0)).reshape(-1, 1), (1, p)) * f_j
        )
        g_j_list.append(np.asarray(g_j, dtype=float))

    lambda_j = np.zeros((p, p))
    if (p - n) <= q.pzw and q.pzw > 0:
        lo = max(0, p - n)
        lambda_j[lo : q.pzw, 0 : q.pzw] = 1.0 - c

    for i in range(q.numint):
        dis_g_j = np.vstack([np.zeros((1, p)), g_j_list[i], np.zeros((1, p))])
        ep_lo_j = (1.0 / a) * q.endpoints[i, 0] ** (1.0 / a - 1.0) * endpoints_j[i, 0, :]
        ep_hi_j = (1.0 / a) * q.endpoints[i, 1] ** (1.0 / a - 1.0) * endpoints_j[i, 1, :]
        dis_zeta_j = np.vstack([ep_lo_j, zeta_j_list[i], ep_hi_j])
        if p == n and i == 0:
            dis_zeta_j[0, :] = 0.0
        dzeta_j = dis_zeta_j[1:, :] - dis_zeta_j[:-1, :]
        dzeta = np.diff(q.dis_zeta_list[i])
        dis_g_mean = 0.5 * (q.dis_g_list[i][1:] + q.dis_g_list[i][:-1])
        dis_g_j_mean = 0.5 * (dis_g_j[1:, :] + dis_g_j[:-1, :])
        g_j_raw = np.vstack(
            [
                np.zeros((1, p)),
                np.cumsum(
                    dzeta_j * dis_g_mean.reshape(-1, 1) + dzeta.reshape(-1, 1) * dis_g_j_mean,
                    axis=0,
                ),
            ]
        )
        total = q.dis_g_raw[i][-1]
        dis_g_j = (
            (q.fend[i] - q.fstart[i]) / total * (g_j_raw - np.outer(q.dis_g_raw[i], g_j_raw[-1, :]))
        )

        f_j = dis_g_j[q.f_idx_list[i], :]
        x_j_f = (
            np.outer(a * q.dis_zeta_list[i][q.f_idx_list[i]] ** (a - 1.0), np.ones(p))
            * (dis_zeta_j[q.f_idx_list[i], :])
        )
        f_j_diff = f_j[1:, :] - f_j[:-1, :]
        x_j_f_mean = 0.5 * (x_j_f[1:, :] + x_j_f[:-1, :])
        x_j_f_diff = x_j_f[1:, :] - x_j_f[:-1, :]
        x_j_integral = (
            f_j_diff * q.x_f_mean_list[i].reshape(-1, 1)
            + np.outer(q.f_diff_list[i], np.ones(p)) * x_j_f_mean
        )

        quant = q.quant_list[i]
        b = q.bins_list[i]
        s = quant - q.f_vals_list[i][b]
        fd = q.f_diff_list[i][b]
        kappa_j = (
            np.outer(quant, np.ones(p)) * x_j_f[b, :]
            - x_j_f[b, :] * np.outer(q.f_vals_list[i][b], np.ones(p))
            - f_j[b, :] * np.outer(q.x_f_list[i][b], np.ones(p))
            - f_j[b, :] * np.outer(s * q.x_f_diff_list[i][b] / fd, np.ones(p))
            + x_j_f_diff[b, :] * np.outer(0.5 * s * s / fd, np.ones(p))
            - f_j_diff[b, :] * np.outer(0.5 * s * s * q.x_f_diff_list[i][b] / fd**2, np.ones(p))
        )
        x_j_kappa = q.integral_indic_list[i] @ x_j_integral + kappa_j
        lo = int(round(q.f_vals_list[i][0] * p))
        hi = int(round(q.f_vals_list[i][-1] * p))
        lambda_j[lo:hi, :] = np.diff(x_j_kappa, axis=0) * p

    if not np.isfinite(lambda_j).all():
        raise ValueError("quest Jacobian produced non-finite entries")
    return lambda_j


def linear_shrinkage_eigenvalues(centered: Array, n_eff: int) -> Array:
    """LW-2004 linear shrinkage applied to sample eigenvalues (QuEST init)."""
    x = np.asarray(centered, dtype=float)
    n_obs, p = x.shape
    sample = (x.T @ x) / float(n_eff)
    lam = np.sort(np.linalg.eigvalsh(sample))
    lam[lam < 0] = 0.0
    if p > n_eff:
        lam[: p - n_eff] = 0.0
    z = x * x
    phi = float(
        np.sum(z.T @ z) / n_eff - 2.0 * np.sum((x.T @ x) * sample) / n_eff + np.sum(sample * sample)
    )
    gamma = float(np.sum((sample - np.mean(np.diag(sample)) * np.eye(p)) ** 2))
    if not np.isfinite(phi) or not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("linear shrinkage init is degenerate")
    shrinkage = min(1.0, max(0.0, phi / gamma / n_eff))
    lam_mean = float(np.mean(lam))
    return np.asarray(lam_mean + np.sqrt(1.0 - shrinkage) * (lam - lam_mean), dtype=np.float64)


def estimate_population_eigenvalues(
    sample_eigs: Array, n_eff: int, *, centered: Array
) -> tuple[Array, dict[str, float | int | str]]:
    """Invert QuEST: estimate the population spectrum from sample eigenvalues.

    ``sample_eigs`` are the sorted (ascending) eigenvalues of the sample
    covariance of ``centered`` with ``n_eff`` degrees of freedom; the
    optimizer minimizes the mean squared gap between QuEST-predicted and
    observed sample eigenvalues over bounded population spectra, warm-
    started from linear shrinkage, with the analytic Jacobian.
    """
    lam = np.asarray(sample_eigs, dtype=float)
    n_eff = int(n_eff)
    p = lam.size
    if p == 0 or not np.isfinite(lam).all() or n_eff <= 0:
        raise ValueError("population spectrum inversion needs finite eigenvalues and n_eff > 0")
    lam = np.sort(np.maximum(lam, 0.0))
    x0 = linear_shrinkage_eigenvalues(np.asarray(centered, dtype=float), n_eff)
    x0 = np.maximum(np.sort(x0), 0.0)
    lb = float(np.min(lam))
    ub = float(np.max(lam))
    if not np.isfinite(lb) or not np.isfinite(ub) or ub <= 0.0:
        raise ValueError("sample spectrum is degenerate for quest inversion")
    if lb >= ub:
        lb = max(0.0, ub * 1e-6)

    def objective(tau: Array) -> tuple[float, Array]:
        order = np.argsort(tau)
        q = quest_forward(tau[order], n_eff)
        lam_hat = q.lam[order]
        lam_j = quest_lambda_jacobian(q)[order, :]
        resid = lam_hat - lam
        return float(np.mean(resid * resid)), (2.0 / p) * (lam_j.T @ resid)

    try:
        out = minimize(
            objective,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=[(lb, ub)] * p,
            options={
                "maxiter": QUEST_LBFGSB_MAXITER,
                "ftol": 1e-15,
                "gtol": 1e-10,
                "maxcor": 20,
            },
        )
    except ValueError as exc:
        raise ValueError(f"quest inversion failed: {exc}") from exc
    if not np.isfinite(out.fun) or not np.isfinite(out.x).all():
        raise ValueError("quest inversion produced non-finite output")
    if out.status not in (0, 1, 2):  # L-BFGS-B convergence/f-limit/iter-limit codes
        raise ValueError(f"quest inversion did not converge: {out.message}")
    tau_hat = np.maximum(np.asarray(out.x, dtype=float), 0.0)
    return np.sort(tau_hat), {
        "quest_objective": float(out.fun),
        "quest_iterations": int(out.nit),
    }


def quest_shrunk_eigenvalues(q: QuestForward) -> Array:
    """Oracle nonlinear shrinkage map on the QuEST internals (nlshrink_est).

    Returns ``delta[i]``: the shrunk eigenvalue paired with the ``i``-th
    sorted sample eigenvalue, from integrating the LW oracle weight
    ``x / |1 - c * m_LF|^2`` over each quantile bucket of the discretized
    sample law. Handles the singular ``p > n`` null-space adjustment and
    the ``p == n`` zero-eigenvalue floor.
    """
    p, n, c = q.p, q.n, q.c
    delta = np.zeros(p)
    if (p - n) > q.pzw:
        lb = float(np.min(q.tau)) - float(np.sum(q.tau)) / n - 1.0
        ub = (p - q.pw[0]) * q.t[0] / p
        u0 = _brent_min_squared(lambda u: (np.sum(q.pw * q.t / (q.t - u)) - n) ** 2, lb, ub)
        delta[: p - n] = u0 / (1.0 - c)

    lim0 = n / float(np.sum(q.pw / q.t)) if (p == n and np.all(q.tau > 0)) else 0.0
    for i in range(q.numint):
        m_dis = q.dis_m_lf_list[i][q.f_idx_list[i]]
        dr = np.abs(1.0 - c * m_dis) ** 2
        x_f = q.x_f_list[i]
        if p == n:
            mask = (x_f == 0.0) & (dr == 0.0)
            y = np.where(mask, lim0, np.where(mask, 0.0, x_f / np.where(dr == 0.0, 1.0, dr)))
            y = np.where(mask, lim0, y)
        else:
            y = x_f / dr
        f_vals = q.f_vals_list[i]
        quant = q.quant_list[i]
        b = q.bins_list[i]
        f_diff = q.f_diff_list[i]
        integ_ydF = f_diff * 0.5 * (y[1:] + y[:-1])
        partial = (quant - f_vals[b]) * (
            y[b] + 0.5 * (quant - f_vals[b]) * (y[b + 1] - y[b]) / f_diff[b]
        )
        cum = q.integral_indic_list[i] @ integ_ydF + partial
        lo = int(round(f_vals[0] * p))
        hi = int(round(f_vals[-1] * p))
        delta[lo:hi] = np.diff(cum) * p
    if p == n:
        delta = np.maximum(delta, lim0)
    if not np.isfinite(delta).all() or np.any(delta < 0):
        raise ValueError("quest shrinkage map produced non-finite weights")
    return delta


def quest_covariance(centered: Array, n_eff: int) -> tuple[Array, dict[str, float | int | str]]:
    """Numerical-QuEST nonlinear shrinkage covariance of a centered matrix.

    Eigendecompose ``S = X'X / n_eff``, invert QuEST for the population
    spectrum ``tau``, recompute the forward map's oracle shrinkage, and
    reassemble ``U diag(delta) U'``. Returns the covariance and the
    inversion diagnostics. Raises on any degenerate step — callers must
    not substitute another estimator on failure.
    """
    x = np.asarray(centered, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("quest covariance needs a non-empty finite 2D centered matrix")
    if n_eff < 2:
        raise ValueError("quest covariance needs n_eff >= 2")
    n_obs, p = x.shape
    sample = (x.T @ x) / float(n_eff)
    sample = 0.5 * (sample + sample.T)
    if not np.isfinite(sample).all():
        raise ValueError("quest sample covariance is non-finite")
    lam, vecs = np.linalg.eigh(sample)
    lam = np.maximum(np.asarray(lam, dtype=float), 0.0)
    if not np.isfinite(lam).all() or float(np.sum(lam)) <= 0.0:
        raise ValueError("quest sample spectrum is degenerate")
    tau_hat, info = estimate_population_eigenvalues(lam, n_eff, centered=x)
    q = quest_forward(tau_hat, n_eff)
    delta = quest_shrunk_eigenvalues(q)
    sigma = (vecs * delta.reshape(1, -1)) @ vecs.T
    sigma = 0.5 * (sigma + sigma.T)
    if not np.isfinite(sigma).all():
        raise ValueError("quest covariance produced non-finite values")
    return sigma, {
        **info,
        "quest_p": p,
        "quest_n_obs": n_obs,
        "quest_n_eff": n_eff,
        "quest_numint": int(q.numint),
        "quest_tau_min": float(np.min(tau_hat)),
        "quest_tau_max": float(np.max(tau_hat)),
    }


__all__ = [
    "QUEST_MIN_GRID",
    "QUEST_POWER",
    "QuestForward",
    "estimate_population_eigenvalues",
    "linear_shrinkage_eigenvalues",
    "quest_covariance",
    "quest_forward",
    "quest_lambda_jacobian",
    "quest_shrunk_eigenvalues",
]
