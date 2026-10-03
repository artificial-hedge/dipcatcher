"""Model-agnostic MSE-optimal noise reduction for high-dimensional series.

Implements the estimator stack of Wouters & Diks (2026) for a panel
``y_t = x_t + eps_t`` (t = 1..T, y_t in R^n) where the latent signal
``x_t = U xi_t`` lives in a d-dimensional *dynamic space* M (d << n) that
carries all serial dependence, and ``eps_t`` is white noise with arbitrary
cross-sectional covariance whose span (the *noise space* M_eps) may be
oblique to, or partly intersect, M.

Pipeline (paper section 2.3):

1. Dynamic space.  Because the lagged autocovariance of white noise vanishes,
   ``Sigma_y(k) = Sigma_x(k)`` for k != 0 and the matrix

       K = sum_{k=1..k0} c_k Sigma_y(k) Sigma_y(k)^T          (Eq. 2.1)

   has exactly d non-zero eigenvalues, whose eigenvectors span M (Lam, Yao &
   Bathia 2011; Bathia, Yao & Ziegelmann 2010).  ``estimate_dynamic_subspace``
   runs the eigenanalysis of the sample K-hat built from naive lagged
   autocovariances — unbiased for M even though Sigma_y(0) is noise-biased.
2. Dimension.  ``bootstrap_dimension_select`` runs the sequential residual-
   resampling tests of H0: d = d0 (paper Appendix A, Algorithm 1; Bathia et
   al. 2010): project onto the first d0 eigenvectors, resample the residual
   vectors iid over t (destroying any serial dependence beyond d0), and
   compare the observed (d0+1)-th eigenvalue of K-hat with its bootstrap
   null distribution.
3. MSE-optimal (oblique) projection.  Among all linear projections onto M,
   the MSE-minimizer (paper Theorem 1, Eq. 2.2) is

       P_opt = U U^T (I_n - Sigma_y W_perp Sigma_chi^{-1} W_perp^T),

   where the columns of W_perp hold the leading eigenvectors of the
   orthogonal-noise covariance Sigma_eps_perp = (I - V V^T) Sigma_y (I - V V^T)
   (a low-rank structured-noise representation, dimension chosen by the
   explained-variance rule with cutoff tau_perp) and Sigma_chi holds the
   matching eigenvalues.  The correction regresses the parallel noise eps_par
   on the observable orthogonal noise eps_perp; P_opt is generally OBLIQUE
   (non-symmetric).  Geometric characterization (paper Theorem 2): if
   M ∩ M_eps = {0} optimal denoising is asymptotically perfect, while
   orthogonal projection always leaves eps_par in the signal; any shared
   subspace contributes an irreducible error term.  When Cov[eps_par,
   eps_perp] = 0 (e.g. iid isotropic noise) P_opt collapses to the
   orthogonal projection V V^T.
4. Baseline.  ``orthogonal_projection_denoise`` is P_ortho = V V^T
   (Lam et al. 2011-style denoising), the contrast method of the paper.

Under Assumptions 1-5 (paper section 2.4, Theorem 3) the estimated optimal
denoiser converges to its population target at the parametric rate
O_P(T^{-1/2}) — square-root-T consistency.

Any validation shipped with this module uses SYNTHETIC planted data and is a
correctness test only — never market evidence.

References:
- Wouters, Diks (2026). Model-agnostic noise reduction for high-dimensional
  time series data. arXiv:2609.27614 (stat.ME / q-fin.ST).
- Lam, Yao, Bathia (2011). Estimation of latent factors for high-dimensional
  time series. *Biometrika* 98(4), 901-918 — lagged-autocovariance K matrix.
- Bathia, Yao, Ziegelmann (2010). Identifying the finite dimensionality of
  curve time series. *Annals of Statistics* 38(6) — bootstrap dimension test.
- Pan, Yao (2008). Modelling multiple time series via common factors.
  *Biometrika* 95(2), 365-379.

Fail-closed: non-finite panels, out-of-range lags/dimensions/thresholds,
non-positive K coefficients, or degenerate noise structure all raise
``ValueError`` rather than silently returning a distorted projection.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

# Minimum panel length: enough rows for lag-k autocovariances and eigensteps
# to be meaningful at all. Callers needing statistical power use far more.
_MIN_ROWS = 24


def _as_panel(y: Array, name: str = "y") -> Array:
    m = np.asarray(y, dtype=float)
    if m.ndim != 2 or not np.all(np.isfinite(m)):
        raise ValueError(f"{name} must be a finite 2-D (T x n) panel")
    t, n = m.shape
    if t < _MIN_ROWS or n < 2:
        raise ValueError(f"{name} must have >= {_MIN_ROWS} rows and >= 2 columns")
    return m


def _check_dim(d: int, n: int) -> int:
    if isinstance(d, bool) or not isinstance(d, (int, np.integer)):
        raise ValueError("d must be an integer")
    di = int(d)
    if di < 1 or di > n - 1:
        raise ValueError("d must be in [1, n-1] (a non-trivial dynamic space)")
    return di


def _check_coefficients(k0: int, coefficients: Sequence[float] | None, t: int) -> Array:
    if isinstance(k0, bool) or not isinstance(k0, (int, np.integer)):
        raise ValueError("k0 must be an integer")
    k0i = int(k0)
    if k0i < 1 or k0i > t - 2:
        raise ValueError(f"k0 must be in [1, T-2] (got k0={k0i}, T={t})")
    if coefficients is None:
        return np.ones(k0i, dtype=float)
    c = np.asarray(coefficients, dtype=float).reshape(-1)
    if c.size != k0i or not np.all(np.isfinite(c)):
        raise ValueError("coefficients must be a finite vector of length k0")
    # The paper allows arbitrary signs (Eq. 2.1); we require c_k > 0 so that
    # K-hat stays positive semi-definite and the eigenstep is well posed.
    if np.any(c <= 0.0):
        raise ValueError("coefficients must be strictly positive")
    return c


def lagged_autocovariance(y: Array, k: int) -> Array:
    """Naive sample lagged autocovariance ``Sigma_hat_y(k)``.

    ``1/(T-k) * sum_t (y_{t+k} - ybar)(y_t - ybar)^T`` (Wouters & Diks 2026,
    section 2.3).  Unbiased for the *signal* autocovariance at k != 0 since
    white noise has vanishing lagged covariance; noise-biased at k = 0.
    """
    m = _as_panel(y)
    t = m.shape[0]
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer")
    ki = int(k)
    if ki < 0 or ki > t - 2:
        raise ValueError(f"k must be in [0, T-2] (got k={ki}, T={t})")
    mc = m - m.mean(axis=0)
    cov = mc[ki:].T @ mc[: t - ki] / float(t - ki)
    if ki == 0:
        cov = (cov + cov.T) / 2.0
    return np.asarray(cov, dtype=float)


def dynamic_space_matrix(
    y: Array, k0: int = 2, coefficients: Sequence[float] | None = None
) -> Array:
    """Sample ``K_hat = sum_{k=1..k0} c_k Sigma_hat_y(k) Sigma_hat_y(k)^T``.

    Wouters & Diks (2026) Eq. 2.1 with the default ``c_1 = c_2 = 1``
    (k0 = 2) used throughout the paper.  Its non-zero eigenvalues number
    exactly d and their eigenvectors span the dynamic space M.
    """
    m = _as_panel(y)
    t, n = m.shape
    c = _check_coefficients(k0, coefficients, t)
    k = np.zeros((n, n), dtype=float)
    for lag in range(1, int(k0) + 1):
        s = lagged_autocovariance(m, lag)
        k += float(c[lag - 1]) * (s @ s.T)
    k = (k + k.T) / 2.0
    if not np.all(np.isfinite(k)):
        raise ValueError("K matrix is not finite — panel is degenerate")
    return k


def _eigen_descending(k: Array) -> tuple[Array, Array]:
    vals, vecs = np.linalg.eigh(k)
    order = np.argsort(vals)[::-1]
    vals = np.asarray(vals[order], dtype=float)
    vecs = np.asarray(vecs[:, order], dtype=float)
    if not np.all(np.isfinite(vals)) or not np.all(np.isfinite(vecs)):
        raise ValueError("K eigendecomposition produced non-finite values")
    return vals, vecs


def bootstrap_dimension_select(
    y: Array,
    *,
    seed: int,
    n_boot: int = 300,
    alpha: float = 0.05,
    k0: int = 2,
    coefficients: Sequence[float] | None = None,
    max_d: int | None = None,
) -> dict[str, Any]:
    """Sequential bootstrap tests for the dynamic-space dimension d.

    Wouters & Diks (2026) Appendix A, Algorithm 1 (Bathia, Yao & Ziegelmann
    2010): for d0 = 1, 2, ..., test H0: d = d0 via its implication that the
    (d0+1)-th eigenvalue theta_{d0+1} of K is zero.  Pseudo-samples
    ``y*_t = ybar + V_{d0} eta_t + eps*_t`` keep the fitted first-d0 factor
    dynamics and resample the residual vectors ``eps*_t`` iid over t (with
    replacement), which destroys any serial dependence beyond d0.  H0 is
    rejected when ``theta_hat_{d0+1}`` lies in the UPPER tail of the
    bootstrap null: with ``counter_ge = #{theta*_b >= theta_obs}``, reject
    when ``counter_ge <= floor(alpha * n_boot)`` (empirical p-value <= alpha,
    the Bathia-Yao-Ziegelmann convention); the first non-rejected d0 is
    returned.  Paper defaults: ``n_boot=300``, ``alpha=0.05``.

    Deterministic given ``seed``.  Returns dict with:
      - ``d``: selected dimension (>= 1);
      - ``p_values``: per-d0 empirical p-values #{theta* >= theta_obs}/B
        (small p-value => reject H0: d = d0 => keep increasing d0);
      - ``eigenvalues``: descending eigenvalues of K_hat;
      - ``capped``: True if the search hit ``max_d`` while still rejecting.
    """
    m = _as_panel(y)
    t, n = m.shape
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer (deterministic bootstrap)")
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)):
        raise ValueError("n_boot must be an integer")
    nb = int(n_boot)
    if nb < 1:
        raise ValueError("n_boot must be >= 1")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    cap = n - 1
    if max_d is not None:
        if isinstance(max_d, bool) or not isinstance(max_d, (int, np.integer)):
            raise ValueError("max_d must be an integer or None")
        if int(max_d) < 1 or int(max_d) > n - 1:
            raise ValueError("max_d must be in [1, n-1]")
        cap = int(max_d)

    rng = np.random.default_rng(int(seed))
    k_hat = dynamic_space_matrix(m, k0, coefficients)
    c = _check_coefficients(k0, coefficients, t)
    theta, vecs = _eigen_descending(k_hat)
    yc = m - m.mean(axis=0)
    threshold = math.floor(float(alpha) * nb)

    d0 = 0
    reject = True
    p_values: list[float] = []
    while reject and d0 < cap:
        d0 += 1
        v = vecs[:, :d0]
        fit = yc @ v @ v.T  # first-d0 factor reconstruction (centered)
        resid = yc - fit  # residual vectors, resampled iid under H0
        theta_obs = float(theta[d0])  # (d0+1)-th eigenvalue, 0-indexed
        counter_ge = 0
        for _ in range(nb):
            idx = rng.integers(0, t, t)
            ystar = fit + resid[idx]
            ystar -= ystar.mean(axis=0)
            ks = np.zeros((n, n), dtype=float)
            for lag in range(1, int(k0) + 1):
                s = ystar[lag:].T @ ystar[: t - lag] / float(t - lag)
                ks += float(c[lag - 1]) * (s @ s.T)
            theta_star = float(np.linalg.eigvalsh((ks + ks.T) / 2.0)[n - d0 - 1])
            if theta_star >= theta_obs:
                counter_ge += 1
        # Empirical p-value (Bathia-Yao-Ziegelmann): reject H0: d = d0 when
        # the observed eigenvalue sits in the upper tail of the bootstrap
        # null, i.e. few bootstrap draws reach it.
        p_values.append(counter_ge / nb)
        reject = counter_ge <= threshold
    return {
        "d": d0,
        "p_values": p_values,
        "eigenvalues": theta,
        "capped": bool(reject and d0 >= cap),
    }


def estimate_dynamic_subspace(
    y: Array,
    d: int | None = None,
    *,
    k0: int = 2,
    coefficients: Sequence[float] | None = None,
    seed: int | None = None,
    n_boot: int = 300,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Estimate an orthonormal basis V_hat of the dynamic space M.

    V_hat holds the eigenvectors of K_hat (Eq. 2.1) for the d largest
    eigenvalues (Wouters & Diks 2026, section 2.3; Lam, Yao & Bathia 2011).
    ``d`` may be given (oracle) or, when ``d is None``, selected by
    :func:`bootstrap_dimension_select` — which requires ``seed``.

    Returns dict with ``basis`` (n x d), ``eigenvalues`` (descending, full
    spectrum of K_hat), ``d``, ``k_matrix``, and ``bootstrapped`` (bool).
    """
    m = _as_panel(y)
    n = m.shape[1]
    k_hat = dynamic_space_matrix(m, k0, coefficients)
    vals, vecs = _eigen_descending(k_hat)
    bootstrapped = d is None
    if bootstrapped:
        if seed is None:
            raise ValueError("d=None requires an integer seed for the bootstrap")
        d = bootstrap_dimension_select(
            m,
            seed=seed,
            n_boot=n_boot,
            alpha=alpha,
            k0=k0,
            coefficients=coefficients,
        )["d"]
    di = _check_dim(int(d), n)  # type: ignore[arg-type]
    return {
        "basis": np.asarray(vecs[:, :di], dtype=float),
        "eigenvalues": vals,
        "d": di,
        "k_matrix": k_hat,
        "bootstrapped": bootstrapped,
    }


def _orthogonal_noise_representation(
    sigma_y0: Array, v: Array, tau_perp: float
) -> tuple[Array, Array]:
    """Low-rank structured-noise representation (Wouters & Diks 2026, 2.3).

    Eigendecomposes ``Sigma_hat_eps_perp = (I - V V^T) Sigma_hat_y(0)
    (I - V V^T)`` and keeps the leading d_perp eigenvectors W_perp with the
    explained-variance rule ``min{d: cumsum >= tau_perp * total}``.  Returns
    ``(w_perp, chi)`` with chi the matching (positive) eigenvalues, the
    diagonal of Sigma_hat_chi.
    """
    if not np.isfinite(tau_perp) or not 0.0 < tau_perp < 1.0:
        raise ValueError("tau_perp must be in (0, 1)")
    n = sigma_y0.shape[0]
    proj = np.eye(n) - v @ v.T
    sig_perp = proj @ sigma_y0 @ proj
    sig_perp = (sig_perp + sig_perp.T) / 2.0
    vals, vecs = np.linalg.eigh(sig_perp)
    vals = np.clip(vals[::-1], 0.0, None)
    vecs = np.asarray(vecs[:, ::-1], dtype=float)
    total = float(vals.sum())
    scale = float(np.abs(sigma_y0).max())
    if not np.isfinite(total) or total <= max(scale, 1.0) * 1e-12:
        raise ValueError("orthogonal-noise covariance is degenerate — nothing to denoise")
    cum = np.cumsum(vals)
    d_perp = int(np.searchsorted(cum, tau_perp * total) + 1)
    d_perp = max(1, min(d_perp, n))
    w_perp = vecs[:, :d_perp]
    chi = vals[:d_perp]
    # Pseudo-inverse of Sigma_chi: drop numerically-zero retained directions.
    tol = float(chi[0]) * n * np.finfo(float).eps
    inv_chi = np.where(chi > tol, 1.0 / np.maximum(chi, tol), 0.0)
    return np.asarray(w_perp, dtype=float), np.asarray(inv_chi, dtype=float)


def optimal_projection_denoise(
    y: Array,
    d: int | None = None,
    *,
    k0: int = 2,
    coefficients: Sequence[float] | None = None,
    tau_perp: float = 0.95,
    seed: int | None = None,
    n_boot: int = 300,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """MSE-optimal (generally oblique) projection denoiser.

    ``P_opt = V V^T (I - Sigma_hat_y(0) W_perp Sigma_chi^{-1} W_perp^T)``
    (Wouters & Diks 2026, Theorem 1 / Eq. 2.2, estimated per section 2.3).
    Among all linear projections onto the dynamic space this minimizes
    ``E[||x_t - P y_t||^2]``: it removes the observable orthogonal noise
    eps_perp *and* regresses out the part of the parallel noise eps_par that
    eps_perp predicts.  Unlike ``V V^T`` it is non-symmetric whenever
    Cov[eps_par, eps_perp] != 0.  ``d=None`` triggers bootstrap dimension
    selection (requires ``seed``); ``tau_perp`` is the explained-variance
    cutoff for the noise rank (paper default 0.95).

    Returns dict with ``projection`` (n x n), ``denoised`` (T x n),
    ``basis``, ``w_perp``, ``inv_chi`` (diagonal of Sigma_chi^{-1}),
    ``d_perp``, ``d``, ``eigenvalues`` (of K_hat).
    """
    m = _as_panel(y)
    sub = estimate_dynamic_subspace(
        m, d, k0=k0, coefficients=coefficients, seed=seed, n_boot=n_boot, alpha=alpha
    )
    v: Array = sub["basis"]
    sigma_y0 = lagged_autocovariance(m, 0)
    w_perp, inv_chi = _orthogonal_noise_representation(sigma_y0, v, tau_perp)
    n = m.shape[1]
    correction = sigma_y0 @ w_perp @ (inv_chi[:, None] * w_perp.T)
    p = v @ v.T @ (np.eye(n) - correction)
    if not np.all(np.isfinite(p)):
        raise ValueError("optimal projection is not finite — panel is degenerate")
    denoised = m @ p.T
    return {
        "projection": np.asarray(p, dtype=float),
        "denoised": np.asarray(denoised, dtype=float),
        "basis": v,
        "w_perp": w_perp,
        "inv_chi": inv_chi,
        "d_perp": int(w_perp.shape[1]),
        "d": int(sub["d"]),
        "eigenvalues": sub["eigenvalues"],
    }


def orthogonal_projection_denoise(
    y: Array,
    d: int | None = None,
    *,
    k0: int = 2,
    coefficients: Sequence[float] | None = None,
    seed: int | None = None,
    n_boot: int = 300,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Orthogonal-projection baseline ``P_ortho = V V^T``.

    Lam et al. (2011)-style denoising as contrasted in Wouters & Diks (2026,
    section 2.1): keeps the full component of y_t inside the estimated
    dynamic space, including the parallel noise eps_par.  Optimal denoising
    weakly dominates it in MSE, strictly whenever Cov[eps_par, eps_perp] != 0.

    Returns dict with ``projection``, ``denoised``, ``basis``, ``d``,
    ``eigenvalues``.
    """
    m = _as_panel(y)
    sub = estimate_dynamic_subspace(
        m, d, k0=k0, coefficients=coefficients, seed=seed, n_boot=n_boot, alpha=alpha
    )
    v: Array = sub["basis"]
    p = v @ v.T
    p = np.asarray((p + p.T) / 2.0, dtype=float)
    if not np.all(np.isfinite(p)):
        raise ValueError("orthogonal projection is not finite — panel is degenerate")
    return {
        "projection": p,
        "denoised": np.asarray(m @ p.T, dtype=float),
        "basis": v,
        "d": int(sub["d"]),
        "eigenvalues": sub["eigenvalues"],
    }


def principal_angles(a: Array, b: Array) -> Array:
    """Principal (canonical) angles in radians between span(a) and span(b).

    ``a`` (n x p) and ``b`` (n x q) must have orthonormal columns.  Returns
    ``min(p, q)`` angles in ascending order: 0 means a shared direction (for
    dynamic vs noise spaces, an irreducible-error intersection per Wouters &
    Diks 2026, Theorem 2), pi/2 means orthogonal.  This is the geometric
    language in which the paper characterizes the residual denoising error —
    the relative orientation of the signal and noise spaces.
    """
    ma = np.asarray(a, dtype=float)
    mb = np.asarray(b, dtype=float)
    if ma.ndim != 2 or mb.ndim != 2 or ma.shape[0] != mb.shape[0]:
        raise ValueError("a and b must be 2-D with the same number of rows")
    if ma.shape[1] < 1 or mb.shape[1] < 1:
        raise ValueError("a and b must have at least one column")
    if not np.all(np.isfinite(ma)) or not np.all(np.isfinite(mb)):
        raise ValueError("a and b must be finite")
    for name, mm in (("a", ma), ("b", mb)):
        gram = mm.T @ mm
        if np.max(np.abs(gram - np.eye(mm.shape[1]))) > 1e-6:
            raise ValueError(f"{name} must have orthonormal columns")
    svals = np.linalg.svd(ma.T @ mb, compute_uv=False)
    svals = np.clip(np.asarray(svals, dtype=float), -1.0, 1.0)
    # Values within machine epsilon of 1 are roundoff noise in an exact overlap;
    # clipping them prevents arccos from magnifying the noise into a near-zero
    # principal angle (especially sensitive to BLAS differences on Windows).
    svals = np.where(np.isclose(svals, 1.0, atol=np.finfo(float).eps), 1.0, svals)
    return np.asarray(np.arccos(svals), dtype=float)
