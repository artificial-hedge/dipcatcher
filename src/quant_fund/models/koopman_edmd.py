"""Koopman-operator spectral analysis for nonlinear regime dynamics.

Lifted observable dictionaries + Extended Dynamic Mode Decomposition (EDMD):
given snapshots ``x_m`` from a nonlinear dynamical system, the Koopman operator
``K`` acts linearly on observables ``psi``; EDMD approximates it on a finite
dictionary via

    G = (1/M) sum_m psi(x_m) psi(x_m)^T
    A = (1/M) sum_m psi(x_m) psi(x_{m+1})^T
    K = pinv(G) A

so ``psi(x_{m+1}) ~= K^T psi(x_m)`` pointwise in expectation. Its eigenvalues
give the system's intrinsic timescales/frequencies; its eigenfunctions give
slow-mode coordinates usable for projection forecasts and drift detection.

Hankel-DMD (delay-embedded trajectory DMD) is the special case where the
observable dictionary is exactly the delay coordinates; :func:`edmd_fit` with
``dictionary="delays"`` therefore reproduces :func:`hankel_dmd` output.

References
----------
- Williams, Kevrekidis & Rowley (2015). A data-driven approximation of the
  Koopman operator: extending dynamic mode decomposition. *Journal of
  Nonlinear Science* 25(6). arXiv:1410.1401.
- Brunton, Brunton, Proctor & Kutz (2017). Chaos as an intermittently forced
  linear system. *Nature Communications* 8. arXiv:1608.05306.
- Arbabi & Mezic (2017). Ergodic theory, dynamic mode decomposition, and
  computation of spectral properties of the Koopman operator. *SIAM J. Appl.
  Dyn. Syst.* 16(4). arXiv:1611.06676.
- Kutz, Brunton, Brunton & Proctor (2016). *Dynamic Mode Decomposition:
  Data-Driven Modeling of Complex Systems*. SIAM (journal/book — no arXiv).

Honesty
-------
All benches here run on seeded SYNTHETIC dynamical systems (linear
oscillator, chaotic logistic map, Van der Pol oscillator, regime-switching
AR) generated inside this module. They are correctness tests for the
operator machinery — they are not evidence about any real market.

Composition notes
-----------------
- ``models/regime.py``: unsupervised regime inference; the drift detector
  below is an orthogonal, spectrum-driven regime-change statistic.
- ``models/spectral.py``: frequency-domain tools; EDMD eigenvalues give the
  operator-theoretic spectrum rather than Fourier power.
- ``models/leadlag.py``: delay structure; the delay embedding here is a
  multi-tap Hankel view of a scalar series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.linalg import pinv
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_series(x: FloatArray, min_len: int = 8) -> FloatArray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError("series must be one-dimensional")
    if a.size < min_len:
        raise ValueError(f"series too short: {a.size} < {min_len}")
    if not np.isfinite(a).all():
        raise ValueError("series contains non-finite values")
    return a


def delay_embed(x: FloatArray, dim: int, delay: int = 1) -> FloatArray:
    """Hankel delay embedding: rows [x_t, x_{t-delay}, ..., x_{t-(dim-1)delay}]."""
    a = _check_series(x, min_len=(dim - 1) * delay + 2)
    if dim < 2 or delay < 1:
        raise ValueError("dim>=2 and delay>=1 required")
    n = a.size - (dim - 1) * delay
    if n < 3:
        raise ValueError("not enough embedded rows")
    cols = [a[(dim - 1 - k) * delay : (dim - 1 - k) * delay + n] for k in range(dim)]
    return np.column_stack(cols)


def _poly_features(z: FloatArray, degree: int) -> FloatArray:
    """Degree-1..degree polynomial features on the last coordinate."""
    last = z[:, -1:]
    feats = [last**p for p in range(1, degree + 1)]
    return np.hstack([z, *feats[1:]]) if degree >= 2 else z


def observables(z: FloatArray, dictionary: str = "delays", degree: int = 3) -> FloatArray:
    """Observable map psi(Z) per dictionary choice.

    ``delays``: identity on the delay embedding (Hankel-DMD equivalent).
    ``poly``: delay embedding plus polynomial features of the last coordinate.
    ``fourier``: delay embedding plus sin/cos of the last coordinate.
    """
    if dictionary == "delays":
        return z
    if dictionary == "poly":
        return _poly_features(z, degree)
    if dictionary == "fourier":
        last = z[:, -1:]
        return np.hstack([z, np.sin(last), np.cos(last)])
    raise ValueError(f"unknown dictionary {dictionary!r}")


@dataclass(frozen=True)
class EDMDResult:
    """Estimated Koopman operator on the observable space."""

    operator: FloatArray  # K (psi' ~= K^T psi): columns map feature -> next feature
    eigenvalues: NDArray[np.complex128]
    eigenfunctions: NDArray[np.complex128]  # right eigenvectors, columns
    residual: float  # mean ||K^T psi - psi_next|| / mean ||psi_next||
    dictionary: str
    n_observables: int

    @property
    def implied_timescales(self) -> FloatArray:
        """-1 / log|lambda| per eigenvalue (inf for |lambda| ~ 0 or 1)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            tau = -1.0 / np.log(np.abs(self.eigenvalues))
        return np.real(tau)

    def top_modes(self, r: int) -> NDArray[np.int64]:
        """Indices of the r slowest eigenmodes (|lambda| closest to 1)."""
        dist = np.abs(np.abs(self.eigenvalues) - 1.0)
        return np.argsort(dist)[: max(1, r)]


def edmd_fit(
    z: FloatArray,
    dictionary: str = "delays",
    degree: int = 3,
) -> EDMDResult:
    """Estimate the Koopman operator on pairs (z_m, z_{m+1}).

    ``z`` is an (M, d) array of system snapshots — use :func:`delay_embed`
    on a scalar series for the standard Hankel lift.
    """
    a = np.asarray(z, dtype=float)
    if a.ndim != 2 or a.shape[0] < 4 or a.shape[1] < 1:
        raise ValueError("z must be an (M>=4, d) snapshot array")
    if not np.isfinite(a).all():
        raise ValueError("z contains non-finite values")
    psi = observables(a[:-1], dictionary, degree)
    psi_next = observables(a[1:], dictionary, degree)
    m = psi.shape[0]
    g = (psi.T @ psi) / m
    a_mat = (psi.T @ psi_next) / m
    k = pinv(g) @ a_mat
    # Koopman acts psi' = K^T psi  =>  store the forward map on feature space
    fwd = k.T
    evals, evecs = np.linalg.eig(fwd)
    pred = psi @ fwd.T
    denom = float(np.linalg.norm(psi_next))
    resid = float(np.linalg.norm(psi_next - pred) / denom) if denom > 0 else 0.0
    return EDMDResult(
        operator=fwd,
        eigenvalues=evals,
        eigenfunctions=evecs,
        residual=resid,
        dictionary=dictionary,
        n_observables=psi.shape[1],
    )


def hankel_dmd(x: FloatArray, dim: int, delay: int = 1) -> EDMDResult:
    """Classic DMD on the delay-embedded trajectory (Hankel matrix)."""
    return edmd_fit(delay_embed(x, dim, delay), dictionary="delays")


def koopman_forecast(
    res: EDMDResult,
    z_last: FloatArray,
    steps: int,
    rank: int | None = None,
) -> FloatArray:
    """Project-ahead forecast: advance the observable vector by K each step.

    ``z_last`` is the last snapshot row (already in the lifted coordinates —
    for ``delays`` dictionary, a delay vector). Returns an (steps, d) array
    of observable trajectories; for delays dictionary the first d columns
    trace the delay coordinates.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    z0 = np.asarray(z_last, dtype=float).ravel()
    if z0.size != res.n_observables:
        # Lift raw delay coordinates into the dictionary space
        z0 = observables(z0.reshape(1, -1), res.dictionary).ravel()
    if z0.size != res.n_observables:
        raise ValueError("z_last does not match observable dimension")
    op = res.operator
    if rank is not None and rank < op.shape[0]:
        idx = res.top_modes(rank)
        eig = np.linalg.eig(op)
        lam = np.zeros(op.shape[0], dtype=complex)
        lam[idx] = eig.eigenvalues[idx]
        op = np.real(eig.eigenvectors @ np.diag(lam) @ np.linalg.pinv(eig.eigenvectors))
    out = np.empty((steps, z0.size))
    cur = z0
    for t in range(steps):
        cur = op.T @ cur
        out[t] = cur
    return out


def eigenfunction_drift(
    x: FloatArray,
    window: int,
    step: int | None = None,
    dim: int = 6,
    rank: int = 4,
) -> FloatArray:
    """Distance of successive eigenfunction-weighted window embeddings.

    For each window, embed the window's series, project onto the global
    top-``rank`` eigenfunctions, and record the L2 distance between
    consecutive window projections — large jumps flag regime change.
    """
    a = _check_series(x, min_len=window + 8)
    if window < 16:
        raise ValueError("window must be >= 16")
    step = step or window // 2
    if step < 1:
        raise ValueError("step must be positive")
    res_full = hankel_dmd(a, dim=dim)
    idx = res_full.top_modes(rank)
    eigvecs = res_full.eigenfunctions[:, idx]
    starts = list(range(0, a.size - window + 1, step))
    proj = []
    for s in starts:
        w = delay_embed(a[s : s + window], dim=dim)
        mean_delay = w.mean(axis=0)
        proj.append(np.abs(eigvecs.conj().T @ mean_delay.astype(complex)))
    p = np.asarray(proj)
    return np.linalg.norm(np.diff(p, axis=0), axis=1)


# ---------------------------------------------------------------------------
# Synthetic generators (seeded; correctness tests only)


def synth_linear_oscillator(
    n: int, omega: float = 0.35, decay: float = 0.0, seed: int = 0
) -> FloatArray:
    """x_t = A cos(omega t) exp(-decay t) + small noise — known spectrum."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.cos(omega * t) * np.exp(-decay * t) + 0.01 * rng.standard_normal(n)


def synth_logistic_map(n: int, r: float = 3.9, seed: int = 0) -> FloatArray:
    """Chaotic logistic map x_{t+1} = r x_t (1 - x_t)."""
    if not 3.5 < r < 4.0:
        raise ValueError("r in (3.5, 4.0) keeps the map bounded")
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = rng.uniform(0.2, 0.8)
    for i in range(1, n):
        x[i] = r * x[i - 1] * (1.0 - x[i - 1])
    return x


def synth_van_der_pol(n: int, mu: float = 1.5, dt: float = 0.05, seed: int = 0) -> FloatArray:
    """Van der Pol oscillator x'' - mu(1-x^2)x' + x = 0, returns x_t."""
    rng = np.random.default_rng(seed)
    x, v = rng.uniform(-1, 1), rng.uniform(-1, 1)
    out = np.empty(n)
    for i in range(n):
        a = mu * (1.0 - x * x) * v - x
        v += a * dt
        x += v * dt
        out[i] = x
    return out + 0.005 * rng.standard_normal(n)


def synth_regime_switching_ar(
    n: int, phi_lo: float = 0.1, phi_hi: float = 0.85, p: float = 0.04, seed: int = 0
) -> FloatArray:
    """Two-state Markov AR(1): phi_lo <-> phi_hi with switch prob p."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    phi = phi_lo
    for i in range(1, n):
        if rng.random() < p:
            phi = phi_hi if phi == phi_lo else phi_lo
        x[i] = phi * x[i - 1] + rng.standard_normal() * 0.5
    return x


def auc_from_scores(pos: FloatArray, neg: FloatArray) -> float:
    """Mann-Whitney AUC of pos > neg (drift-detector separability)."""
    p = np.asarray(pos, dtype=float).ravel()
    n = np.asarray(neg, dtype=float).ravel()
    if p.size == 0 or n.size == 0:
        raise ValueError("empty score array")
    wins = sum(np.sum(p_i > n) + 0.5 * np.sum(p_i == n) for p_i in p)
    return float(wins / (p.size * n.size))


def bench_koopman_edmd(seed: int = 20260131) -> dict[str, float]:
    """SYNTHETIC bench for the Koopman/EDMD machinery. Correctness only."""
    out: dict[str, float] = {}
    # 1) eigenvalue recovery on a damped oscillator: |lambda| = exp(-decay),
    #    arg(lambda) = omega * delay-sample. Compare modulus + angle.
    n, omega, decay = 2000, 0.35, 0.002
    x = synth_linear_oscillator(n, omega=omega, decay=decay, seed=seed)
    res = hankel_dmd(x, dim=12)
    mags = np.sort(np.abs(res.eigenvalues))[::-1]
    target_mag = np.exp(-decay)
    out["synthetic_eig_modulus_err"] = float(abs(mags[0] - target_mag))
    angs = np.sort(np.abs(np.angle(res.eigenvalues)))
    cand = angs[angs > 0.05]
    out["synthetic_eig_angle_err"] = float(abs(cand[0] - omega)) if cand.size else float("nan")
    out["synthetic_operator_residual"] = res.residual

    # 2) forecast R2 vs persistence on the chaotic logistic map
    x_log = synth_logistic_map(600, seed=seed + 1)
    z = delay_embed(x_log, dim=8)
    res_log = edmd_fit(z[:400])
    fc = koopman_forecast(res_log, z[399], steps=50)
    true = z[400:450, 0]
    pred = fc[:, 0]
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    out["synthetic_forecast_r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    pers = z[399, 0] * np.ones(50)
    ss_pers = float(np.sum((true - pers) ** 2))
    out["synthetic_forecast_beats_persistence"] = float(ss_res < ss_pers)

    # 3) EDMD == DMD on delay observables
    res_d = edmd_fit(z[:400], dictionary="delays")
    res_h = hankel_dmd(x_log[:408], dim=8)
    same_eval = np.linalg.norm(
        np.sort(np.abs(res_d.eigenvalues)) - np.sort(np.abs(res_h.eigenvalues))
    )
    out["synthetic_edmd_dmd_agreement"] = float(same_eval)

    # 4) drift detection AUC on regime-switching AR vs constant AR
    sw = synth_regime_switching_ar(4000, seed=seed + 2)
    base = synth_regime_switching_ar(4000, p=0.0, seed=seed + 3)
    d_sw = eigenfunction_drift(sw, window=400, step=200)
    d_base = eigenfunction_drift(base, window=400, step=200)
    # switch windows: top-quantile distances of the switching series vs
    # median distances of the constant series as negatives
    thr = float(np.quantile(d_sw, 0.75))
    pos = d_sw[d_sw >= thr]
    neg = d_base
    out["synthetic_drift_auc"] = auc_from_scores(pos, neg)
    out["synthetic_drift_switch_mean"] = float(d_sw.mean())
    out["synthetic_drift_base_mean"] = float(d_base.mean())

    # 5) determinism: two identical runs agree bit-for-bit on eigenvalues
    a1 = hankel_dmd(x, dim=12).eigenvalues
    a2 = hankel_dmd(x, dim=12).eigenvalues
    out["synthetic_determinism_delta"] = float(np.max(np.abs(a1 - a2)))
    return out
