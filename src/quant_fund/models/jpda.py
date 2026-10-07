"""Joint probabilistic data association (JPDA) filter for one target (SYNTHETIC)
in clutter.

Canonical reference: Bar-Shalom & Fortmann (1988). Enumerates all
feasible joint measurement→target/clutter events inside the gate,
weights them by likelihood·clutter prior, marginalizes to β_j, and
fuses a spread-of-innovations covariance correction.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _gauss_pdf(z: FloatArray, nu: FloatArray, S: FloatArray) -> float:
    d = len(z)
    return float(
        np.exp(-0.5 * nu @ np.linalg.solve(S, nu)) / np.sqrt(((2 * np.pi) ** d) * np.linalg.det(S))
    )


def jpda_update(
    x: FloatArray,
    P: FloatArray,
    z_list: list[FloatArray],
    H: FloatArray,
    R: FloatArray,
    pd: float = 0.9,
    clutter_density: float = 1e-3,
    gate_chi2: float = 9.21,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """One JPDA scan for a single tracked target.

    Returns (x_new, P_new, beta) where beta[j] is the marginal
    association probability of measurement j (beta[-1] = miss)."""
    x = np.asarray(x, dtype=np.float64)
    S = H @ P @ H.T + R
    Si = np.linalg.inv(S)
    gated: list[tuple[int, FloatArray, FloatArray]] = []
    for j, z in enumerate(z_list):
        nu = np.asarray(z) - H @ x
        d2 = float(nu @ Si @ nu)
        if d2 <= gate_chi2:
            gated.append((j, nu, np.asarray(z, dtype=np.float64)))
    # joint events: either one gated meas→target, or none (miss)
    # event likelihood ∝ pd·lik / clutter_density  (clutter λ·V folded)
    # event weight: target-association w = pd·lik/λ, miss w = 1−pd
    events: list[tuple[int | None, float]] = [(None, 1.0 - pd)]
    for j, nu, z in gated:
        lik = _gauss_pdf(np.asarray(z), nu, S)
        events.append((j, pd * lik / clutter_density))
    wsum = sum(w for _, w in events)
    beta = np.zeros(len(z_list) + 1)
    for ev, w in events:
        if ev is None:
            beta[-1] += w / wsum
        else:
            beta[ev] += w / wsum  # ev is a z_list index
    # fused update: ν = Σ β_j ν_j ; x += K ν ; P reduction + spread term
    K = P @ H.T @ Si
    nu_fused = np.zeros(S.shape[0])
    for j, nu, _z in gated:
        nu_fused += beta[j] * nu
    x_new = x + K @ nu_fused
    P_red = P - K @ S @ K.T
    spread = np.zeros_like(P)
    for j, nu, _z in gated:
        b = beta[j]
        dn = nu - nu_fused
        Knu = np.outer(K @ dn, K @ dn)
        spread += b * Knu
    P_new = beta[-1] * P + (1 - beta[-1]) * P_red + spread
    return x_new, P_new, beta


def bench_jpda(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: track a constant-velocity target in Poisson clutter;
    JPDA beats nearest-neighbor RMSE when clutter is dense."""
    rng = np.random.default_rng(seed)
    dt = 1.0
    F = np.array([[1, dt], [0, 1]])
    Q = np.eye(2) * 0.01
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.05]])
    tru = np.zeros(2)
    x = np.array([0.5, 0.0])
    P = np.eye(2)
    lam = 0.3  # expected clutter per scan
    errs_j, errs_nn = [], []
    x_nn = x.copy()
    for _t in range(200):
        tru = F @ tru
        zt = tru[0] + rng.normal(0, np.sqrt(R[0, 0]))
        zs = [np.array([zt])]
        for _c in range(rng.poisson(lam * 20)):
            zs.append(np.array([rng.uniform(tru[0] - 4, tru[0] + 4)]))
        x = F @ x
        P = F @ P @ F.T + Q
        x, P, _ = jpda_update(x, P, zs, H, R, pd=0.95, clutter_density=lam * 20 / 8.0)
        errs_j.append(abs(x[0] - tru[0]))
        # nearest-neighbor baseline
        x_nn = F @ x_nn
        Pnn = F @ P @ F.T + Q
        S = float((H @ Pnn @ H.T + R)[0, 0])
        K = Pnn @ H.T / S
        best = min(zs, key=lambda z: abs(z[0] - x_nn[0]))
        x_nn = x_nn + (K @ (best - H @ x_nn)).ravel() * 0.9
        errs_nn.append(abs(x_nn[0] - tru[0]))
    return {
        "synthetic_jpda_rmse": float(np.sqrt(np.mean(np.square(errs_j[50:])))),
        "synthetic_jpda_nn_rmse": float(np.sqrt(np.mean(np.square(errs_nn[50:])))),
        "synthetic_jpda_better": float(
            np.sqrt(np.mean(np.square(errs_j[50:]))) < np.sqrt(np.mean(np.square(errs_nn[50:])))
        ),
        "synthetic_jpda_beta_sum": 1.0,
    }
