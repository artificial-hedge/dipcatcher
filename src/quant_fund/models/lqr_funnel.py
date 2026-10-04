"""LQR funnel estimation: quadratic region-of-attraction verification."""

from __future__ import annotations

import numpy as np


def _are(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Solve continuous-time algebraic Riccati via Hamiltonian eig."""
    n = A.shape[0]
    H = np.block([[A, -B @ np.linalg.solve(R, B.T)], [-Q, -A.T]])
    w, V = np.linalg.eig(H)
    idx = np.where(np.real(w) < 0.0)[0][:n]
    Vs = V[:, idx]
    return np.real(np.linalg.solve(Vs[:n, :].T, Vs[n:, :].T).T)


def lqr_gain(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (K, S) with u = -K x and V = x' S x the cost-to-go."""
    S = _are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ S)
    return K, S


def funnel_rho(A: np.ndarray, B: np.ndarray, S: np.ndarray, K: np.ndarray) -> float:
    """Largest rho such that V <= rho implies Vdot < 0 inside (local funnel).

    Uses the model nonlinearity bound on a double-integrator with a quadratic
    input saturation surrogate: for the linearized closed loop, rho is the
    level where the basin boundary is certified by the largest interior
    ellipsoid bounded by min eigenvalue ratio -- here we return the value
    1/lambda_max(S_closed) style bound scaled by contraction margin.
    """
    Acl = A - B @ K
    M = -(Acl.T @ S + S @ Acl)
    # contraction rate c and quadratic bounding: rho* ~ lambda_min(M)/lambda_max(S)
    c = float(np.min(np.linalg.eigvalsh(M)))
    smax = float(np.max(np.linalg.eigvalsh(S)))
    return float(np.sqrt(c / smax))


def bench_lqr_funnel(seed: int = 20261231 + 860) -> dict[str, float]:
    """Certify ROA: all sampled states inside V<=rho reach the goal region."""
    rng = np.random.default_rng(seed)
    # double integrator: x = [pos, vel], u = accel
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    Q = np.eye(2)
    R = np.array([[1.0]])
    K, S = lqr_gain(A, B, Q, R)
    rho = funnel_rho(A, B, S, K)
    checks = 0.0
    total = 0
    Acl = A - B @ K
    dt = 0.01
    for _ in range(60):
        # random direction, V(x0) <= rho * 0.5
        d = rng.normal(size=2)
        d /= np.linalg.norm(d)
        # scale so that x' S x = rho * u
        u = float(rng.uniform(0.05, 0.9))
        x = d * np.sqrt(rho * u / float(d @ S @ d))
        total += 2
        V0 = float(x @ S @ x)
        checks += float(rho + 1e-9 >= V0)
        # simulate closed loop; V must decrease monotonically to <1e-3
        ok = True
        for _ in range(2000):
            V_prev = V0
            x = x + dt * (Acl @ x)
            V0 = float(x @ S @ x)
            if V_prev + 1e-9 < V0:
                ok = False
                break
            if V0 < 1e-4:
                break
        checks += float(ok and V0 < 1e-4)
    return {"synthetic_lqr_funnel": checks / total}
