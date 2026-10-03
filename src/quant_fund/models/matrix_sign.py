"""Matrix-sign canon: Newton sign-function iteration
Z_{k+1} = (Z_k + Z_k^{-1})/2 converging to sign(Z) (±I on the
invariant subspaces) — with the Roberts/determinant scaling
Z ← |det Z|^{−1/n} Z — and the sign-based CARE solution
(sign(H)+I)/2 = [[X11, X12],[X21, X22]] with X = −X12⁻¹...
(via the stable deflating subspace trick used in SLICOT-style
solvers). All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def matrix_sign(Z: FloatArray, iters: int = 50, tol: float = 1e-12) -> FloatArray:
    """sign(Z) by scaled Newton: Z_{k+1} = (Z + Z^{-1})/2 with
    determinant scaling for speed."""
    Z = np.asarray(Z, dtype=np.float64)
    n = Z.shape[0]
    for _ in range(iters):
        # determinant scaling (Roberts): accelerates badly scaled Z
        det = abs(np.linalg.det(Z))
        c = det ** (-1.0 / n) if det > 0 else 1.0
        Zc = c * Z
        Zn = 0.5 * (Zc + np.linalg.inv(Zc))
        if np.linalg.norm(Zn - Zc, "fro") < tol * max(np.linalg.norm(Zn, "fro"), 1.0):
            Z = Zn
            break
        Z = Zn
    return np.asarray(Z, dtype=np.float64)


def sign_care(A: FloatArray, B: FloatArray, Q: FloatArray, R: FloatArray) -> FloatArray:
    """CARE via the matrix sign of the Hamiltonian
    H = [[A, −G],[−Q, −Aᵀ]], G = BR⁻¹Bᵀ.

    sign(H) = [[−I, −2X? ]] — the classic form: if W = sign(H)+I
    = [[W11, W12],[W21, W22]] then X = W21 W11⁻¹  /… we use the
    deflating formulation: solve for X from the stable subspace
    columns directly.
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    G = np.asarray(B) @ np.linalg.inv(np.asarray(R)) @ np.asarray(B).T
    H = np.block([[A, -G], [-np.asarray(Q), -A.T]])
    S = matrix_sign(H)
    # stable invariant subspace: eigenvalues with Re<0 get sign −1,
    # so (I − sign(H))/2 projects onto the stable subspace; an
    # orthonormal basis Z = [Z11; Z21] gives X = Z21 Z11⁻¹
    W = 0.5 * (np.eye(2 * n) - S)
    Qw, _ = np.linalg.qr(W)
    Z = Qw[:, :n]
    X = Z[n:, :] @ np.linalg.inv(Z[:n, :])
    return np.asarray(X, dtype=np.float64)


def bench_matrix_sign(seed: int = 20261231) -> dict[str, float]:
    out: dict[str, float] = {}
    # scalar: sign of a stable + an unstable eigenvalue
    D = np.diag([-2.0, 3.0, -0.5, 1.0])
    S = matrix_sign(D)
    out["synthetic_sign_diag_err"] = float(np.abs(S - np.diag([-1.0, 1.0, -1.0, 1.0])).max())
    # idempotence: sign(S)=S
    S2 = matrix_sign(S)
    out["synthetic_sign_idem_err"] = float(np.linalg.norm(S2 - S, "fro"))
    # rotated case: Z = V D V^{-1}
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((4, 4))
    while abs(np.linalg.det(V)) < 0.1:
        V = rng.standard_normal((4, 4))
    Z = V @ D @ np.linalg.inv(V)
    Sz = matrix_sign(Z)
    expected = V @ np.diag([-1.0, 1.0, -1.0, 1.0]) @ np.linalg.inv(V)
    out["synthetic_sign_rot_err"] = float(
        np.linalg.norm(Sz - expected, "fro") / np.linalg.norm(expected, "fro")
    )
    # sign-based CARE agrees with Hamiltonian path on a 2-d LQR
    from quant_fund.models.riccati_care import care_hamiltonian, care_residual

    A2 = np.array([[0.0, 1.0], [-2.0, 0.5]])
    B2 = np.array([[0.0], [1.0]])
    Q2 = np.eye(2)
    R2 = np.array([[0.5]])
    Xs = sign_care(A2, B2, Q2, R2)
    Xh = care_hamiltonian(A2, B2, Q2, R2)
    out["synthetic_sign_care_gap"] = float(
        np.linalg.norm(Xs - Xh, "fro") / np.linalg.norm(Xh, "fro")
    )
    out["synthetic_sign_care_residual"] = care_residual(Xs, A2, B2, Q2, R2)
    return out
