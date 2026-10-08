"""Quantum kernel (ZZ-feature-map fidelity kernel) on 2-qubit XOR data — (SYNTHETIC)
kernel perceptron accuracy vs linear baseline. On this encoding the
fidelity Gram is flat — same-class vs cross-class overlap both ~0.53
— so the quantum kernel carries no class signal and loses to the
linear baseline; reported honestly, not gated on the win.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._qc_synth import I2, H, Z, apply1, init_state


def _feature_state(x: np.ndarray) -> np.ndarray:
    n = 2
    psi = init_state(n)
    for q in range(n):
        psi = apply1(psi, H, q, n)
    # ZZ entangling map
    zz = np.diag(np.kron(np.diag(Z), np.diag(Z)))
    phase = x[0] * x[1] * np.diag(zz)
    psi = np.asarray(np.exp(-1j * phase) * psi)
    for q in range(n):
        psi = apply1(psi, H, q, n)
        diagz = np.ones(1)
        for qq in range(n):
            diagz = np.kron(diagz, np.diag(Z) if qq == q else np.diag(I2))
        psi = np.asarray(np.exp(-1j * x[q] * diagz) * psi)
    return np.asarray(psi)


def bench_qkernel_svm(seed: int = 3081) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n_data = 40
    X = rng.uniform(-np.pi, np.pi, (n_data, 2))
    y = (np.sin(X[:, 0]) * np.sin(X[:, 1]) > 0).astype(float) * 2 - 1
    K = np.zeros((n_data, n_data))
    states = [_feature_state(x) for x in X]
    for i in range(n_data):
        for j in range(n_data):
            K[i, j] = abs(np.vdot(states[i], states[j])) ** 2
    # kernel perceptron
    alpha = np.zeros(n_data)
    for _ in range(30):
        for i in range(n_data):
            if y[i] * (K[:, i] @ (alpha * y)) <= 0:
                alpha[i] += 1
    pred = np.sign(K @ (alpha * y))
    acc_q = float((pred == y).mean())
    # linear baseline
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    acc_l = float((np.sign(X @ w) == y).mean())
    # measured: the fidelity Gram is flat for this label — same-class
    # overlap 0.526 vs cross-class 0.526, so acc_q wanders near chance
    # (0.5-0.6 across epochs) while the linear arm gets 0.7 — the win
    # claim is disproven and reported via synthetic_qk_gain; the gate
    # is a non-degeneracy floor on the kernel run itself.
    if acc_q < 0.5:
        raise ValueError("quantum kernel run degenerate")
    same = (y[:, None] == y[None, :]) & ~np.eye(n_data, dtype=bool)
    return {
        "synthetic_qk_acc": acc_q,
        "synthetic_linear_acc": acc_l,
        "synthetic_qk_gain": float(acc_q - acc_l),
        "synthetic_qk_overlap_same": float(K[same].mean()),
        "synthetic_qk_overlap_cross": float(K[~same].mean()),
        "synthetic_torch_available": 0.0,
    }
