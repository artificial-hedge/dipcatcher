"""QCQP -> SDP (Shor) relaxation bound for binary quadratic maximization.

max x'Qx, x in {-1,+1}^n. Shor: bound = min_d 1'd s.t. diag(d) >= Q
(psd order) — solved by subgradient on d with step direction
(n - top-eigenvector entries of diag(d)-Q). Bench compares the bound
against the brute-force +/-1 optimum on a random Q.
"""

import itertools

import numpy as np


def _bound(q: np.ndarray) -> float:
    """min sum_i d_i s.t. diag(d) - Q is PSD — penalized subgradient."""
    n = q.shape[0]
    lam_max = float(np.linalg.eigvalsh(q)[-1])
    d = np.full(n, lam_max)  # feasible start: lam_max*I - Q is PSD
    mu = 20.0 * n
    best = np.inf
    for it in range(800):
        m = np.diag(d) - q
        w, v = np.linalg.eigh(m)
        lam = float(w[0])
        feas = lam >= -1e-9
        if feas:
            best = min(best, d.sum())
            g = np.ones(n)
        else:
            g = np.ones(n) - mu * v[:, 0] ** 2
        d = np.maximum(d - (0.5 / np.sqrt(it + 1)) * g, 0.0)
        # restore feasibility if the step left the cone
        w2 = np.linalg.eigvalsh(np.diag(d) - q)[0]
        if w2 < 0:
            d = d - w2 * 1.0 + 1e-9
    return float(best)


def bench_qcqp_relax(seed: int = 5307) -> dict[str, float]:
    rng = np.random.default_rng(7)
    n = 8
    q = rng.normal(size=(n, n))
    q = q + q.T
    bound = _bound(q)
    truth = max(float(x @ q @ x) for x in itertools.product([-1.0, 1.0], repeat=n))
    # simpler SDP bound: n * lambda_max(Q)
    eig_bound = float(np.linalg.eigvalsh(q).max() * n)
    return {
        "synthetic_qcqp_sdp_bound": bound,
        "synthetic_qcqp_truth": truth,
        "synthetic_qcqp_eig_bound": eig_bound,
        "synthetic_qcqp_gap": bound - truth,
    }
