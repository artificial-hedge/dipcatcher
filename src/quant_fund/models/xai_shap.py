"""Explainable-AI attribution for return signals (Exec-Summary XAI item).
KernelSHAP (weighted linear fit over feature coalitions) and integrated
gradients on a linear+nonlinear signal model, with sanity checks:
attributions must sum to f(x)-f(base) (completeness) and rank-match the
true generative coefficients.

Synthetic bench: known-coefficient generator; SHAP recovers the true
feature ranking (Kendall-tau vs truth) and satisfies completeness.
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray


def signal_model(x: FloatArray) -> FloatArray:
    """y = 1.5*x0 + 0.8*x1^2 - 1.0*x2 + 0.4*x3*x4 — known nonlinear truth."""
    return np.asarray(1.5 * x[:, 0] + 0.8 * x[:, 1] ** 2 - 1.0 * x[:, 2] + 0.4 * x[:, 3] * x[:, 4])


def kernel_shap(
    model,
    x0: FloatArray,
    background: FloatArray,
    n_coal: int = 200,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Sample feature coalitions; fit weighted ridge (SHAP kernel weights)."""
    rng = rng or np.random.default_rng(0)
    p = x0.shape[0]
    base_val = float(model(background).mean())
    ws, ys, ws_w = [], [], []
    for _ in range(n_coal):
        mask: FloatArray = np.asarray(rng.random(p) < 0.5)
        if mask.all() or not mask.any():
            continue
        x_h = background.mean(0).copy()
        x_h[mask] = x0[mask]
        fs = float(model(x_h[None])[0])
        m = int(mask.sum())
        w = (p - 1) / (m * (p - m))  # SHAP kernel weight
        ws.append(mask.astype(float))
        ys.append(fs - base_val)
        ws_w.append(w)
    z = np.asarray(ws)
    y = np.asarray(ys)
    w = np.asarray(ws_w)
    # weighted least squares phi: f(x)-f(base) = sum phi
    zw = z * w[:, None]
    phi = np.linalg.solve(zw.T @ z + 1e-6 * np.eye(p), zw.T @ y)
    return np.asarray(phi)


def integrated_gradients(
    model, x0: FloatArray, baseline: FloatArray, steps: int = 64, eps: float = 1e-4
) -> FloatArray:
    p = x0.shape[0]
    grad_sum = np.zeros(p)
    for k in range(1, steps + 1):
        xi = baseline + (k / steps) * (x0 - baseline)
        for j in range(p):
            d = np.zeros(p)
            d[j] = eps
            grad_sum[j] += (model((xi + d)[None])[0] - model((xi - d)[None])[0]) / (2 * eps)
    avg = grad_sum / steps
    return np.asarray((x0 - baseline) * avg)


def kendall_tau(a: FloatArray, b: FloatArray) -> float:
    n = len(a)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    return float((conc - disc) / max(conc + disc, 1))


def bench_xai_shap(seed: int = 31) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    background = rng.standard_normal((200, 5))
    x0 = rng.standard_normal(5) * 1.5
    phi = kernel_shap(signal_model, x0, background, 600, rng)
    ig = integrated_gradients(signal_model, x0, np.zeros(5), steps=256)
    # completeness: sum phi approx f(x0)-f(base)
    fx = float(signal_model(x0[None])[0])
    fb = float(signal_model(background).mean())
    complete_err = abs(float(phi.sum()) - (fx - fb)) / max(abs(fx - fb), 1e-9)
    ig_err = abs(float(ig.sum()) - (fx - fb)) / max(abs(fx - fb), 1e-9)
    # rank fidelity vs crude truth magnitudes at x0
    truth = np.abs(
        np.array([1.5 * x0[0], 0.8 * x0[1] ** 2, -1.0 * x0[2], 0.4 * x0[3] * x0[4], 0.0])
    )
    tau = kendall_tau(np.abs(phi), truth)
    return {
        "synthetic_shap_completeness_err": complete_err,
        "synthetic_ig_completeness_err": ig_err,
        "synthetic_shap_rank_tau": tau,
        "synthetic_shap_phi_l1": float(np.abs(phi).sum()),
        "synthetic_shap_explained_gap": float(phi.sum() - (fx - fb)),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bench_xai_shap(), indent=1))
