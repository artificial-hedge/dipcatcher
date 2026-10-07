"""Neural ODE factor dynamics (Exec-Summary continuous-time item). Latent (SYNTHETIC)
drift f_theta(z) learned from return windows; integrate with RK4 and train
end-to-end on next-step prediction. Compared with an AR baseline on a
synthetic nonlinear oscillator + trend factor process.
"""

from __future__ import annotations

from typing import Any

import numpy as np

FloatArray = np.ndarray


def _torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("neural_ode requires the `nn` extra (make sync)") from exc


def synth_process(T: int, rng: np.random.Generator) -> FloatArray:
    """2-D latent: damped oscillator (rotational drift) + small noise —
    returns observed through noisy projection."""
    z = np.zeros((T, 2))
    z[0] = [1.0, 0.0]
    for t in range(1, T):
        rot = np.array([[-0.1, 1.2], [-1.2, -0.1]])
        z[t] = z[t - 1] + 0.1 * (rot @ z[t - 1]) + 0.02 * rng.standard_normal(2)
    y = z[:, 0] + 0.05 * rng.standard_normal(T)
    return y


def bench_neural_ode(seed: int = 61) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    torch = _torch()
    torch.manual_seed(seed)
    T = 400
    y = synth_process(T, rng)
    yt = torch.tensor(y, dtype=torch.float32)
    tr = 300

    f = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.Tanh(), torch.nn.Linear(32, 2))
    opt = torch.optim.Adam(f.parameters(), lr=5e-3)
    dt = 0.1

    def rk4(z0, steps):
        z = z0
        for _ in range(steps):
            k1 = f(z)
            k2 = f(z + dt / 2 * k1)
            k3 = f(z + dt / 2 * k2)
            k4 = f(z + dt * k3)
            z = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return z

    # windows: encode last 2 obs as z0=(y_t, y_{t-1}); predict y_{t+k}
    for _ in range(600):
        i = int(rng.integers(2, tr - 20))
        z0 = torch.stack([yt[i], yt[i - 1]])
        pred = rk4(z0, 10)  # ~1 obs ahead in latent time
        loss = (pred[0] - yt[i + 1]) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
    # eval: teacher-forced 1-step + free 20-step rollout
    with torch.no_grad():
        one_step = [
            abs(float(rk4(torch.stack([yt[t], yt[t - 1]]), 10)[0]) - float(yt[t + 1]))
            for t in range(tr, tr + 40)
        ]
    z = torch.stack([yt[tr - 1], yt[tr - 2]])
    preds = []
    with torch.no_grad():
        for _ in range(20):
            z = rk4(z, 10)
            preds.append(float(z[0]))
    errs_ode = np.abs(np.array(preds[: T - tr]) - y[tr : tr + len(preds)])
    # AR baseline
    xs = np.array([[y[t - 1], y[t - 2], 1.0] for t in range(2, tr)])
    w = np.linalg.solve(xs.T @ xs + 1e-4 * np.eye(3), xs.T @ y[2:tr])
    x_ar = np.array([[y[t - 1], y[t - 2], 1.0] for t in range(tr, min(T, tr + 20))])
    errs_ar = np.abs(x_ar @ w - y[tr : tr + len(x_ar)])
    n = min(len(errs_ode), len(errs_ar))
    xs1 = np.array([[y[t], y[t - 1], 1.0] for t in range(2, tr)])
    w1 = np.linalg.solve(xs1.T @ xs1 + 1e-4 * np.eye(3), xs1.T @ y[3 : tr + 1])
    ar_1step = np.abs(
        np.array([[y[t], y[t - 1], 1.0] for t in range(tr, tr + 40)]) @ w1 - y[tr + 1 : tr + 41]
    )
    return {
        "synthetic_node_rollout_mae": float(np.mean(errs_ode[:n])),
        "synthetic_node_ar_mae": float(np.mean(errs_ar[:n])),
        "synthetic_node_margin": float(np.mean(errs_ar[:n]) - np.mean(errs_ode[:n])),
        "synthetic_node_1step_mae": float(np.mean(one_step)),
        "synthetic_node_1step_ar_mae": float(np.mean(ar_1step)),
        "synthetic_node_1step_margin": float(np.mean(ar_1step) - np.mean(one_step)),
        "synthetic_torch_available": 1.0,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bench_neural_ode(), indent=1))
