"""Contrastive representation learning for return windows (torch).

SimCLR-lite: two augmented views of each return window (jitter + scale)
are encoded by a small MLP and pulled together with an NT-Xent loss over
in-batch negatives. The learned embedding is evaluated by a ridge linear
probe predicting the next step — versus the same probe on the raw window.
Requires the ``nn`` extra; SYNTHETIC regime data only.

Bench: windows sampled from 4 latent regimes with distinct dynamics —
contrastive embeddings cluster regime, so the probe should win.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError(
            "contrastive_repr torch path requires the `nn` extra (make sync)"
        ) from exc


def torch_available() -> bool:
    try:
        _torch()
        return True
    except ImportError:
        return False


def synth_regime_windows(
    n: int, win: int, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Windows from 4 regimes: trend up/down, mean-revert, vol-cluster.

    Label = next-step value (regime-dependent mapping of the window).
    """
    x = np.zeros((n, win))
    y = np.zeros(n)
    reg = np.zeros(n, dtype=int)
    for i in range(n):
        r = int(rng.integers(0, 4))
        reg[i] = r
        eps = rng.standard_normal(win + 1)
        if r == 0:
            s = np.cumsum(0.3 + 0.4 * eps)
        elif r == 1:
            s = np.cumsum(-0.3 + 0.4 * eps)
        elif r == 2:
            s = np.zeros(win + 1)
            for t in range(1, win + 1):
                s[t] = -0.6 * s[t - 1] + 0.5 * eps[t]
        else:
            vol = 0.2 + 1.5 * abs(np.sin(2 * np.pi * np.arange(win + 1) / win))
            s = np.cumsum(vol * eps)
        x[i] = s[:win] - s[0]
        # label: regime-dependent next-step drift — nonlinear latent the
        # embedding must separate for the probe to predict
        y[i] = [0.5, -0.5, 0.0, 0.1][r] + 0.05 * eps[win]
    return x, y, reg


def _ridge(x: FloatArray, y: FloatArray, tr: int, lam: float = 1e-2) -> float:
    xa = np.hstack([x, np.ones((len(x), 1))])
    w = np.asarray(
        np.linalg.solve(xa[:tr].T @ xa[:tr] + lam * np.eye(xa.shape[1]), xa[:tr].T @ y[:tr])
    )
    pred = xa[tr:] @ w
    return float(np.mean(np.abs(pred - y[tr:])))


def _class_acc(x: FloatArray, reg: FloatArray, tr: int, lam: float = 1e-2) -> float:
    """Multiclass ridge probe: one-hot targets -> argmax accuracy."""
    k = int(reg.max()) + 1
    t = np.eye(k)[reg.astype(int)]
    xa = np.hstack([x, np.ones((len(x), 1))])
    w = np.asarray(
        np.linalg.solve(xa[:tr].T @ xa[:tr] + lam * np.eye(xa.shape[1]), xa[:tr].T @ t[:tr])
    )
    pred = np.argmax(xa[tr:] @ w, 1)
    return float(np.mean(pred == reg[tr:]))


def bench_contrastive_repr(seed: int = 67) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    torch = _torch()
    torch.manual_seed(seed)
    n, win, tr = 4000, 32, 3200
    x, y, reg = synth_regime_windows(n, win, rng)
    xs = torch.tensor(x, dtype=torch.float32)

    enc = torch.nn.Sequential(
        torch.nn.Linear(win, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 16),
    )
    proj = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 8))
    mods = torch.nn.ModuleList([enc, proj])
    opt = torch.optim.Adam(mods.parameters(), lr=3e-3)
    tau = 0.2

    def augment(xb):
        jitter = xb + 0.1 * torch.randn_like(xb)
        scale = xb * (1.0 + 0.2 * torch.randn(len(xb), 1))
        return jitter, scale

    for _ in range(300):
        idx = torch.randint(0, tr, (256,))
        a, b = augment(xs[idx])
        za = torch.nn.functional.normalize(proj(enc(a)), dim=1)
        zb = torch.nn.functional.normalize(proj(enc(b)), dim=1)
        sim = za @ zb.T / tau
        loss = -torch.mean(torch.diag(sim) - torch.logsumexp(sim, dim=1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        emb = enc(xs).numpy()
    mae_emb = _ridge(emb, y, tr)
    mae_raw = _ridge(x, y, tr)
    acc_emb = _class_acc(emb, reg, tr)
    acc_raw = _class_acc(x, reg, tr)
    return {
        "synthetic_contrastive_probe_mae": mae_emb,
        "synthetic_contrastive_raw_probe_mae": mae_raw,
        "synthetic_contrastive_margin": mae_raw - mae_emb,
        "synthetic_contrastive_probe_acc": acc_emb,
        "synthetic_contrastive_raw_probe_acc": acc_raw,
        "synthetic_contrastive_acc_margin": acc_emb - acc_raw,
        "torch_available": 1.0,
    }
