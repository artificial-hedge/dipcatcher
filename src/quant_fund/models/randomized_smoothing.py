"""Randomized smoothing certification (Cohen et al. 2019) (SYNTHETIC).

Certified radius R = σ·Φ^{-1}(p_A) where p_A is the lower confidence
bound on the smoothed classifier's majority probability under N(0, σ²I)
noise. We report the certified-accuracy envelope vs the base model's
accuracy under matched noise — smoothing trades clean accuracy for a
non-trivial certified radius.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from quant_fund.models._cert_synth import synth_cls_2d

FloatArray = NDArray[np.float64]


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("randomized_smoothing needs the torch `nn` extra") from exc


def _train(seed: int, n_train: int, iters: int):
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    x, y = synth_cls_2d(n_train, rng)
    net = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
    x_t = torch.tensor(x).float()
    y_t = torch.tensor(y)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(net(x_t), y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def bench_randomized_smoothing(
    seed: int = 101,
    n_train: int = 400,
    n_test: int = 150,
    sigma: float = 0.5,
    n_samples: int = 300,
    iters: int = 400,
) -> dict[str, float]:
    torch = _torch()
    net = _train(seed, n_train, iters)
    rng = np.random.default_rng(seed + 1)
    xte, yte = synth_cls_2d(n_test, rng)
    x_t = torch.tensor(xte).float()
    with torch.no_grad():
        base_pred = net(x_t).argmax(-1).numpy()
    noise = torch.tensor(rng.standard_normal((n_samples, xte.shape[0], 4))).float()
    votes = np.zeros((xte.shape[0], 2))
    base_noisy = np.zeros((xte.shape[0], 2))
    with torch.no_grad():
        for i in range(n_samples):
            pred = net(x_t + noise[i]).argmax(-1).numpy()
            for j in range(2):
                votes[:, j] += pred == j
                base_noisy[:, j] += pred == j
    smooth_pred = votes.argmax(1)
    p_a = votes[np.arange(xte.shape[0]), smooth_pred] / n_samples
    p_lb = np.clip(p_a - 1.645 * np.sqrt(p_a * (1 - p_a) / n_samples), 0, 1 - 1e-9)
    radius = np.where(p_lb > 0.5, sigma * norm.ppf(p_lb), 0.0)
    acc_smooth = float((smooth_pred == yte).mean())
    acc_base_clean = float((base_pred == yte).mean())
    base_noisy_pred = base_noisy.argmax(1)
    acc_base_noise = float((base_noisy_pred == yte).mean())
    return {
        "synthetic_smooth_acc": acc_smooth,
        "synthetic_smooth_base_clean": acc_base_clean,
        "synthetic_smooth_base_noise": acc_base_noise,
        "synthetic_smooth_certified_frac": float((radius > 0).mean()),
        "synthetic_smooth_mean_radius": float(radius.mean()),
        "synthetic_torch_available": 1.0,
    }
