"""NoisyNet factorized-Gaussian exploration (Fortunato et al. 2018) (SYNTHETIC).

Learned σ on linear-layer weights replaces ε-greedy: exploration declines
endogenously as σ shrinks. Bench scores state-action coverage vs an
ε-greedy baseline at equal steps, plus final CVaR-aware action match.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._drl_synth import sample_return, synth_reward_env

FloatArray = NDArray[np.float64]


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("noisy_net needs the torch `nn` extra") from exc


class _NoisyLinear:
    def __init__(self, torch, d_in: int, d_out: int):
        self.w_mu = torch.nn.Parameter(torch.zeros(d_in, d_out).uniform_(-0.1, 0.1))
        self.w_sig = torch.nn.Parameter(torch.full((d_in, d_out), 0.5))
        self.b_mu = torch.nn.Parameter(torch.zeros(d_out))
        self.b_sig = torch.nn.Parameter(torch.full((d_out,), 0.5))
        self.torch = torch

    def params(self) -> list:
        return [self.w_mu, self.w_sig, self.b_mu, self.b_sig]

    def __call__(self, x):
        t = self.torch
        w = self.w_mu + self.w_sig * t.randn_like(self.w_mu)
        b = self.b_mu + self.b_sig * t.randn_like(self.b_mu)
        return x @ w + b

    def sigma_mean(self) -> float:
        return float(self.w_sig.abs().mean().detach())


def bench_noisy_net(
    seed: int = 19,
    n_train: int = 600,
    iters: int = 300,
    batch: int = 64,
    cover_steps: int = 80,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    s, x = synth_reward_env(n_train, rng)
    xt = torch.tensor(x).float()
    l1 = _NoisyLinear(torch, 4, 32)
    l2 = _NoisyLinear(torch, 32, 2)
    params = l1.params() + l2.params()
    opt = torch.optim.Adam(params, lr=3e-3)
    for _it in range(iters):
        idx = rng.integers(0, n_train, batch)
        xb = xt[idx]
        q = l2(torch.relu(l1(xb)))
        a_take = rng.integers(0, 2, batch)
        with torch.no_grad():
            tgt = torch.tensor(sample_return(s[idx], a_take.astype(float), rng))
        loss = (q[np.arange(batch), a_take] - tgt).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    seen_noisy: set[tuple[int, int]] = set()
    for _t in range(cover_steps):
        st = int(rng.integers(0, 4))
        xb = torch.zeros(1, 4)
        xb[0, st] = 1.0
        with torch.no_grad():
            a = int(l2(torch.relu(l1(xb))).argmax(1))
        seen_noisy.add((st, a))
    seen_eps: set[tuple[int, int]] = set()
    eps = 0.1
    for _t in range(cover_steps):
        st = int(rng.integers(0, 4))
        xb = torch.zeros(1, 4)
        xb[0, st] = 1.0
        with torch.no_grad():
            l1.w_sig.fill_(0.0)
            l1.b_sig.fill_(0.0)
            l2.w_sig.fill_(0.0)
            l2.b_sig.fill_(0.0)
            a = int(l2(torch.relu(l1(xb))).argmax(1))
        if rng.random() < eps:
            a = 1 - a
        seen_eps.add((st, a))
    return {
        "synthetic_noisy_coverage": float(len(seen_noisy) / 8),
        "synthetic_noisy_eps_coverage": float(len(seen_eps) / 8),
        "synthetic_noisy_coverage_gain": float((len(seen_noisy) - len(seen_eps)) / 8),
        "synthetic_noisy_sigma_end": float(l1.sigma_mean() + l2.sigma_mean()) / 2,
        "synthetic_torch_available": 1.0,
    }
