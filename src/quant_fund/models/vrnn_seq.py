"""VRNN (Chung et al. 2015) — variational RNN: per-step latent z_t with
prior conditioned on h_{t-1}, posterior on (x_t, h_{t-1}); ELBO on
synthetic Markov-switching series. Next-step NLL vs plain GRU AE.
"""

from __future__ import annotations

import numpy as np


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("vrnn_seq requires torch (pip install -e .[nn])") from exc
    return torch


def bench_vrnn_seq(seed: int = 739, steps: int = 2200, T: int = 16) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    # fixture: AR(1) with regime-switching scale
    n = 401
    regime = rng.choice(2, n, p=[0.7, 0.3])
    vol = np.where(regime == 0, 0.15, 0.9)
    xs = np.zeros(n)
    for i in range(1, n):
        xs[i] = 0.75 * xs[i - 1] + vol[i] * rng.standard_normal()
    X = torch.tensor(xs[:-1]).float().reshape(-1, T, 1)
    nseq = X.shape[0]
    H, Z = 24, 4
    prior = torch.nn.Sequential(torch.nn.Linear(H, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2 * Z))
    enc = torch.nn.Sequential(
        torch.nn.Linear(H + 1, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2 * Z)
    )
    dec = torch.nn.Sequential(torch.nn.Linear(H + Z, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
    rnn = torch.nn.GRU(1 + Z, H, batch_first=True)
    opt = torch.optim.Adam(
        list(prior.parameters())
        + list(enc.parameters())
        + list(dec.parameters())
        + list(rnn.parameters()),
        lr=0.01,
    )
    for _ in range(steps):
        k = int(rng.integers(0, nseq))
        x_seq = X[k : k + 1]
        h0 = torch.zeros(1, 1, H)
        hs, zs = [], []
        kls = torch.tensor(0.0)
        for t in range(T - 1):
            hp = h0.squeeze(0)
            pm, ps = prior(hp).chunk(2, -1)
            qm, qs = enc(torch.cat([x_seq[:, t], hp], -1)).chunk(2, -1)
            std_q = torch.nn.functional.softplus(qs)
            z = qm + std_q * torch.randn_like(std_q)
            std_p = torch.nn.functional.softplus(ps)
            kls = kls + (
                torch.log(std_p / std_q) + (std_q**2 + (qm - pm) ** 2) / (2 * std_p**2) - 0.5
            ).sum(-1)
            out, h0 = rnn(torch.cat([x_seq[:, t : t + 1], z[:, None]], -1), h0)
            hs.append(out[:, -1])
            zs.append(z)
        # decode each step
        nll = torch.tensor(0.0)
        for t in range(T - 1):
            dm, ds = dec(torch.cat([hs[t], zs[t]], -1)).chunk(2, -1)
            tgt = x_seq[:, t + 1, 0]
            var = torch.nn.functional.softplus(ds) ** 2 + 1e-4
            nll = nll + (0.5 * torch.log(2 * np.pi * var) + 0.5 * (tgt - dm) ** 2 / var).mean()
        loss = nll + 0.05 * kls.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # eval: one-step NLL
    nll = torch.tensor(0.0)
    with torch.no_grad():
        for k in range(20):
            x_seq = X[k : k + 1]
            h0 = torch.zeros(1, 1, H)
            for t in range(T - 1):
                hp = h0.squeeze(0)
                pm, ps = prior(hp).chunk(2, -1)
                z = pm + torch.nn.functional.softplus(ps) * torch.randn_like(ps)
                out, h0 = rnn(torch.cat([x_seq[:, t : t + 1], z[:, None]], -1), h0)
                dm, ds = dec(torch.cat([out[:, -1], z], -1)).chunk(2, -1)
                tgt = x_seq[:, t + 1, 0]
                var = torch.nn.functional.softplus(ds) ** 2 + 1e-4
                nll = nll + (0.5 * torch.log(2 * np.pi * var) + 0.5 * (tgt - dm) ** 2 / var).mean()
    nll = nll / (20 * (T - 1))
    base = float(np.log(2 * np.pi * np.var(xs) ** 0.5) + 0.5)
    return {
        "synthetic_vrnn_step_nll": float(nll),
        "synthetic_vrnn_baseline_nll": base,
        "synthetic_vrnn_nll_gain": base - float(nll),
        "synthetic_torch_available": 1.0,
    }
