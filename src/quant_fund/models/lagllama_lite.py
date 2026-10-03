"""Lag-Llama-lite (Rasul et al. 2024) — lag-vector features → transformer
encoder → Student-T head; quantiles from the analytic t-inverse — vs
seasonal-naive pinball.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from quant_fund.models._tsf_synth import naive_quantiles, pinball, ts_series


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("lagllama_lite requires torch (pip install -e .[nn])") from exc
    return torch


def bench_lagllama_lite(
    seed: int = 747,
    n_train: int = 60,
    iters: int = 120,
    horizon: int = 24,
    n_tau: int = 9,
) -> dict[str, float]:
    torch = _torch()
    taus = np.linspace(0.1, 0.9, n_tau)
    lags = [1, 2, 3, 6, 12, 24]
    series = [ts_series(seed + i) for i in range(n_train)]
    xs, ys = [], []
    for s in series:
        for t0 in range(max(lags) + 1, len(s) - horizon, 12):
            xs.append([s[t0 - lag] for lag in lags])
            ys.append(s[t0 : t0 + horizon])
    x = torch.tensor(np.asarray(xs)).float()
    y = torch.tensor(np.asarray(ys)).float()
    torch.manual_seed(seed)
    emb = torch.nn.Linear(1, 24)
    blk = torch.nn.TransformerEncoderLayer(24, 4, 48, batch_first=True)
    head = torch.nn.Linear(24, 3)  # mu, log_sigma, log_df
    params = list(emb.parameters()) + list(blk.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=0.005)
    for _i in range(iters):
        z = blk(emb(x[:, :, None])).mean(1)
        mu, ls, ld = z[:, 0:1], z[:, 1:2], z[:, 2:3].clamp(-1, 3)
        sig = torch.exp(ls).clamp_min(1e-3)
        df = torch.exp(ld) + 2.1
        # point forecast: predict next-step mean; quantiles via t
        nll = -torch.distributions.StudentT(df, mu, sig).log_prob(y[:, 0:1]).mean()
        loss = nll
        opt.zero_grad()
        loss.backward()
        opt.step()
    s_te = ts_series(seed + 999)
    hist = s_te[: len(s_te) - horizon]
    y_true = s_te[-horizon:]
    with torch.no_grad():
        feat = torch.tensor([[hist[-lag] for lag in lags]]).float()
        z = blk(emb(feat[:, :, None])).mean(1)
        mu, ls, ld = z[:, 0].item(), z[:, 1].item(), z[:, 2].item()
        sig = np.exp(ls)
        df = np.exp(ld) + 2.1
        qs = np.stack([stats.t.ppf(taus, df, mu, sig)] * horizon)
    pb = pinball(y_true, qs, taus)
    pb_n = pinball(y_true, naive_quantiles(hist, taus, period=24), taus)
    return {
        "synthetic_ll_pinball": pb,
        "synthetic_ll_naive_pinball": pb_n,
        "synthetic_ll_gain": pb_n - pb,
        "torch_available": 1.0,
    }
