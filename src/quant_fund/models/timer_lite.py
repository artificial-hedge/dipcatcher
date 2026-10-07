"""Timer-lite (Liu et al. 2024) — generic TS backbone: next-token
prediction on continuous tokens via a causal transformer; the same
weights serve any series (genericity = the contribution).
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._tsf_synth import naive_quantiles, pinball, ts_series


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("timer_lite requires torch (pip install -e .[nn])") from exc
    return torch


def bench_timer_lite(
    seed: int = 751,
    n_train: int = 60,
    iters: int = 120,
    ctx: int = 24,
    horizon: int = 24,
    n_tau: int = 9,
) -> dict[str, float]:
    torch = _torch()
    taus = np.linspace(0.1, 0.9, n_tau)
    series = [ts_series(seed + i) for i in range(n_train)]
    xs, ys = [], []
    for s in series:
        for t0 in range(0, len(s) - ctx - horizon, 12):
            xs.append(s[t0 : t0 + ctx])
            ys.append(s[t0 + ctx : t0 + ctx + horizon])
    x = torch.tensor(np.asarray(xs)).float()
    y = torch.tensor(np.asarray(ys)).float()
    torch.manual_seed(seed)
    emb = torch.nn.Linear(1, 32)
    pos = torch.nn.Parameter(torch.randn(ctx, 32) * 0.02)
    blk = torch.nn.TransformerEncoderLayer(32, 4, 64, batch_first=True)
    head = torch.nn.Linear(32, 1 + n_tau)  # next-token + quantile offsets
    params = list(emb.parameters()) + [pos] + list(blk.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=0.005)
    mask = torch.triu(torch.ones(ctx, ctx) * float("-inf"), 1)
    for _i in range(iters):
        z = emb(x[:, :, None]) + pos[None]
        z = blk(z, src_mask=mask)
        out = head(z)
        # next-token supervision: predict x shifted by 1
        nxt = torch.cat([x[:, 1:], y[:, 0:1]], 1)
        mu = out[:, :, 0]
        offs = out[:, :, 1:]
        diff = nxt[:, :, None] - (mu[:, :, None] + offs)
        tt = torch.tensor(taus).float()[None, None]
        loss = torch.maximum(tt * diff, (tt - 1) * diff).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    s_te = ts_series(seed + 999)
    hist = s_te[: len(s_te) - horizon]
    y_true = s_te[-horizon:]
    # autoregressive quantile rollout
    cur = torch.tensor(hist[-ctx:]).float()[None, :, None]
    qs = []
    for _h in range(horizon):
        with torch.no_grad():
            z = emb(cur) + pos[: cur.shape[1]][None]
            z = blk(z, src_mask=mask[: cur.shape[1], : cur.shape[1]])
            out = head(z[:, -1:])
        mu = out[0, 0, 0].item()
        offs = out[0, 0, 1:].numpy()
        qs.append(mu + offs)
        cur = torch.cat([cur[:, 1:], torch.tensor([[[mu]]]).float()], 1)
    pb = pinball(y_true, np.asarray(qs), taus)
    pb_n = pinball(y_true, naive_quantiles(hist, taus, period=24), taus)
    return {
        "synthetic_timer_pinball": pb,
        "synthetic_timer_naive_pinball": pb_n,
        "synthetic_timer_gain": pb_n - pb,
        "synthetic_torch_available": 1.0,
    }
