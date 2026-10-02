"""Moirai-lite (Woo et al. 2024) — any-variate masked-patch transformer:
series packed as (variate, patch) tokens with random masking; output head
per patch → quantile forecast.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._tsf_synth import naive_quantiles, pinball, ts_series


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("moirai_lite requires torch (pip install -e .[nn])") from exc
    return torch


def bench_moirai_lite(
    seed: int = 743,
    n_train: int = 60,
    iters: int = 120,
    p: int = 8,
    mask_p: float = 0.3,
    horizon: int = 24,
    n_tau: int = 9,
) -> dict[str, float]:
    torch = _torch()
    taus = np.linspace(0.1, 0.9, n_tau)
    ctx = p * 4
    series = [ts_series(seed + i) for i in range(n_train)]
    xs = []
    for s in series:
        for t0 in range(0, len(s) - ctx - horizon, 12):
            xs.append(s[t0 : t0 + ctx])
    x = torch.tensor(np.asarray(xs)).float()
    torch.manual_seed(seed)
    emb = torch.nn.Linear(p + 1, 32)  # +mask flag
    blk = torch.nn.TransformerEncoderLayer(32, 4, 64, batch_first=True)
    head = torch.nn.Linear(32, p * n_tau)
    params = list(emb.parameters()) + list(blk.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=0.005)
    rng = np.random.default_rng(seed)
    for _i in range(iters):
        xp = x.reshape(len(x), ctx // p, p)
        m = torch.tensor(rng.uniform(size=xp.shape[:2]) < mask_p).float()[:, :, None]
        tok = torch.cat([xp * (1 - m), m], -1)
        z = blk(emb(tok))
        pred = head(z).reshape(len(x), ctx // p, p, n_tau)
        # reconstruct masked patches → SSL objective on context
        diff = xp[:, :, :, None] - pred
        tt = torch.tensor(taus).float()[None, None, None]
        loss = (
            (torch.maximum(tt * diff, (tt - 1) * diff) * m[:, :, :, None]).sum()
            / m.sum().clamp_min(1)
            / n_tau
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    # forecast: append horizon/p zero-masked patches? predict via last patch head
    s_te = ts_series(seed + 999)
    hist = s_te[: len(s_te) - horizon]
    y_true = s_te[-horizon:]
    with torch.no_grad():
        xp = torch.tensor(hist[-ctx:]).float().reshape(1, ctx // p, p)
        tok = torch.cat([xp, torch.zeros(1, ctx // p, 1)], -1)
        z = blk(emb(tok))
        qs = head(z[:, -1]).reshape(p, n_tau).numpy()
        # tile seasonal forecast to horizon
        qs = np.tile(qs, (horizon // p, 1))[:horizon]
    pb = pinball(y_true, qs, taus)
    pb_n = pinball(y_true, naive_quantiles(hist, taus, period=24), taus)
    return {
        "synthetic_moirai_pinball": pb,
        "synthetic_moirai_naive_pinball": pb_n,
        "synthetic_moirai_gain": pb_n - pb,
        "torch_available": 1.0,
    }
