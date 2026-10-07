"""MOMENT-lite (Goswami et al. 2024) — masked-TS pretraining: random (SYNTHETIC)
patch masking + reconstruction, then linear-probe the encoder for
quantile forecast — vs seasonal-naive.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._tsf_synth import naive_quantiles, pinball, ts_series


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("moment_lite requires torch (pip install -e .[nn])") from exc
    return torch


def bench_moment_lite(
    seed: int = 757,
    n_train: int = 60,
    iters: int = 100,
    p: int = 8,
    horizon: int = 24,
    n_tau: int = 9,
) -> dict[str, float]:
    torch = _torch()
    taus = np.linspace(0.1, 0.9, n_tau)
    ctx = p * 3
    series = [ts_series(seed + i) for i in range(n_train)]
    xs, ys = [], []
    for s in series:
        for t0 in range(0, len(s) - ctx - horizon, 12):
            xs.append(s[t0 : t0 + ctx])
            ys.append(s[t0 + ctx : t0 + ctx + horizon])
    x = torch.tensor(np.asarray(xs)).float()
    y = torch.tensor(np.asarray(ys)).float()
    torch.manual_seed(seed)
    emb = torch.nn.Linear(p + 1, 32)
    blk = torch.nn.TransformerEncoderLayer(32, 4, 64, batch_first=True)
    recon = torch.nn.Linear(32, p)
    rng = np.random.default_rng(seed)
    # masked pretrain
    params = list(emb.parameters()) + list(blk.parameters()) + list(recon.parameters())
    opt = torch.optim.Adam(params, lr=0.005)
    for _i in range(iters):
        xp = x.reshape(len(x), ctx // p, p)
        m = torch.tensor(rng.uniform(size=xp.shape[:2]) < 0.4).float()[:, :, None]
        tok = torch.cat([xp * (1 - m), m], -1)
        z = blk(emb(tok))
        rec = recon(z)
        loss = (((rec - xp) ** 2) * m).sum() / m.sum().clamp_min(1)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # linear probe on frozen encoder
    with torch.no_grad():
        xp = x.reshape(len(x), ctx // p, p)
        z_tr = blk(emb(torch.cat([xp, torch.zeros(len(x), ctx // p, 1)], -1))).mean(1)
    probe = torch.nn.Linear(32, horizon * n_tau)
    opt = torch.optim.Adam(probe.parameters(), lr=0.01)
    for _i in range(200):
        pred = probe(z_tr).reshape(-1, horizon, n_tau)
        diff = y[:, :, None] - pred
        tt = torch.tensor(taus).float()[None, None]
        loss = torch.maximum(tt * diff, (tt - 1) * diff).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    s_te = ts_series(seed + 999)
    hist = s_te[: len(s_te) - horizon]
    y_true = s_te[-horizon:]
    with torch.no_grad():
        xp = torch.tensor(hist[-ctx:]).float().reshape(1, ctx // p, p)
        z = blk(emb(torch.cat([xp, torch.zeros(1, ctx // p, 1)], -1))).mean(1)
        qs = probe(z).reshape(horizon, n_tau).numpy()
    pb = pinball(y_true, qs, taus)
    pb_n = pinball(y_true, naive_quantiles(hist, taus, period=24), taus)
    return {
        "synthetic_moment_pinball": pb,
        "synthetic_moment_naive_pinball": pb_n,
        "synthetic_moment_gain": pb_n - pb,
        "synthetic_torch_available": 1.0,
    }
