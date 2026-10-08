"""TimesFM-lite (Das et al. 2024) — patchify the context into patches of (SYNTHETIC)
p, decoder over patches predicts next patch; quantile head.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._tsf_synth import naive_quantiles, pinball, ts_series


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("timesfm_lite requires torch (pip install -e .[nn])") from exc
    return torch


def bench_timesfm_lite(
    seed: int = 739,
    n_train: int = 60,
    iters: int = 120,
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
    emb = torch.nn.Linear(p, 32)
    blk = torch.nn.TransformerEncoderLayer(32, 4, 64, batch_first=True)
    head = torch.nn.Linear(32, horizon * n_tau)
    params = list(emb.parameters()) + list(blk.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=0.005)
    for _i in range(iters):
        z = emb(x.reshape(len(x), ctx // p, p))
        z = blk(z)
        pred = head(z.mean(1)).reshape(-1, horizon, n_tau)
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
        z = emb(torch.tensor(hist[-ctx:]).float().reshape(1, ctx // p, p))
        z = blk(z)
        qs = head(z.mean(1)).reshape(horizon, n_tau).numpy()
    pb = pinball(y_true, qs, taus)
    pb_n = pinball(y_true, naive_quantiles(hist, taus, period=24), taus)
    return {
        "synthetic_tfm_pinball": pb,
        "synthetic_tfm_naive_pinball": pb_n,
        "synthetic_tfm_gain": pb_n - pb,
        "synthetic_torch_available": 1.0,
    }
