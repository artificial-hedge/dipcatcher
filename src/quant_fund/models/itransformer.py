"""iTransformer: inverted-dimension transformer forecaster.

Liu et al. 2024 (iTransformer): embed each *variate's whole history*
as a token (not each timestep). Self-attention then mixes variates —
attention maps capture cross-channel correlation, which is the right
object for multivariate forecasting; timestep-token attention is
out-of-distribution at long horizons.

Bench: synthetic panel where the target channel's future depends on a
*lead indicator* channel — variate-mixing attention should beat a
channel-independent MLP head and a per-channel AR model.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover - exercised in no-torch envs
        raise ImportError("itransformer requires the `nn` extra (make sync)") from exc


def synth_variate_panel(
    n: int, win: int, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    """(n, win, 4) panel: channel 0 future = f(lead channel 1 lags).

    Channel 1 leads channel 0 by `lead` steps; channels 2,3 are noise.
    A model that mixes variates can exploit the leader; per-channel
    models cannot see it.
    """
    lead = 6
    x = np.zeros((n, win, 4))
    y = np.zeros(n)
    for i in range(n):
        leader = np.cumsum(0.15 * rng.standard_normal(win + lead))
        own = 0.4 * leader[lead:] + 0.5 * np.cumsum(0.15 * rng.standard_normal(win))
        tgt = (
            0.7 * leader[-1]
            + 0.3 * own[-1]
            + 0.2 * np.sin(3 * own[-1])
            + 0.15 * rng.standard_normal()
        )
        x[i] = np.stack(
            [own, leader[:win], rng.standard_normal(win), rng.standard_normal(win)],
            1,
        )
        y[i] = tgt
    return x.astype(np.float64), y.astype(np.float64)


def bench_itransformer(
    seed: int = 20261231,
    n: int = 320,
    win: int = 48,
    steps: int = 900,
    iters: int = 1600,
) -> dict[str, float]:
    """Fit inverted-attention forecaster vs channel-mix MLP + AR baseline."""
    torch = _torch()
    rng = np.random.default_rng(seed)
    xs, y = synth_variate_panel(n, win, rng)
    xs = (xs - xs.mean()) / (xs.std() + 1e-9)
    y = (y - y.mean()) / (y.std() + 1e-9)
    xb = torch.tensor(xs, dtype=torch.float32)
    yb = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    tr = slice(0, 3 * n // 4)
    te = slice(3 * n // 4, n)

    # iTransformer: variate tokens -> Linear(win, d) -> MHSA over C tokens
    # -> Linear(d, 1) per variate, take target channel.
    d = 64
    emb = torch.nn.Linear(win, d)
    attn = torch.nn.MultiheadAttention(d, 4, batch_first=True)
    head = torch.nn.Linear(d, 1)
    params = torch.nn.ModuleList([emb, attn, head])
    opt = torch.optim.Adam(params.parameters(), lr=2e-3)
    for _ in range(iters):
        tok = emb(xb[tr].permute(0, 2, 1))  # (B, C, d)
        h, _ = attn(tok, tok, tok)
        pred = head(h)[:, 0]
        loss = torch.mean((pred.squeeze(1) - yb[tr].squeeze(1)) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        tok = emb(xb[te].permute(0, 2, 1))
        h, _ = attn(tok, tok, tok)
        it_mae = float(torch.mean(torch.abs(head(h)[:, 0].squeeze(1) - yb[te].squeeze(1))))

    # baseline: flat channel-mix MLP on full window
    flat = torch.nn.Sequential(
        torch.nn.Linear(win * 4, 96), torch.nn.ReLU(), torch.nn.Linear(96, 1)
    )
    p2 = torch.nn.ModuleList([flat])
    opt2 = torch.optim.Adam(p2.parameters(), lr=2e-3)
    x2 = xb.reshape(n, win * 4)
    for _ in range(iters):
        loss = torch.mean((flat(x2[tr]).squeeze(1) - yb[tr].squeeze(1)) ** 2)
        opt2.zero_grad()
        loss.backward()
        opt2.step()
    with torch.no_grad():
        mlp_mae = float(torch.mean(torch.abs(flat(x2[te]).squeeze(1) - yb[te].squeeze(1))))

    # per-channel MLP baseline: channel-0 window only (the PatchTST /
    # channel-independent paradigm the paper argues against)
    pc = torch.nn.Sequential(torch.nn.Linear(win, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    pp = torch.nn.ModuleList([pc])
    optp = torch.optim.Adam(pp.parameters(), lr=2e-3)
    x0 = xb[:, :, 0]
    for _ in range(iters):
        loss = torch.mean((pc(x0[tr]).squeeze(1) - yb[tr].squeeze(1)) ** 2)
        optp.zero_grad()
        loss.backward()
        optp.step()
    with torch.no_grad():
        pc_mae = float(torch.mean(torch.abs(pc(x0[te]).squeeze(1) - yb[te].squeeze(1))))
    # per-channel ridge baseline: full channel-0 window -> y (cannot
    # see the leader channel at all)
    X = np.concatenate([xs[tr, :, 0], np.ones((xs[tr].shape[0], 1))], 1)
    beta = np.asarray(np.linalg.solve(X.T @ X + 1e-2 * np.eye(win + 1), X.T @ y[tr]))
    ar_pred = np.concatenate([xs[te, :, 0], np.ones((xs[te].shape[0], 1))], 1) @ beta
    ar_mae = float(np.mean(np.abs(ar_pred - y[te])))
    return {
        "synthetic_itransformer_mae": it_mae,
        "synthetic_itransformer_flatmlp_mae": mlp_mae,
        "synthetic_itransformer_margin_vs_mlp": mlp_mae - it_mae,
        "synthetic_itransformer_perchan_mae": pc_mae,
        "synthetic_itransformer_margin_vs_perchan": pc_mae - it_mae,
        "synthetic_itransformer_ar_mae": ar_mae,
        "torch_available": 1.0,
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(bench_itransformer()))
