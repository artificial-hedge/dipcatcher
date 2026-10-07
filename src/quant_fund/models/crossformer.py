"""CrossFormer-TS: two-stage attention over time and channel (SYNTHETIC).

Zhang & Yan 2023 (Crossformer): segment the series into patches,
attend first *within* time segments per channel (TSW) then *across*
channels via a learned router — attention over both axes without
quadratic blow-up.

Bench: synthetic panel where channels couple at a *patch level*
(channel 1's patch-3 mean predicts channel 0's tail); cross-channel
attention should beat per-channel temporal attention.
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
    except ImportError as exc:  # pragma: no cover
        raise ImportError("crossformer requires the `nn` extra (make sync)") from exc


def synth_patch_coupled(
    n: int, win: int, n_ch: int, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    """(n, win, C) panel; target = f(patch-mean of channel 1 in patch 1)."""
    x = np.zeros((n, win, n_ch))
    y = np.zeros(n)
    seg = win // 4
    for i in range(n):
        base = np.cumsum(0.15 * rng.standard_normal((n_ch, win)), 1)
        driver = base[1, seg : 2 * seg].mean()
        x[i] = base.T
        y[i] = 0.8 * np.tanh(2.0 * driver) + 0.3 * base[0, -1] + 0.15 * rng.standard_normal()
    return x.astype(np.float64), y.astype(np.float64)


def bench_crossformer(
    seed: int = 20261231,
    n: int = 320,
    win: int = 48,
    n_ch: int = 4,
    iters: int = 900,
) -> dict[str, float]:
    """Patch-wise + cross-channel attention vs per-channel transformer."""
    torch = _torch()
    rng = np.random.default_rng(seed)
    xs, y = synth_patch_coupled(n, win, n_ch, rng)
    xb = torch.tensor(xs, dtype=torch.float32)
    yb = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    tr = slice(0, 3 * n // 4)
    te = slice(3 * n // 4, n)
    seg = win // 4

    d = 32
    patch_emb = torch.nn.Linear(seg, d)
    t_attn = torch.nn.MultiheadAttention(d, 4, batch_first=True)
    c_attn = torch.nn.MultiheadAttention(d, 4, batch_first=True)
    head = torch.nn.Linear(4 * d, 1)
    params = torch.nn.ModuleList([patch_emb, t_attn, c_attn, head])
    opt = torch.optim.Adam(params.parameters(), lr=1e-3)

    def forward(x_in: Any) -> Any:
        B = x_in.shape[0]
        tok = patch_emb(x_in.reshape(B, n_ch, 4, seg))  # (B, C, 4, d)
        t = tok.reshape(B * n_ch, 4, d)
        t, _ = t_attn(t, t, t)  # within-channel patch attention
        t = t.reshape(B, n_ch, 4, d).permute(0, 2, 1, 3).reshape(B * 4, n_ch, d)
        c, _ = c_attn(t, t, t)  # cross-channel attention per patch
        c = c.reshape(B, 4, n_ch, d).permute(0, 2, 1, 3).reshape(B, n_ch, 4 * d)
        return head(c[:, 0])  # channel 0: concat over patches

    for _ in range(iters):
        loss = torch.mean((forward(xb[tr]).squeeze(1) - yb[tr].squeeze(1)) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        cf_mae = float(torch.mean(torch.abs(forward(xb[te]).squeeze(1) - yb[te].squeeze(1))))

    # per-channel baseline: channel-0 patch attention only (no cross)
    p_attn = torch.nn.MultiheadAttention(d, 4, batch_first=True)
    p_emb = torch.nn.Linear(seg, d)
    p_head = torch.nn.Linear(4 * d, 1)
    pb = torch.nn.ModuleList([p_attn, p_emb, p_head])
    optp = torch.optim.Adam(pb.parameters(), lr=1e-3)
    for _ in range(iters):
        tok = p_emb(xb[tr, :, 0].reshape(-1, 4, seg))
        h, _ = p_attn(tok, tok, tok)
        loss = torch.mean((p_head(h.reshape(-1, 4 * d)).squeeze(1) - yb[tr].squeeze(1)) ** 2)
        optp.zero_grad()
        loss.backward()
        optp.step()
    with torch.no_grad():
        tok = p_emb(xb[te, :, 0].reshape(-1, 4, seg))
        h, _ = p_attn(tok, tok, tok)
        pc_mae = float(
            torch.mean(torch.abs(p_head(h.reshape(-1, 4 * d)).squeeze(1) - yb[te].squeeze(1)))
        )
    return {
        "synthetic_crossformer_mae": cf_mae,
        "synthetic_crossformer_perchan_mae": pc_mae,
        "synthetic_crossformer_margin_vs_perchan": pc_mae - cf_mae,
        "synthetic_torch_available": 1.0,
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(bench_crossformer()))
