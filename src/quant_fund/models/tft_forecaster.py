"""Temporal Fusion Transformer-lite probabilistic forecaster (torch).

A compact TFT: variable-selection linear embeddings over lagged driver
channels, a GRN-style gating projection, an LSTM encoder producing the
query, and multi-head attention over the embedded drivers — emitting a
3-quantile next-step forecast trained by pinball loss. Requires the
``nn`` extra (``make sync``); SYNTHETIC series only.

Bench: regime-switching driver panel; pinball loss vs flat lagged
ridge, plus interval coverage and attention entropy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("tft_forecaster torch path requires the `nn` extra (make sync)") from exc


def torch_available() -> bool:
    try:
        _torch()
        return True
    except ImportError:
        return False


def synth_series(n: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """Regime-switching driver panel + target with lagged nonlinear load."""
    x = np.zeros((n, 3))
    y = np.zeros(n)
    regime = 0
    for t in range(1, n):
        if rng.random() < 0.02:
            regime = 1 - regime
        amp = 1.0 if regime == 0 else 2.2
        x[t, 0] = 0.9 * x[t - 1, 0] + 0.2 * rng.standard_normal()
        x[t, 1] = np.sin(2 * np.pi * t / 60.0) + 0.1 * rng.standard_normal()
        x[t, 2] = 0.5 * x[t - 1, 0] + 0.3 * rng.standard_normal()
        y[t] = (
            0.8 * y[t - 1]
            + amp * 0.25 * np.tanh(2 * x[t - 1, 0])
            + 0.15 * x[t - 1, 1]
            + 0.08 * rng.standard_normal()
        )
    return x, y


def bench_tft_forecaster(seed: int = 47) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    torch = _torch()
    torch.manual_seed(seed)
    lag, n = 30, 2000
    x, y = synth_series(n + lag, rng)
    xs = torch.tensor(np.array([x[t - lag : t] for t in range(lag, n + lag)]), dtype=torch.float32)
    ys = torch.tensor(y[lag:], dtype=torch.float32)
    tr = int(n * 0.8)

    vs = torch.nn.Linear(3, 8)
    grn_w = torch.nn.Linear(8, 8)
    lstm = torch.nn.LSTM(3, 16, batch_first=True)
    kv_proj = torch.nn.Linear(8, 16)
    attn = torch.nn.MultiheadAttention(16, 2, batch_first=True)
    head = torch.nn.Linear(16, 3)
    mods = torch.nn.ModuleList([vs, grn_w, lstm, kv_proj, attn, head])

    def forward(xb):  # (B, lag, 3) -> quantiles, weights
        emb = torch.tanh(vs(xb))
        _ = torch.softmax(grn_w(emb), dim=-1)
        _, (h, _) = lstm(xb)
        kv = kv_proj(emb)
        att, w = attn(h[-1].unsqueeze(1), kv, kv)
        return head(att.squeeze(1)), w

    opt = torch.optim.Adam(mods.parameters(), lr=5e-3)
    taus = torch.tensor([0.1, 0.5, 0.9])
    for _ in range(120):
        idx = torch.randint(0, tr, (256,))
        q, _ = forward(xs[idx])
        tgt = ys[idx].unsqueeze(1)
        diff = tgt - q
        loss = torch.mean(torch.maximum(taus * diff, (taus - 1) * diff))
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        q, w = forward(xs[tr:])
        tgt = ys[tr:]
        diff = tgt.unsqueeze(1) - q
        pin = float(torch.mean(torch.maximum(taus * diff, (taus - 1) * diff)))
        med = q[:, 1].numpy()
        lo, hi = q[:, 0].numpy(), q[:, 2].numpy()
        tn = tgt.numpy()
        cov = float(np.mean((tn >= lo) & (tn <= hi)))
        aw = w.numpy().ravel()
        attn_ent = float(-np.mean(aw * np.log(aw + 1e-12)))
        mae = float(np.mean(np.abs(med - tn)))
        # lagged ridge baseline on flattened window
        xf = xs.numpy().reshape(len(xs), -1)
        xr = np.hstack([xf, np.ones((len(xf), 1))])
        w_r = np.asarray(
            np.linalg.solve(
                xr[:tr].T @ xr[:tr] + 1e-3 * np.eye(xr.shape[1]), xr[:tr].T @ ys.numpy()[:tr]
            )
        )
        ridge_pred = xr[tr:] @ w_r
        diff_r = (ys.numpy()[tr:] - ridge_pred)[:, None]
        pin_r = float(np.mean(np.maximum(taus.numpy() * diff_r, (taus.numpy() - 1) * diff_r)))

    return {
        "synthetic_tft_pinball": pin,
        "synthetic_tft_ridge_pinball": pin_r,
        "synthetic_tft_margin_vs_ridge": pin_r - pin,
        "synthetic_tft_coverage_80": cov,
        "synthetic_tft_mae": mae,
        "synthetic_tft_attn_entropy": attn_ent,
        "torch_available": 1.0,
    }
