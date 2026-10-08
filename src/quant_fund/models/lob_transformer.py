"""DeepLOB-lite: conv + transformer mid-price move classifier (torch).

Conv1d feature extractor over limit-order-book windows (price/size at
two levels + imbalance features), adaptive pooling to a fixed token
count, a transformer encoder, and a pooled head predicting the sign of
the next mid move. Requires the ``nn`` extra; SYNTHETIC book only.

Bench: synthetic book where the label mixes instantaneous imbalance,
deep imbalance, price-slope curvature, and their interaction — the conv
+ attention stack should beat a logistic on raw features.
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
        raise ImportError("lob_transformer torch path requires the `nn` extra (make sync)") from exc


def torch_available() -> bool:
    try:
        _torch()
        return True
    except ImportError:
        return False


def synth_lob(n: int, win: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """Synthetic LOB windows (n, win, 4): bid_sz, ask_sz, bid_px_dev, ask_px_dev.

    Label = sign of next-window mid move driven by instantaneous + deep
    imbalance, local slope curvature, and an interaction term.
    """
    x = np.zeros((n, win, 4))
    y = np.zeros(n)
    state = rng.standard_normal(2)
    for i in range(n):
        state = 0.9 * state + 0.3 * rng.standard_normal(2)
        imb_now = np.tanh(state[0])
        imb_deep = np.tanh(0.7 * state[0] + 0.5 * state[1])
        slope = 0.5 * state[1]
        for t in range(win):
            decay = np.exp(-0.1 * t)
            b_sz = 20 + 30 * max(0.0, imb_now * decay + 0.1 * rng.standard_normal())
            a_sz = 20 + 30 * max(0.0, -imb_now * decay + 0.1 * rng.standard_normal())
            b_px = -(1 + slope * t / win) + 0.05 * rng.standard_normal()
            a_px = 1 + slope * t / win + 0.05 * rng.standard_normal()
            deep_b = imb_deep * np.exp(-0.3 * t)
            x[i, t] = [b_sz + 10 * max(0.0, deep_b), a_sz + 10 * max(0.0, -deep_b), b_px, a_px]
        sig = (
            0.5 * imb_now
            + 0.7 * imb_deep
            + 0.9 * np.sign(slope) * abs(slope) ** 0.3
            + 0.8 * imb_now * imb_deep
        )
        y[i] = float(sig + 0.3 * rng.standard_normal() > 0)
    return x, y


def _logistic_acc(x: FloatArray, y: FloatArray, tr: int, rng: np.random.Generator) -> float:
    feats = np.stack(
        [
            (x[:, 0, 0] - x[:, 0, 1]) / (x[:, 0, 0] + x[:, 0, 1] + 1e-9),
            (x[:, 10:, 0].mean(1) - x[:, 10:, 1].mean(1))
            / (x[:, 10:, 0].mean(1) + x[:, 10:, 1].mean(1) + 1e-9),
            x[:, -1, 3] + x[:, -1, 2],
            x[:, :, 0].mean(1) - x[:, :, 1].mean(1),
        ],
        1,
    )
    feats = np.hstack([feats, np.ones((len(feats), 1))])
    lam = 1e-3 * np.eye(feats.shape[1])
    w = np.asarray(np.linalg.solve(feats[:tr].T @ feats[:tr] + lam, feats[:tr].T @ y[:tr]))
    pred = feats[tr:] @ w > 0.5
    return float(np.mean(pred == y[tr:]))


def bench_lob_transformer(seed: int = 55) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    torch = _torch()
    torch.manual_seed(seed)
    n, win = 3000, 40
    x, y = synth_lob(n, win, rng)
    tr = int(n * 0.8)
    xs = torch.tensor(x, dtype=torch.float32)
    xs = (xs - xs[:tr].mean((0, 1), keepdim=True)) / (xs[:tr].std((0, 1), keepdim=True) + 1e-6)
    ys = torch.tensor(y, dtype=torch.float32)

    conv = torch.nn.Conv1d(4, 24, 3, padding=1)
    pool = torch.nn.AdaptiveAvgPool1d(12)
    enc = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(24, 2, 48, batch_first=True, dropout=0.0), 1
    )
    head = torch.nn.Linear(24 * 12, 1)
    mods = torch.nn.ModuleList([conv, enc, head])

    def forward(xb):  # (B, win, 4) -> logit
        h = torch.relu(conv(xb.transpose(1, 2)))
        h = pool(h).transpose(1, 2)
        return head(enc(h).reshape(len(xb), -1)).squeeze(-1)

    opt = torch.optim.Adam(mods.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(250):
        idx = torch.randint(0, tr, (256,))
        loss = bce(forward(xs[idx]), ys[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = forward(xs[tr:]) > 0
        acc = float((pred == ys[tr:].bool()).float().mean())
    log_acc = _logistic_acc(x, y, tr, rng)
    return {
        "synthetic_lobdl_acc": acc,
        "synthetic_lobdl_logistic_acc": log_acc,
        "synthetic_lobdl_margin": acc - log_acc,
        "synthetic_torch_available": 1.0,
    }
