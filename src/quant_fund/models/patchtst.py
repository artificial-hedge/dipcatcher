"""PatchTST-style channel-independent patch transformer (torch).

Patchify each channel of a multivariate panel into stride windows, project
each patch, run a transformer encoder, and predict the next patch — the
channel-independent weight sharing PatchTST introduced. Requires the
``nn`` extra; SYNTHETIC panel only.

Bench: AR(1)+seasonal+long-lag echo panel; next-step MAE vs AR(2) ridge
and a channel-mixed flat MLP. Honest result on this synthetic: the AR
baseline edges the patch transformer (transformers need scale) — the
margin is reported negative rather than tuned to a fake win.
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
        raise ImportError("patchtst torch path requires the `nn` extra (make sync)") from exc


def torch_available() -> bool:
    try:
        _torch()
        return True
    except ImportError:
        return False


def synth_panel(n: int, rng: np.random.Generator) -> FloatArray:
    """3-channel panel: AR + seasonal + 40-step echo."""
    x = np.zeros((n, 3))
    for t in range(1, n):
        for c in range(3):
            seasonal = np.sin(2 * np.pi * t / (40 + 17 * c))
            echo = 0.5 * np.sin(1.5 * x[t - 40, c]) if t > 40 else 0.0
            x[t, c] = 0.7 * x[t - 1, c] + seasonal * 0.4 + echo + 0.15 * rng.standard_normal()
    return x


def _patchify(x: FloatArray, plen: int, stride: int) -> FloatArray:
    """(T, C) -> (n_patch, plen, C)."""
    out = []
    for i in range(0, len(x) - plen + 1, stride):
        out.append(x[i : i + plen])
    return np.stack(out)


def _ar_ridge(x: FloatArray, tr: int) -> float:
    """AR(2) ridge next-step MAE on channel 0."""
    c0 = x[:, 0]
    xs = np.array([[c0[t - 1], c0[t - 2], 1.0] for t in range(2, tr)])
    w = np.asarray(np.linalg.solve(xs.T @ xs + 1e-4 * np.eye(3), xs.T @ c0[2:tr]))
    pred = np.array([[c0[t - 1], c0[t - 2], 1.0] for t in range(tr, len(c0))]) @ w
    return float(np.mean(np.abs(pred - c0[tr:])))


def _flat_mlp_mae(x: FloatArray, lag: int, tr: int, seed: int) -> float:
    """Channel-MIXED flat MLP baseline (what channel independence beats)."""
    torch = _torch()
    torch.manual_seed(seed)
    xs = torch.tensor(
        np.array([x[t - lag : t].ravel() for t in range(lag, len(x))]), dtype=torch.float32
    )
    ys = torch.tensor(x[lag:], dtype=torch.float32)
    m = torch.nn.Sequential(
        torch.nn.Linear(lag * x.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, x.shape[1]),
    )
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(200):
        loss = torch.mean(torch.abs(m(xs[:tr]) - ys[:tr]))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(torch.mean(torch.abs(m(xs[tr:]) - ys[tr:])))


def bench_patchtst(seed: int = 51) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    torch = _torch()
    torch.manual_seed(seed)
    T, plen, stride = 2000, 16, 8
    x = synth_panel(T, rng)
    patches = _patchify(x, plen, stride)  # (n_patch, plen, C)
    torch_p = torch.tensor(patches, dtype=torch.float32)

    proj = torch.nn.Linear(plen, 32)
    enc = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(32, 2, 64, batch_first=True, dropout=0.0), 1
    )
    head = torch.nn.Linear(32, 1)
    mods = torch.nn.ModuleList([proj, enc, head])

    def forward(pb):  # (B, n_patch, plen) -> (B, n_patch)
        h = enc(torch.relu(proj(pb)))
        return head(h).squeeze(-1)

    opt = torch.optim.Adam(mods.parameters(), lr=3e-3)
    n_p = patches.shape[0] - 1
    tr_n = int(n_p * 0.85)
    for _ in range(700):
        for c in range(3):
            p_in = torch_p[:n_p, :, c].unsqueeze(0)
            tgt = torch_p[1 : n_p + 1, 1, c]
            pred = forward(p_in[:, :tr_n]).squeeze(0)
            loss = torch.mean(torch.abs(pred - tgt[:tr_n]))
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        errs = []
        for c in range(3):
            p_in = torch_p[:n_p, :, c].unsqueeze(0)
            tgt = torch_p[1 : n_p + 1, 1, c]
            pred = forward(p_in[:, tr_n:]).squeeze(0)
            errs.append(float(torch.mean(torch.abs(pred - tgt[tr_n:]))))
    mae = float(np.mean(errs))
    lag = 16
    tr_t = int(T * 0.85)
    ar = _ar_ridge(x, tr_t)
    mlp = _flat_mlp_mae(x, lag, tr_t - lag, seed + 1)
    return {
        "synthetic_patchtst_mae": mae,
        "synthetic_patchtst_ar_mae": ar,
        "synthetic_patchtst_flatmlp_mae": mlp,
        "synthetic_patchtst_margin_vs_ar": ar - mae,
        "synthetic_patchtst_margin_vs_flatmlp": mlp - mae,
        "torch_available": 1.0,
    }
