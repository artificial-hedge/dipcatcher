"""MambaTS: selective-scan state-space model for time series.

Gu & Dao 2024 (Mamba): a linear state-space recurrence whose
transition/ input/ output maps are *input-dependent* (selective
scan) — the model learns which inputs to remember vs forget at each
step, unlike a GRU whose gates mix state and input cheaply.

Bench: synthetic series where a rare "shock" input must be remembered
~20 steps (selective memory); Mamba-style selective scan should beat
a GRU on the recall task.
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
        raise ImportError("mambats requires the `nn` extra (make sync)") from exc


def synth_selective_memory(
    n: int, win: int, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    """(n, win, 2) [signal, shock-flag]; target = shock value at t+20.

    A shock (flag=1) injects a signed value at step t; the target is
    that value, recalled at the END of the window regardless of how
    early the shock arrived — a selective-memory task.
    """
    x = np.zeros((n, win, 2))
    y = np.zeros(n)
    for i in range(n):
        sig = np.cumsum(0.15 * rng.standard_normal(win))
        flag = np.zeros(win)
        pos = rng.integers(2, win - 25)
        val = 1.5 * rng.standard_normal()
        flag[pos] = 1.0
        sig[pos] += val
        x[i] = np.stack([sig, flag], 1)
        y[i] = val
    return x.astype(np.float64), y.astype(np.float64)


def _mamba_layer(torch: Any, d_in: int, d_h: int) -> Any:
    """Input-dependent (B, C, delta) selective SSM step, forward via scan."""
    mods = torch.nn.ModuleDict(
        {
            "in_proj": torch.nn.Linear(d_in, 3 * d_h),
            "x_proj": torch.nn.Linear(d_in, d_h),
        }
    )
    return mods, d_h


def bench_mambats(
    seed: int = 20261231,
    n: int = 320,
    win: int = 32,
    iters: int = 900,
) -> dict[str, float]:
    """Selective-scan recall of a 20-step-old shock vs GRU baseline."""
    torch = _torch()
    rng = np.random.default_rng(seed)
    xs, y = synth_selective_memory(n, win, rng)
    xb = torch.tensor(xs, dtype=torch.float32)
    yb = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    tr = slice(0, 3 * n // 4)
    te = slice(3 * n // 4, n)

    d_h = 32
    mods, _ = _mamba_layer(torch, 2, d_h)
    head = torch.nn.Linear(d_h, 1)
    params = torch.nn.ModuleList([mods, head])
    opt = torch.optim.Adam(params.parameters(), lr=2e-3)

    def ssm_forward(x_in: Any) -> Any:
        dt = torch.nn.functional.softplus(mods["in_proj"](x_in))
        B = dt.shape[0]
        h_state = torch.zeros(B, d_h)
        for t in range(x_in.shape[1]):
            gates = dt[:, t].reshape(B, 3, d_h)
            a = torch.sigmoid(gates[:, 0])  # state retention
            b = gates[:, 1]  # input scaling
            c = mods["x_proj"](x_in[:, t])
            h_state = a * h_state + b * c
        return h_state

    for _ in range(iters):
        loss = torch.mean((head(ssm_forward(xb[tr])).squeeze(1) - yb[tr].squeeze(1)) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        m_mae = float(
            torch.mean(torch.abs(head(ssm_forward(xb[te])).squeeze(1) - yb[te].squeeze(1)))
        )

    gru = torch.nn.GRU(2, d_h, batch_first=True)
    g_head = torch.nn.Linear(d_h, 1)
    pg = torch.nn.ModuleList([gru, g_head])
    optg = torch.optim.Adam(pg.parameters(), lr=2e-3)
    for _ in range(iters):
        _, hh = gru(xb[tr])
        loss = torch.mean((g_head(hh[-1]).squeeze(1) - yb[tr].squeeze(1)) ** 2)
        optg.zero_grad()
        loss.backward()
        optg.step()
    with torch.no_grad():
        _, hh = gru(xb[te])
        g_mae = float(torch.mean(torch.abs(g_head(hh[-1]).squeeze(1) - yb[te].squeeze(1))))

    flat = torch.nn.Sequential(
        torch.nn.Linear(win * 2, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)
    )
    pf = torch.nn.ModuleList([flat])
    optf = torch.optim.Adam(pf.parameters(), lr=1e-3)
    xf = xb.reshape(n, win * 2)
    for _ in range(iters):
        loss = torch.mean((flat(xf[tr]).squeeze(1) - yb[tr].squeeze(1)) ** 2)
        optf.zero_grad()
        loss.backward()
        optf.step()
    with torch.no_grad():
        f_mae = float(torch.mean(torch.abs(flat(xf[te]).squeeze(1) - yb[te].squeeze(1))))
    return {
        "synthetic_mamba_mae": m_mae,
        "synthetic_mamba_gru_mae": g_mae,
        "synthetic_mamba_flat_mae": f_mae,
        "synthetic_mamba_margin_vs_gru": g_mae - m_mae,
        "synthetic_mamba_margin_vs_flat": f_mae - m_mae,
        "synthetic_torch_available": 1.0,
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(bench_mambats()))
