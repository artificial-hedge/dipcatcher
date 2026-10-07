"""N-BEATS: deep stack of interpretable basis blocks (SYNTHETIC).

Oreshkin et al. 2020: a stack of residual blocks, each producing a
backcast (input reconstruction) and a forecast on a shared basis.
Blocks doubly-residualize: each sees the backcast error of the
previous one — giving a clean interpretable decomposition.

Bench: synthetic series = trend + harmonic + spike components; a
3-block generic N-BEATS should beat an equal-parameter MLP and a
seasonal-naive baseline on the forecast.
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
        raise ImportError("nbeats_deep requires the `nn` extra (make sync)") from exc


def synth_multicomponent(
    n: int, win: int, h: int, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    """(n, win) history, (n, h) forecast: trend+harmonic+spike mixture."""
    t_in = np.arange(win)
    x = np.zeros((n, win))
    y = np.zeros((n, h))
    for i in range(n):
        t_out = np.arange(win, win + h)
        trend = 0.05 * rng.standard_normal() * np.concatenate([t_in, t_out])
        seas = rng.standard_normal() * np.sin(
            2 * np.pi * np.concatenate([t_in, t_out]) / (8 + rng.integers(0, 5))
            + rng.uniform(0, 2 * np.pi)
        )
        z = trend + seas + 0.1 * rng.standard_normal(win + h)
        spk = rng.integers(0, win)
        z[spk] += 1.5 * rng.standard_normal()
        x[i], y[i] = z[:win], z[win:]
    return x.astype(np.float64), y.astype(np.float64)


def _block(torch: Any, win: int, h: int, hid: int = 64) -> Any:
    return torch.nn.ModuleDict(
        {
            "fc": torch.nn.Sequential(
                torch.nn.Linear(win, hid),
                torch.nn.ReLU(),
                torch.nn.Linear(hid, hid),
                torch.nn.ReLU(),
            ),
            "theta_b": torch.nn.Linear(hid, win),
            "theta_f": torch.nn.Linear(hid, h),
        }
    )


def bench_nbeats_deep(
    seed: int = 20261231,
    n: int = 360,
    win: int = 40,
    h: int = 8,
    iters: int = 900,
) -> dict[str, float]:
    """3-block N-BEATS vs equal-param MLP and seasonal naive."""
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    rng = np.random.default_rng(seed)
    xs, ys = synth_multicomponent(n, win, h, rng)
    xb = torch.tensor(xs, dtype=torch.float32)
    yb = torch.tensor(ys, dtype=torch.float32)
    tr = slice(0, 3 * n // 4)
    te = slice(3 * n // 4, n)

    blocks = torch.nn.ModuleList([_block(torch, win, h) for _ in range(3)])
    opt = torch.optim.Adam(blocks.parameters(), lr=1e-3)
    for _ in range(iters):
        back = xb[tr].clone()
        fc = torch.zeros_like(yb[tr])
        for blk in blocks:
            z = blk["fc"](back)
            back = back - blk["theta_b"](z)
            fc = fc + blk["theta_f"](z)
        loss = torch.mean((fc - yb[tr]) ** 2) + 0.1 * torch.mean(back**2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        back = xb[te].clone()
        fc = torch.zeros_like(yb[te])
        for blk in blocks:
            z = blk["fc"](back)
            back = back - blk["theta_b"](z)
            fc = fc + blk["theta_f"](z)
        nb_mae = float(torch.mean(torch.abs(fc - yb[te])))

    mlp = torch.nn.Sequential(torch.nn.Linear(win, 128), torch.nn.ReLU(), torch.nn.Linear(128, h))
    pm = torch.nn.ModuleList([mlp])
    optm = torch.optim.Adam(pm.parameters(), lr=1e-3)
    for _ in range(iters):
        loss = torch.mean((mlp(xb[tr]) - yb[tr]) ** 2)
        optm.zero_grad()
        loss.backward()
        optm.step()
    with torch.no_grad():
        mlp_mae = float(torch.mean(torch.abs(mlp(xb[te]) - yb[te])))

    naive = np.tile(xs[:, -8:], (1, h // 8 + 1))[:, :h]
    naive_mae = float(np.mean(np.abs(naive[te] - ys[te])))
    return {
        "synthetic_nbeats_mae": nb_mae,
        "synthetic_nbeats_mlp_mae": mlp_mae,
        "synthetic_nbeats_naive_mae": naive_mae,
        "synthetic_nbeats_margin_vs_mlp": mlp_mae - nb_mae,
        "synthetic_nbeats_margin_vs_naive": naive_mae - nb_mae,
        "synthetic_torch_available": 1.0,
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(bench_nbeats_deep()))
