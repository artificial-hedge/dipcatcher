"""Hypernetwork conditional allocator (torch).

A hypernetwork maps market state (vol level, cross-asset correlation
level) to the weights of a small target allocator network — so the
allocation function itself is regenerated per regime instead of one
static weighting. Requires the ``nn`` extra; SYNTHETIC regimes only.

Bench: 3-asset synthetic panel whose optimal tilt flips with a hidden
regime; hypernetwork-conditional allocator vs unconditional trained
allocator and an equal-weight baseline — mean per-step return of the
held tilt (a cost-free SYNTHETIC metric, not a P&L claim).
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
        raise ImportError(
            "hypernetwork_alloc torch path requires the `nn` extra (make sync)"
        ) from exc


def torch_available() -> bool:
    try:
        _torch()
        return True
    except ImportError:
        return False


def synth_market(n: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray, FloatArray]:
    """3 assets; hidden regime z ∈ {0,1} flips which asset leads.

    Returns: returns (n,3), observable state (n,2) = [vol proxy, corr proxy],
    regime labels.
    """
    rets = np.zeros((n, 3))
    state = np.zeros((n, 2))
    z = np.zeros(n, dtype=int)
    mus = {
        0: np.array([0.6, -0.2, 0.0]),
        1: np.array([-0.2, 0.6, 0.0]),
        2: np.array([-0.2, -0.2, 0.7]),
    }
    regime = 0
    for t in range(n):
        # t>=0.75n: HELD-OUT regime 2; 0.6n..0.75n stays regime 1 (in-dist test)
        if t >= 3 * n // 4:
            regime = 2
        elif t >= 3 * n // 5:
            regime = 1
        elif rng.random() < 0.03:
            regime = 1 - regime
        z[t] = regime
        mu = mus[regime]
        vol = 0.3 + 0.35 * regime
        corr = 0.2 + 0.25 * regime
        rets[t] = mu + vol * rng.standard_normal(3) @ np.linalg.cholesky(
            np.array([[1.0, corr, 0.1], [corr, 1.0, 0.2], [0.1, 0.2, 1.0]])
        )
        state[t] = [vol + 0.05 * rng.standard_normal(), corr]
    return rets, state, z


def _alloc_net(torch: Any) -> Any:
    return torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))


def bench_hypernetwork_alloc(seed: int = 71) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    torch = _torch()
    torch.manual_seed(seed)
    n = 4000
    rets, state, z = synth_market(n, rng)
    s_t = torch.tensor(state, dtype=torch.float32)
    r_t = torch.tensor(rets, dtype=torch.float32)
    # train on regimes 0,1 only — regime 2 is a held-out generalization test
    tr = 3 * n // 5
    train_mask = z[:tr] < 2

    # hypernet: state -> target-net weights (16*2 + 16 + 3*16 + 3 = 99 params)
    hyper = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.ReLU(), torch.nn.Linear(32, 99))
    opt = torch.optim.Adam(hyper.parameters(), lr=3e-3)

    def apply(xb, wvec):
        w1 = wvec[:, :32].reshape(-1, 2, 16)
        b1 = wvec[:, 32:48]
        w2 = wvec[:, 48:96].reshape(-1, 16, 3)
        b2 = wvec[:, 96:99]
        h = torch.relu(torch.einsum("bi,bij->bj", xb, w1) + b1)
        return torch.einsum("bi,bij->bj", h, w2) + b2

    for _ in range(600):
        idx = torch.tensor(train_mask.nonzero()[0][rng.integers(0, train_mask.sum(), 256)])
        wv = hyper(s_t[idx])
        w_alloc = torch.softmax(apply(s_t[idx], wv), dim=1)
        ret = (w_alloc * r_t[idx]).sum(1)
        loss = -ret.mean() + 0.1 * (w_alloc**2).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    # unconditional allocator trained on same objective
    flat = _alloc_net(torch)
    fopt = torch.optim.Adam(flat.parameters(), lr=3e-3)
    for _ in range(600):
        idx = torch.tensor(train_mask.nonzero()[0][rng.integers(0, train_mask.sum(), 256)])
        w_alloc = torch.softmax(flat(s_t[idx]), dim=1)
        ret = (w_alloc * r_t[idx]).sum(1)
        loss = -ret.mean() + 0.1 * (w_alloc**2).sum(1).mean()
        fopt.zero_grad()
        loss.backward()
        fopt.step()

    with torch.no_grad():
        hold = z[tr:] == 2
        ind = ~hold
        wv = hyper(s_t[tr:])
        w_hyper = torch.softmax(apply(s_t[tr:], wv), dim=1).numpy()
        w_flat = torch.softmax(flat(s_t[tr:]), dim=1).numpy()
    r_hyper = float(np.mean(np.sum(w_hyper[ind] * rets[tr:][ind], 1)))
    r_flat = float(np.mean(np.sum(w_flat[ind] * rets[tr:][ind], 1)))
    r_ew = float(np.mean(rets[tr:][ind]))
    h_hyper = float(np.mean(np.sum(w_hyper[hold] * rets[tr:][hold], 1)))
    h_flat = float(np.mean(np.sum(w_flat[hold] * rets[tr:][hold], 1)))
    h_ew = float(np.mean(rets[tr:][hold]))
    # oracle regime-conditional tilt
    return {
        "synthetic_hypernet_mean_ret": r_hyper,
        "synthetic_hypernet_flat_mean_ret": r_flat,
        "synthetic_hypernet_ew_mean_ret": r_ew,
        "synthetic_hypernet_margin_vs_flat": r_hyper - r_flat,
        "synthetic_hypernet_margin_vs_ew": r_hyper - r_ew,
        "synthetic_hypernet_holdout_ret": h_hyper,
        "synthetic_hypernet_holdout_flat_ret": h_flat,
        "synthetic_hypernet_holdout_ew_ret": h_ew,
        "synthetic_hypernet_holdout_margin_vs_flat": h_hyper - h_flat,
        "synthetic_torch_available": 1.0,
    }
