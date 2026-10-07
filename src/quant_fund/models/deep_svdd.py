"""Deep SVDD (Ruff et al. 2018) (SYNTHETIC).

Minimize mean squared distance of normal-window embeddings to a center c;
score = distance to c on test windows — vs reconstruction-AE and chance.
"""

from __future__ import annotations

from quant_fund.models._anom_synth import anom_series, auc, windows


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("deep_svdd requires torch (pip install -e .[nn])") from exc
    return torch


def bench_deep_svdd(
    seed: int = 617,
    w: int = 16,
    iters: int = 120,
    h: int = 16,
) -> dict[str, float]:
    torch = _torch()
    x, y = anom_series(seed=seed)
    wins, ctr = windows(x, w)
    yw = y[ctr]
    normal = wins[yw == 0]
    n_tr = int(0.7 * len(normal))
    tr = normal[:n_tr]
    torch.manual_seed(seed)
    enc = torch.nn.Sequential(torch.nn.Linear(w * 3, h), torch.nn.ReLU(), torch.nn.Linear(h, 8))
    tr_f = torch.tensor(tr.reshape(len(tr), -1)).float()
    with torch.no_grad():
        c = enc(tr_f).mean(0)
    opt = torch.optim.Adam(enc.parameters(), lr=0.005)
    for _i in range(iters):
        loss = ((enc(tr_f) - c) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        all_f = torch.tensor(wins.reshape(len(wins), -1)).float()
        sc = ((enc(all_f) - c) ** 2).sum(-1).numpy()
    auc_svdd = auc(sc, yw)
    # reconstruction-AE baseline at matched size
    torch.manual_seed(seed)
    ae = torch.nn.Sequential(torch.nn.Linear(w * 3, h), torch.nn.ReLU(), torch.nn.Linear(h, w * 3))
    opt = torch.optim.Adam(ae.parameters(), lr=0.005)
    for _i in range(iters):
        loss = ((ae(tr_f) - tr_f) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        sc_ae = ((ae(all_f) - all_f) ** 2).mean(-1).numpy()
    auc_ae = auc(sc_ae, yw)
    return {
        "synthetic_svdd_auc": auc_svdd,
        "synthetic_svdd_ae_auc": auc_ae,
        "synthetic_svdd_gain": auc_svdd - auc_ae,
        "synthetic_torch_available": 1.0,
    }
