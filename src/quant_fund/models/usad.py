"""USAD (Audibert et al. 2020) — autoencoder with two decoders trained (SYNTHETIC)
adversarially (D1 reconstructs honestly, D2 fools D1); score = recon error
+ mutual-discrepancy weight — vs single AE.
"""

from __future__ import annotations

from quant_fund.models._anom_synth import anom_series, auc, windows


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("usad requires torch (pip install -e .[nn])") from exc
    return torch


def bench_usad(
    seed: int = 623,
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
    tr = torch.tensor(normal[:n_tr].reshape(n_tr, -1)).float()
    all_f = torch.tensor(wins.reshape(len(wins), -1)).float()
    torch.manual_seed(seed)
    enc = torch.nn.Sequential(torch.nn.Linear(w * 3, h), torch.nn.ReLU(), torch.nn.Linear(h, 4))
    d1 = torch.nn.Sequential(torch.nn.Linear(4, h), torch.nn.ReLU(), torch.nn.Linear(h, w * 3))
    d2 = torch.nn.Sequential(torch.nn.Linear(4, h), torch.nn.ReLU(), torch.nn.Linear(h, w * 3))
    opt = torch.optim.Adam(
        list(enc.parameters()) + list(d1.parameters()) + list(d2.parameters()), lr=0.005
    )
    for ep in range(iters):
        a = ep / max(iters - 1, 1)
        z = enc(tr)
        r1 = d1(z)
        z2 = enc(r1)
        r2 = d2(z2)
        z2b = enc(r2)
        r1b = d1(z2b)
        l1 = a * ((r1 - tr) ** 2).mean() + (1 - a) * ((r1b - tr) ** 2).mean()
        l2 = a * ((r2 - tr) ** 2).mean() - (1 - a) * ((r1b - tr) ** 2).mean()
        loss = l1 + l2
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        z = enc(all_f)
        r1 = d1(z)
        r1b = d1(enc(d2(enc(r1))))
        sc = (
            0.5 * ((r1 - all_f) ** 2).mean(-1).numpy() + 0.5 * ((r1b - all_f) ** 2).mean(-1).numpy()
        )
    auc_u = auc(sc, yw)
    torch.manual_seed(seed)
    ae = torch.nn.Sequential(torch.nn.Linear(w * 3, h), torch.nn.ReLU(), torch.nn.Linear(h, w * 3))
    opt = torch.optim.Adam(ae.parameters(), lr=0.005)
    for _i in range(iters):
        loss = ((ae(tr) - tr) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        sc_ae = ((ae(all_f) - all_f) ** 2).mean(-1).numpy()
    auc_ae = auc(sc_ae, yw)
    return {
        "synthetic_usad_auc": auc_u,
        "synthetic_usad_ae_auc": auc_ae,
        "synthetic_usad_gain": auc_u - auc_ae,
        "synthetic_torch_available": 1.0,
    }
