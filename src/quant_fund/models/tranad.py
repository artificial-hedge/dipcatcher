"""TranAD (Tuli et al. 2022) — transformer with self-conditioning:
encoder attends over the window, decoder re-encodes its own output;
score = 2-stage recon error — vs plain AE.
"""

from __future__ import annotations

from quant_fund.models._anom_synth import anom_series, auc, windows


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("tranad requires torch (pip install -e .[nn])") from exc
    return torch


def bench_tranad(
    seed: int = 637,
    w: int = 16,
    iters: int = 100,
    h: int = 16,
) -> dict[str, float]:
    torch = _torch()
    x, y = anom_series(seed=seed)
    wins, ctr = windows(x, w)
    yw = y[ctr]
    normal = wins[yw == 0]
    n_tr = int(0.7 * len(normal))
    tr = torch.tensor(normal[:n_tr]).float()
    all_w = torch.tensor(wins).float()
    torch.manual_seed(seed)
    enc = torch.nn.Linear(3, h)
    attn = torch.nn.MultiheadAttention(h, 2, batch_first=True)
    dec = torch.nn.Linear(h, 3)
    attn2 = torch.nn.MultiheadAttention(h, 2, batch_first=True)
    dec2 = torch.nn.Linear(h, 3)
    params = (
        list(enc.parameters())
        + list(attn.parameters())
        + list(dec.parameters())
        + list(attn2.parameters())
        + list(dec2.parameters())
    )
    opt = torch.optim.Adam(params, lr=0.005)
    for _i in range(iters):
        z = enc(tr)
        a1, _w = attn(z, z, z)
        r1 = dec(a1)
        z2 = enc(r1)
        a2, _w = attn2(z2, z2, z2)
        r2 = dec2(a2)
        loss = ((r1 - tr) ** 2).mean() + ((r2 - tr) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        z = enc(all_w)
        a1, _w = attn(z, z, z)
        r1 = dec(a1)
        z2 = enc(r1)
        a2, _w = attn2(z2, z2, z2)
        r2 = dec2(a2)
        sc = ((r1 - all_w) ** 2).mean(-1).mean(-1).numpy() + ((r2 - all_w) ** 2).mean(-1).mean(
            -1
        ).numpy()
    auc_t = auc(sc, yw)
    torch.manual_seed(seed)
    flat = torch.tensor(normal[:n_tr].reshape(n_tr, -1)).float()
    ae = torch.nn.Sequential(torch.nn.Linear(w * 3, h), torch.nn.ReLU(), torch.nn.Linear(h, w * 3))
    opt = torch.optim.Adam(ae.parameters(), lr=0.005)
    for _i in range(iters):
        loss = ((ae(flat) - flat) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        af = all_w.reshape(len(all_w), -1)
        sc_ae = ((ae(af) - af) ** 2).mean(-1).numpy()
    auc_ae = auc(sc_ae, yw)
    return {
        "synthetic_tranad_auc": auc_t,
        "synthetic_tranad_ae_auc": auc_ae,
        "synthetic_tranad_gain": auc_t - auc_ae,
        "torch_available": 1.0,
    }
