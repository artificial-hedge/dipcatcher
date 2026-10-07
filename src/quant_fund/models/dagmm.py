"""DAGMM (Zong et al. 2018).

Autoencoder → (recon features, z) → GMM estimation net; anomaly score =
negative mixture likelihood of test windows — vs plain AE recon error.
"""

from __future__ import annotations

from quant_fund.models._anom_synth import anom_series, auc, windows


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("dagmm requires torch (pip install -e .[nn])") from exc
    return torch


def bench_dagmm(
    seed: int = 619,
    w: int = 16,
    iters: int = 140,
    k: int = 3,
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
    dec = torch.nn.Sequential(torch.nn.Linear(4, h), torch.nn.ReLU(), torch.nn.Linear(h, w * 3))
    est = torch.nn.Sequential(
        torch.nn.Linear(6, h), torch.nn.ReLU(), torch.nn.Linear(h, k), torch.nn.Softmax(-1)
    )
    params = list(enc.parameters()) + list(dec.parameters()) + list(est.parameters())
    opt = torch.optim.Adam(params, lr=0.005)

    def feats(fx):
        z = enc(fx)
        rec = dec(z)
        cos = torch.nn.functional.cosine_similarity(fx, rec)
        rel = ((fx - rec) ** 2).mean(-1) / (fx**2).mean(-1).clamp_min(1e-6)
        return z, rec, torch.cat([z, cos[:, None], rel[:, None]], 1)

    def energy(fx):
        z, _r, ef = feats(fx)
        gamma = est(ef)  # (n,k)
        mu = (gamma.T @ z) / gamma.sum(0)[:, None].clamp_min(1e-6)
        en = torch.zeros(len(z))
        for kk in range(k):
            d = z - mu[kk]
            cov = (gamma[:, kk : kk + 1] * d).T @ d / gamma[:, kk].sum().clamp_min(1e-6)
            cov = cov + 1e-3 * torch.eye(4)
            en += (
                gamma[:, kk] * torch.einsum("ni,ij,nj->n", d, torch.linalg.inv(cov), d) * 0.5
                + gamma[:, kk] * torch.log(torch.det(cov).clamp_min(1e-9)) * 0.5
            )
        return en, est

    for _i in range(iters):
        z, rec, _ef = feats(tr)
        en, _est = energy(tr)
        recon = ((rec - tr) ** 2).mean()
        loss = recon + 0.1 * en.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        sc, _est = energy(all_f)
    auc_d = auc(sc.numpy(), yw)
    # AE recon baseline
    with torch.no_grad():
        _, rec, _ef = feats(all_f)
        sc_ae = ((rec - all_f) ** 2).mean(-1).numpy()
    auc_ae = auc(sc_ae, yw)
    return {
        "synthetic_dagmm_auc": auc_d,
        "synthetic_dagmm_ae_auc": auc_ae,
        "synthetic_dagmm_gain": auc_d - auc_ae,
        "synthetic_torch_available": 1.0,
    }
