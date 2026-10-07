"""SWA-Gaussian (Maddox et al. 2019) — diagonal posterior over the last (SYNTHETIC)
checkpoint sequence: mean + diag std over SGD iterates at high LR; sample
weights → predictive mean/var. NLL + OOD gap vs point MLP.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._bdl_synth import bdl_data, coverage, nll_gauss


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("swag_diag requires torch (pip install -e .[nn])") from exc
    return torch


def bench_swag_diag(
    seed: int = 787, iters: int = 300, collect: int = 60, n_samp: int = 20
) -> dict[str, float]:
    torch = _torch()
    x, y, x_te, y_te, x_ood = bdl_data(seed)
    X = torch.tensor(x).float()
    Y = torch.tensor(y).float()[:, None]
    Xt = torch.tensor(x_te).float()
    Xo = torch.tensor(x_ood).float()
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    opt = torch.optim.SGD(net.parameters(), lr=0.02)
    snaps: list[dict] = []
    for i in range(iters):
        loss = ((net(X).squeeze(-1) - Y.squeeze(-1)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i >= iters - collect and i % 3 == 0:
            snaps.append({k: v.detach().clone() for k, v in net.state_dict().items()})
    # posterior mean/diag-std over snapshots
    mean = {k: torch.stack([s[k] for s in snaps]).mean(0) for k in snaps[0]}
    sd = {k: torch.stack([s[k] for s in snaps]).std(0) + 1e-6 for k in snaps[0]}
    rng = np.random.default_rng(seed)
    preds_te = []
    preds_ood = []
    with torch.no_grad():
        for _ in range(n_samp):
            sd_state = {
                k: mean[k] + torch.tensor(rng.standard_normal(mean[k].shape)).float() * sd[k]
                for k in mean
            }
            net.load_state_dict(sd_state)
            preds_te.append(net(Xt).squeeze(-1).numpy())
            preds_ood.append(net(Xo).squeeze(-1).numpy())
    P = np.asarray(preds_te)
    mu, var_raw = P.mean(0), P.var(0)
    var = var_raw + 0.09  # + aleatoric noise floor
    Po = np.asarray(preds_ood)
    var_ood = Po.var(0).mean()
    return {
        "synthetic_swag_nll": nll_gauss(y_te, mu, var),
        "synthetic_swag_cov95": coverage(y_te, mu, np.sqrt(var)),
        "synthetic_swag_ood_gap": float(var_ood / (var_raw.mean() + 1e-9)),
        "synthetic_torch_available": 1.0,
    }
