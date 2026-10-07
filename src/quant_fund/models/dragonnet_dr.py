"""Dragonnet-lite — outcome heads + propensity head (Shi et al. 2019).

Joint propensity + outcome; doubly-robust AIPW pseudo-outcome trained
head → ATE error vs outcome-only regression under confounding.
"""

from __future__ import annotations

from quant_fund.models._causal_synth import synth_observational


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("dragonnet_dr requires torch (pip install -e .[nn])") from exc
    return torch


def bench_dragonnet_dr(
    seed: int = 563,
    n: int = 800,
    iters: int = 120,
) -> dict[str, float]:
    torch = _torch()
    x, t, y, tau, _e = synth_observational(seed, n)
    half = n // 2
    x_t = torch.tensor(x[:half]).float()
    t_t = torch.tensor(t[:half])
    y_t = torch.tensor(y[:half]).float()
    x_te = torch.tensor(x[half:]).float()
    t_te = t[half:]
    y_te = y[half:]
    tau_te = tau[half:]
    torch.manual_seed(seed)
    trunk = torch.nn.Sequential(torch.nn.Linear(5, 24), torch.nn.ReLU())
    h0 = torch.nn.Linear(24, 1)
    h1 = torch.nn.Linear(24, 1)
    hp = torch.nn.Linear(24, 1)
    opt = torch.optim.Adam(
        list(trunk.parameters())
        + list(h0.parameters())
        + list(h1.parameters())
        + list(hp.parameters()),
        lr=0.01,
    )
    for _i in range(iters):
        z = trunk(x_t)
        tf = t_t.float()
        mse = ((h1(z).squeeze(-1) * tf + h0(z).squeeze(-1) * (1 - tf)) - y_t) ** 2
        bce = torch.nn.functional.binary_cross_entropy_with_logits(hp(z).squeeze(-1), tf)
        loss = mse.mean() + bce
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        z = trunk(x_te)
        e_hat = torch.sigmoid(hp(z).squeeze(-1)).numpy().clip(0.02, 0.98)
        m0 = h0(z).squeeze(-1).numpy()
        m1 = h1(z).squeeze(-1).numpy()
    # AIPW estimate of ATE
    aipw = (m1 - m0) + t_te * (y_te - m1) / e_hat - (1 - t_te) * (y_te - m0) / (1 - e_hat)
    ate_dr = float(aipw.mean())
    ate_reg = float((m1 - m0).mean())
    ate_true = float(tau_te.mean())
    return {
        "synthetic_drag_dr_err": abs(ate_dr - ate_true),
        "synthetic_drag_reg_err": abs(ate_reg - ate_true),
        "synthetic_drag_gain": abs(ate_reg - ate_true) - abs(ate_dr - ate_true),
        "synthetic_torch_available": 1.0,
    }
