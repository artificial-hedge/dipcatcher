"""DeepIV-style two-stage IV (Hartford et al. 2017) (SYNTHETIC).

Stage 1: z + net → t̂; Stage 2: t̂ → y. Effect estimate vs OLS
(confounded) — IV recovers the true causal slope 1.5.
"""

from __future__ import annotations

from sklearn.linear_model import LinearRegression

from quant_fund.models._causal_synth import synth_iv


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("deep_iv requires torch (pip install -e .[nn])") from exc
    return torch


def bench_deep_iv(
    seed: int = 569,
    n: int = 800,
    iters: int = 150,
) -> dict[str, float]:
    torch = _torch()
    z, t, y, _ = synth_iv(seed, n)
    half = n // 2
    z_t = torch.tensor(z[:half, None]).float()
    t_t = torch.tensor(t[:half, None]).float()
    z_te = z[half:]
    t_te = t[half:]
    y_te = y[half:]
    torch.manual_seed(seed)
    stage1 = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    opt = torch.optim.Adam(stage1.parameters(), lr=0.02)
    for _i in range(iters):
        loss = ((stage1(z_t).squeeze(-1) - t_t.squeeze(-1)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        t_hat_te = stage1(torch.tensor(z_te[:, None]).float()).squeeze(-1).numpy()
    # stage 2: y ~ t_hat
    lr2 = LinearRegression().fit(t_hat_te[:, None], y_te)
    iv_est = float(lr2.coef_[0])
    ols = LinearRegression().fit(t_te[:, None], y_te)
    ols_est = float(ols.coef_[0])
    return {
        "synthetic_div_iv_est": iv_est,
        "synthetic_div_ols_est": ols_est,
        "synthetic_div_true": 1.5,
        "synthetic_div_iv_err": abs(iv_est - 1.5),
        "synthetic_div_ols_err": abs(ols_est - 1.5),
        "synthetic_torch_available": 1.0,
    }
