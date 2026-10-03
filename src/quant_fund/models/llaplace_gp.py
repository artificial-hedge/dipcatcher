"""Last-Layer Laplace GP — cheap post-hoc predictive uncertainty.

Daxberger et al. 2021: keep a trained regression net, place a Gaussian
posterior on the last layer's weights via the Gauss-Newton Hessian
(Phi^T Phi + prior), and propagate phi(x)^T Sigma phi(x) into predictive
variance. Bench: coverage calibration vs homoscedastic point net. SYNTHETIC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.neural_process import synth_np_tasks

FloatArray = NDArray[np.float64]

_SEED = 20261006


def _torch() -> Any:
    import torch

    _ = torch.nn.Linear  # torch + nn extra required
    return torch


def bench_llaplace_gp(
    seed: int = 23,
    n_train: int = 300,
    n_test: int = 60,
    n_ctx: int = 10,
    n_tgt: int = 50,
    iters: int = 800,
    d: int = 48,
    prior: float = 1.0,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed + _SEED)
    tr = synth_np_tasks(n_train, n_ctx, n_tgt, rng)
    te = synth_np_tasks(n_test, n_ctx, n_tgt, rng)

    feat = torch.nn.Sequential(
        torch.nn.Linear(1, d), torch.nn.ReLU(), torch.nn.Linear(d, d), torch.nn.ReLU()
    )
    head = torch.nn.Linear(d, 1)
    opt = torch.optim.Adam(list(feat.parameters()) + list(head.parameters()), lr=1e-3)
    xc = torch.tensor(tr[0], dtype=torch.float32)
    yc = torch.tensor(tr[1], dtype=torch.float32)
    for _ in range(iters):
        mu = head(feat(xc))[..., 0]
        mse = ((mu - yc[..., 0]) ** 2).mean()
        opt.zero_grad()
        mse.backward()
        opt.step()

    with torch.no_grad():
        phi_c = feat(xc).reshape(-1, d)  # pooled features over all train pts
        y_all = yc.reshape(-1)
        mu_c = head(phi_c)[..., 0]
        resid = (y_all - mu_c).numpy()
        sigma2 = float(np.mean(resid**2)) + 1e-6
        # Gauss-Newton Hessian of last layer: Phi^T Phi / sigma^2 + prior I
        phi_np = phi_c.numpy()
        phi_aug = np.concatenate([np.asarray(phi_np), np.ones((phi_np.shape[0], 1))], 1)
        H = phi_aug.T @ phi_aug / sigma2 + prior * np.eye(d + 1)
        cov = np.linalg.inv(np.asarray(H))  # posterior cov on (w, b)

        xt_t = torch.tensor(te[2], dtype=torch.float32)
        phi_t = feat(xt_t).numpy()
        phi_t_aug = np.concatenate(
            [np.asarray(phi_t), np.ones((phi_t.shape[0], phi_t.shape[1], 1))], -1
        )
        w_aug = np.concatenate([head.weight.detach().numpy()[0], head.bias.detach().numpy()])
        mu_t = phi_t_aug @ np.asarray(w_aug)
        epi_var = (phi_t_aug @ cov * phi_t_aug).sum(-1)
        total_var = sigma2 + np.asarray(epi_var)
        # homoscedastic baseline: point net + sigma2 only
        cov90_epi = float(
            np.mean(np.abs(te[3][..., 0] - np.asarray(mu_t)) <= 1.645 * np.sqrt(total_var))
        )
        cov90_homo = float(
            np.mean(np.abs(te[3][..., 0] - np.asarray(mu_t)) <= 1.645 * np.sqrt(sigma2))
        )
        mse = float(np.mean((np.asarray(mu_t) - te[3][..., 0]) ** 2))
        # edge region |x|>2.2: epistemic var grows where data is sparse —
        # this is where last-layer Laplace earns its keep
        edge = np.abs(te[2][..., 0]) > 2.2
        cov_edge_epi = float(
            np.mean(
                np.abs(te[3][..., 0][edge] - np.asarray(mu_t)[edge])
                <= 1.645 * np.sqrt(total_var)[edge]
            )
        )
        cov_edge_homo = float(
            np.mean(np.abs(te[3][..., 0][edge] - np.asarray(mu_t)[edge]) <= 1.645 * np.sqrt(sigma2))
        )
    return {
        "synthetic_llap_cov90": cov90_epi,
        "synthetic_llap_cov90_homoscedastic": cov90_homo,
        "synthetic_llap_cov90_err": abs(cov90_epi - 0.90),
        "synthetic_llap_mse": mse,
        "synthetic_llap_cov90_edge": cov_edge_epi,
        "synthetic_llap_cov90_edge_homo": cov_edge_homo,
        "synthetic_llap_edge_gain": cov_edge_epi - cov_edge_homo,
        "synthetic_llap_sigma2": sigma2,
        "torch_available": 1.0,
    }
