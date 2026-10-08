"""Variational autoencoder over option smile slices (torch).

Each 5-point synthetic implied-vol smile is encoded to a 2-d latent by a
Gaussian-approximate posterior; the decoder reconstructs the smile. The
latent should align with the true smile factors (level, skew) the
generator was built on. Requires the ``nn`` extra; SYNTHETIC smiles only.

Bench: recon MSE vs PCA-2 reconstruction, latent-factor |corr| with the
true level/skew drivers, KL sanity.
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
        raise ImportError("option_vae torch path requires the `nn` extra (make sync)") from exc


def torch_available() -> bool:
    try:
        _torch()
        return True
    except ImportError:
        return False


def synth_smiles(n: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """Smiles iv(m) = level + skew*m + 0.4*m^2 + noise on moneyness grid."""
    m = np.linspace(-0.1, 0.1, 5)
    level = 0.15 + 0.08 * rng.standard_normal(n)
    skew = -0.5 + 0.5 * rng.standard_normal(n)
    curv = 0.5 + 0.6 * rng.standard_normal(n)
    # nonlinear manifold: multiplicative level x curved smile shape
    shape = np.exp(
        skew[:, None] * np.tanh(30 * m[None, :]) + curv[:, None] * np.abs(m[None, :]) ** 1.5
    )
    iv = level[:, None] * shape + 0.005 * rng.standard_normal((n, 5))
    return iv, np.stack([level, skew, curv], 1)


def _pca_recon_err(iv: FloatArray, tr: int) -> float:
    mu = iv[:tr].mean(0)
    xc = iv[:tr] - mu
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    v = vt[:3].T
    rec = (iv[tr:] - mu) @ v @ v.T + mu
    return float(np.mean((iv[tr:] - rec) ** 2))


def bench_option_vae(seed: int = 79) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    torch = _torch()
    torch.manual_seed(seed)
    n, tr = 6000, 4800
    iv, factors = synth_smiles(n, rng)
    xs = torch.tensor(iv, dtype=torch.float32)

    enc = torch.nn.Sequential(
        torch.nn.Linear(5, 32), torch.nn.ReLU(), torch.nn.Linear(32, 16), torch.nn.ReLU()
    )
    mu_h = torch.nn.Linear(16, 3)
    lv_h = torch.nn.Linear(16, 3)
    dec = torch.nn.Sequential(
        torch.nn.Linear(3, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 5),
    )
    mods = torch.nn.ModuleList([enc, mu_h, lv_h, dec])
    opt = torch.optim.Adam(mods.parameters(), lr=3e-3)

    for _ in range(2500):
        idx = torch.randint(0, tr, (256,))
        h = enc(xs[idx])
        mu, lv = mu_h(h), lv_h(h).clamp(-6, 2)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(lv)
        rec = dec(z)
        recon = torch.mean((rec - xs[idx]) ** 2)
        kl = torch.mean(-0.5 * (1 + lv - mu**2 - torch.exp(lv)).sum(1))
        loss = recon + 0.01 * kl
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        h = enc(xs[tr:])
        z = mu_h(h).numpy()
        rec = dec(mu_h(h)).numpy()
        # generative: prior samples -> decoded smiles should match moments
        zs = torch.randn(2000, 3)
        gen = dec(zs).numpy()
    mse = float(np.mean((rec - iv[tr:]) ** 2))
    mse_pca = _pca_recon_err(iv, tr)
    c_lv = max(abs(np.corrcoef(z[:, i], factors[tr:, 0])[0, 1]) for i in range(3))
    c_sk = max(abs(np.corrcoef(z[:, i], factors[tr:, 1])[0, 1]) for i in range(3))
    c_cv = max(abs(np.corrcoef(z[:, i], factors[tr:, 2])[0, 1]) for i in range(3))
    mean_err = float(np.mean(np.abs(gen.mean(0) - iv[tr:].mean(0))))
    std_err = float(np.mean(np.abs(gen.std(0) - iv[tr:].std(0))))
    return {
        "synthetic_ovae_recon_mse": mse,
        "synthetic_ovae_pca_mse": mse_pca,
        "synthetic_ovae_margin_vs_pca": mse_pca - mse,
        "synthetic_ovae_level_corr": float(c_lv),
        "synthetic_ovae_skew_corr": float(c_sk),
        "synthetic_ovae_curv_corr": float(c_cv),
        "synthetic_ovae_gen_mean_err": mean_err,
        "synthetic_ovae_gen_std_err": std_err,
        "synthetic_torch_available": 1.0,
    }
