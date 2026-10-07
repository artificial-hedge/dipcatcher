"""Sparse autoencoder feature recovery (Bricken et al. 2023) (SYNTHETIC).

Activations a = s·F (sparse superposition, 8 ground-truth directions
in 16 dims). An SAE with L1 penalty recovers decoder directions; the
max-|cos| match between learned and true features quantifies recovery
vs a PCA baseline.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._interp_synth import D_ACT, N_FEAT, feature_directions, synth_activations


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("sae_feature needs the torch `nn` extra") from exc


def _best_cos(rec: np.ndarray, truth: np.ndarray) -> float:
    rn = rec / np.linalg.norm(rec, axis=-1, keepdims=True)
    tn = truth / np.linalg.norm(truth, axis=-1, keepdims=True)
    sims = np.abs(rn @ tn.T)
    return float(sims.max(0).mean())  # each true feature's best match


def bench_sae_feature(
    seed: int = 263,
    n: int = 2000,
    hidden: int = 32,
    l1: float = 0.03,
    iters: int = 400,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dirs = feature_directions(rng)
    a, s_true = synth_activations(n, dirs, rng)
    a_t = torch.tensor(a).float()

    enc = torch.nn.Linear(D_ACT, hidden)
    dec = torch.nn.Linear(hidden, D_ACT)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=5e-3)
    for _i in range(iters):
        h = torch.relu(enc(a_t))
        rec = dec(h)
        loss = ((a_t - rec) ** 2).mean() + l1 * h.abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        w_dec = dec.weight.detach().numpy()  # (D_ACT, hidden)
        m_sae = _best_cos(w_dec.T, dirs)
        # PCA baseline
        cov = np.cov(a.T)
        _, v = np.linalg.eigh(cov)
        m_pca = _best_cos(v[:, -N_FEAT:].T, dirs)
    return {
        "synthetic_sae_match": m_sae,
        "synthetic_sae_pca_match": m_pca,
        "synthetic_sae_gain": m_sae - m_pca,
        "synthetic_torch_available": 1.0,
    }
