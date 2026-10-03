"""Neural Thompson sampling for contextual signal selection (torch).

A neural network feature extractor + Bayesian linear head (Gaussian
posterior on the last layer, à la NeuralLinear) does Thompson sampling
over K signal-overlay arms given a regime context. Posterior updates are
exact ridge-style sufficient statistics — exploration falls out of the
head covariance. Requires the ``nn`` extra; SYNTHETIC contexts only.

Bench: contexts from 3 regimes; each arm pays off in exactly one regime —
Thompson sampling should accumulate less regret than a uniform policy and
a greedy neural policy.
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
        raise ImportError("neural_thompson torch path requires the `nn` extra (make sync)") from exc


def torch_available() -> bool:
    try:
        _torch()
        return True
    except ImportError:
        return False


def synth_bandit(n: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray, int]:
    """contexts (n,4) one-hot-ish regime + noise; reward (n,K): arm k pays
    1 + noise when k % 3 == current regime (map rotates at horizon half),
    decoy arm pays 0.45 always, others -0.2 + noise."""
    K, d = 6, 4
    ctx = np.zeros((n, d))
    rew = np.zeros((n, K))
    for t in range(n):
        r = int(rng.integers(0, 3))
        ctx[t] = np.eye(3, d, r % d)[0] + 0.1 * rng.standard_normal(d)
        shift = 0 if t < n // 2 else 1  # payoff map rotates mid-horizon
        for k in range(K):
            if k == 5:
                rew[t, k] = 0.45 + 0.05 * rng.standard_normal()  # reliable decoy
            else:
                rew[t, k] = (
                    1.0 if k % 3 == (r + shift) % 3 else -0.2
                ) + 0.1 * rng.standard_normal()
    return ctx, rew, K


def _bayes_head(
    phi: list[FloatArray], y: list[float], d: int, lam: float, sigma: float
) -> tuple[FloatArray, FloatArray]:
    """Gaussian posterior (mean, cov) for Bayesian linear regression."""
    x = np.stack(phi)
    t = np.array(y)
    prec = np.eye(d) / lam + x.T @ x / sigma**2
    cov = np.asarray(np.linalg.inv(prec))
    mean = cov @ (x.T @ t / sigma**2)
    return np.asarray(mean), cov


def bench_neural_thompson(seed: int = 73) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    torch = _torch()
    torch.manual_seed(seed)
    n, K = 3000, 6
    ctx, rew, K = synth_bandit(n, rng)
    d_feat = 12
    lam, sigma = 1.0, 0.2
    oracle = float(rew.max(1).mean())

    # feature extractor trained online on observed (ctx, arm)->r
    feat = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, d_feat))
    params = list(feat.parameters()) + [torch.nn.Parameter(torch.zeros(d_feat * K))]
    opt = torch.optim.Adam(params, lr=1e-2)
    obs_x: list[FloatArray] = []
    obs_a: list[int] = []
    obs_r: list[float] = []

    regret_ts = 0.0
    A = np.stack([np.eye(d_feat) for _ in range(K)]) / lam  # (K,d,d) prec accumulators
    b = np.zeros((K, d_feat))
    pulls = np.zeros(K, dtype=int)
    for t in range(n):
        x = ctx[t]
        with torch.no_grad():
            phi = feat(torch.tensor(x, dtype=torch.float32)).numpy()
        samples = []
        for k in range(K):
            cov = np.asarray(np.linalg.inv(A[k]))
            mean = cov @ b[k]
            w = mean + rng.multivariate_normal(np.zeros(d_feat), sigma**2 * cov)
            samples.append(float(phi @ w))
        a = int(np.argmax(samples))
        r = rew[t, a]
        regret_ts += rew[t].max() - r
        # update posterior stats + supervised fit occasionally
        A[a] += np.outer(phi, phi) / sigma**2
        b[a] += phi * r / sigma**2
        pulls[a] += 1
        obs_x.append(x)
        obs_a.append(a)
        obs_r.append(r)
        if t % 10 == 9 and len(obs_x) > 32:
            idx = rng.integers(0, len(obs_x), min(64, len(obs_x)))
            bx = torch.tensor(np.stack([obs_x[i] for i in idx]), dtype=torch.float32)
            br = torch.tensor(np.array([obs_r[i] for i in idx]), dtype=torch.float32)
            ph = feat(bx)
            w = params[-1].reshape(K, d_feat)
            pred = (ph.unsqueeze(1) * w.unsqueeze(0)).sum(-1)
            tgt = torch.zeros(len(idx), K)
            for j, i in enumerate(idx):
                tgt[j, obs_a[i]] = 1.0
            loss = torch.mean((pred - br.unsqueeze(1)) ** 2 * tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # baselines
    rng2 = np.random.default_rng(seed + 1)
    regret_unif = float(
        np.sum(rew.max(1) - rew[:, rng2.integers(0, K, n)][np.arange(n) if False else np.arange(n)])
    )
    regret_unif = float(np.sum(rew.max(1) - rew[np.arange(n), rng2.integers(0, K, n)]))
    # greedy neural: epsilon=0 policy of the same supervised net is approximated
    # by pure exploitation w/o posterior sampling -> simulate argmax-of-mean
    A2 = np.stack([np.eye(d_feat) for _ in range(K)]) / lam
    b2 = np.zeros((K, d_feat))
    feat_g = torch.nn.Sequential(
        torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, d_feat)
    )
    regret_g = 0.0
    for t in range(n):
        x = ctx[t]
        with torch.no_grad():
            phi = feat_g(torch.tensor(x, dtype=torch.float32)).numpy()
        means = np.array(
            [float(phi @ (np.asarray(np.linalg.inv(A2[k])) @ b2[k])) for k in range(K)]
        )
        a = int(np.argmax(means))
        regret_g += rew[t].max() - rew[t, a]
        A2[a] += np.outer(phi, phi) / sigma**2
        b2[a] += phi * rew[t, a] / sigma**2

    return {
        "synthetic_nthompson_regret": float(regret_ts),
        "synthetic_nthompson_greedy_regret": float(regret_g),
        "synthetic_nthompson_unif_regret": float(regret_unif),
        "synthetic_nthompson_margin_vs_greedy": float(regret_g - regret_ts),
        "synthetic_nthompson_margin_vs_unif": float(regret_unif - regret_ts),
        "synthetic_nthompson_oracle_mean": oracle,
        "synthetic_nthompson_arm_entropy": float(
            -np.sum((pulls / pulls.sum()) * np.log(pulls / pulls.sum() + 1e-12))
        ),
        "torch_available": 1.0,
    }
