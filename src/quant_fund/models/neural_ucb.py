"""NeuralUCB (Zhou et al. 2020) — contextual bandit with a small net
predicting reward; UCB bonus from ridge features (penultimate activs).
Regret vs LinUCB-lite on nonlinear context reward.
"""

from __future__ import annotations

import numpy as np


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("neural_ucb requires torch (pip install -e .[nn])") from exc
    return torch


def bench_neural_ucb(seed: int = 1429, T: int = 1800, K: int = 4, d: int = 6) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    # context reward: nonlinear in x, arm-specific
    W = rng.normal(0, 1, (K, d))
    net = torch.nn.Sequential(torch.nn.Linear(d, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.02)
    tot, tot2 = 0.0, 0.0
    A_r = np.eye(16) * 1.0
    A_lin = np.eye(d) * 1.0
    b_lin = np.zeros(d)
    hist_x, hist_a, hist_r = [], [], []
    for t in range(T):
        x = rng.normal(0, 1, d)
        true_r = np.tanh(x @ W.T)  # nonlinear arm rewards
        with torch.no_grad():
            xt = torch.tensor(x).float()
            h = net[:-1](xt).numpy()  # penultimate 16-d
            mu = float(net(xt))
        phi = h
        Ainv = np.linalg.inv(A_r)
        bonus = np.sqrt(phi @ Ainv @ phi)
        a = int(np.argmax(mu + 1.0 * bonus + rng.normal(0, 0.05, K)))
        r = true_r[a] + 0.05 * rng.standard_normal()
        A_r += np.outer(phi, phi)
        hist_x.append(x)
        hist_a.append(a)
        hist_r.append(r)
        tot += true_r[a]
        if (t + 1) % 50 == 0 and len(hist_r) > 20:
            Xs = torch.tensor(np.asarray(hist_x)).float()
            Rs = torch.tensor(np.asarray(hist_r)).float()
            for _ in range(20):
                loss = ((net(Xs).squeeze(-1)[range(len(Rs))] - Rs) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        # LinUCB baseline on same context (shared features across arms approximated)
        Ainv2 = np.linalg.inv(A_lin)
        score = np.array(
            [
                float(x @ np.linalg.solve(A_lin, b_lin)) + 0.3 * np.sqrt(x @ Ainv2 @ x)
                for _ in range(K)
            ]
        )
        a2 = int(np.argmax(score))
        A_lin += np.outer(x, x)
        b_lin += true_r[a2] * x
        tot2 += true_r[a2]
    return {
        "synthetic_nucb_reward_sum": tot / T,
        "synthetic_nucb_linucb_sum": tot2 / T,
        "synthetic_nucb_gain": (tot - tot2) / T,
        "torch_available": 1.0,
    }
