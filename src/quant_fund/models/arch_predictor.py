"""Surrogate arch-performance predictor (Wen et al. 2020, NASBench) (SYNTHETIC).

An MLP surrogate trained on few (arch encoding → acc) pairs predicts
held-out arch performance — rank corr + best-arch-found vs budget.
"""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor

from quant_fund.models._compress_synth import split
from quant_fund.models._nas_synth import _torch, all_archs, eval_arch, noisy_labels


def _enc(arch) -> list[float]:
    h, d, a = arch
    return [h / 48.0, d / 3.0, float(a), h * d / 144.0]


def bench_arch_predictor(
    seed: int = 491,
    n: int = 300,
    n_train: int = 15,
    iters: int = 30,
) -> dict[str, float]:
    torch = _torch()
    x_tr, y_tr, x_te, y_te = split(seed, n)
    x_tr_t = torch.tensor(x_tr).float()
    y_tr_t = torch.tensor(noisy_labels(y_tr, seed))
    x_te_t = torch.tensor(x_te).float()
    y_te_t = torch.tensor(y_te)
    rng = np.random.default_rng(seed)
    space = all_archs()
    cache: dict = {}

    def acc(a):
        if a not in cache:
            cache[a] = eval_arch(a, x_tr_t, y_tr_t, x_te_t, y_te_t, iters, seed=hash(a) % 997)
        return cache[a]

    idx = rng.choice(len(space), n_train, replace=False)
    X = np.array([_enc(space[i]) for i in idx])
    y = np.array([acc(space[i]) for i in idx])
    sur = MLPRegressor(hidden_layer_sizes=(32,), max_iter=800, random_state=seed).fit(X, y)
    # predict all
    Xall = np.array([_enc(a) for a in space])
    pred = sur.predict(Xall)
    # evaluate predicted top-3 → best found
    top = np.argsort(pred)[-3:]
    best_found = max(acc(space[i]) for i in top)
    # true accs on all (for corr) — reuse cache + eval missing lazily
    true_all = np.array([acc(a) for a in space])
    rho = float(np.corrcoef(np.argsort(np.argsort(pred)), np.argsort(np.argsort(true_all)))[0, 1])
    # random picking 3
    rng2 = np.random.default_rng(seed + 1)
    best_rand = max(true_all[i] for i in rng2.choice(len(space), 3, replace=False))
    return {
        "synthetic_ap_rank_rho": rho,
        "synthetic_ap_best_found": best_found,
        "synthetic_ap_random_best": best_rand,
        "synthetic_ap_oracle": float(true_all.max()),
        "synthetic_torch_available": 1.0,
    }
