"""Regularized evolution NAS (Real et al. 2019) (SYNTHETIC).

Population of archs; tournament selection; mutation on one gene;
aging — oldest removed. Best-found vs random-search at equal budget.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._compress_synth import split
from quant_fund.models._nas_synth import _torch, eval_arch, noisy_labels, rand_arch


def bench_evolution_nas(
    seed: int = 461,
    n: int = 300,
    pop_size: int = 8,
    budget: int = 24,
    iters: int = 40,
) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    x_tr, y_tr, x_te, y_te = split(seed, n)
    x_tr_t = torch.tensor(x_tr).float()
    y_tr_t = torch.tensor(noisy_labels(y_tr, seed))
    x_te_t = torch.tensor(x_te).float()
    y_te_t = torch.tensor(y_te)
    rng = np.random.default_rng(seed)
    cache: dict = {}

    def acc(a):
        if a not in cache:
            cache[a] = eval_arch(a, x_tr_t, y_tr_t, x_te_t, y_te_t, iters, seed=hash(a) % 1000)
        return cache[a]

    pop = [(rand_arch(rng), 0) for _i in range(pop_size)]  # (arch, birth-step)
    step = 0
    best_e = 0.0
    while step < budget:
        # tournament
        k = rng.choice(len(pop), min(4, len(pop)), replace=False)
        winner = max(k, key=lambda i: acc(pop[i][0]))
        parent = pop[winner][0]
        # mutate one gene
        genes = list(parent)
        g = int(rng.integers(3))
        if g == 0:
            genes[0] = int([4, 8, 16, 24, 48][rng.integers(5)])
        elif g == 1:
            genes[1] = int(rng.integers(3) + 1)
        else:
            genes[2] = int(rng.integers(2))
        child = tuple(genes)
        step += 1
        best_e = max(best_e, acc(child))
        pop.append((child, step))
        # aging: remove oldest
        pop.pop(0)
    # random same budget
    rng2 = np.random.default_rng(seed + 1)
    best_r = 0.0
    for _i in range(budget):
        best_r = max(best_r, acc(rand_arch(rng2)))
    return {
        "synthetic_enas_best": best_e,
        "synthetic_enas_random_best": best_r,
        "synthetic_enas_gain": best_e - best_r,
        "synthetic_torch_available": 1.0,
    }
