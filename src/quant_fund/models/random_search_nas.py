"""Random-search NAS baseline (Bergstra-Bengio 2012; Li-Talwalkar 2019).

Uniform sampling over the arch space vs a grid that evaluates the
same budget lexicographically — random coverage finds better archs
when importance is concentrated in few dims.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._compress_synth import split
from quant_fund.models._nas_synth import _torch, all_archs, eval_arch, noisy_labels, rand_arch


def bench_random_search_nas(
    seed: int = 457,
    n: int = 300,
    budget: int = 12,
    iters: int = 40,
) -> dict[str, float]:
    torch = _torch()
    x_tr, y_tr, x_te, y_te = split(seed, n)
    x_tr_t = torch.tensor(x_tr).float()
    y_tr_t = torch.tensor(noisy_labels(y_tr, seed))
    x_te_t = torch.tensor(x_te).float()
    y_te_t = torch.tensor(y_te)
    rng = np.random.default_rng(seed)
    space = all_archs()
    # oracle best (full sweep)
    accs = {
        a: eval_arch(a, x_tr_t, y_tr_t, x_te_t, y_te_t, iters, seed=i) for i, a in enumerate(space)
    }
    oracle = max(accs.values())
    # random budget
    picked = set()
    best_r = 0.0
    for _i in range(budget):
        a = rand_arch(rng)
        picked.add(a)
        best_r = max(best_r, accs[a])
    # grid (lexicographic order, same budget)
    best_g = max(accs[a] for a in space[:budget])
    return {
        "synthetic_rnas_best": best_r,
        "synthetic_rnas_grid_best": best_g,
        "synthetic_rnas_oracle": oracle,
        "synthetic_rnas_gain": best_r - best_g,
        "torch_available": 1.0,
    }
