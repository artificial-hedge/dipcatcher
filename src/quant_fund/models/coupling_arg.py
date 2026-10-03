"""Coupling bound: TV distance <= P(chains disagree) after coupling (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _bench_coupling_arg(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    checks = []
    # two-state chain on {0,1}, p(0->1)=p(1->0)=p; start coupled at step k
    p = 0.3
    n = 40
    n_rep = 30000
    # independent starts: X_0 = 0, Y_0 = 1, same moves after meeting
    meet = np.zeros(n_rep, dtype=bool)
    final_same = np.zeros(n_rep, dtype=bool)
    for i in range(n_rep):
        x, y = 0, 1
        met = False
        for _ in range(n):
            if met:
                fl = rng.random() < p
                x = y = 1 - x if fl else x
            else:
                fx = rng.random() < p
                fy = rng.random() < p
                x = 1 - x if fx else x
                y = 1 - y if fy else y
                met = x == y
        meet[i] = met
        final_same[i] = x == y
    # P(disagree) <= P(not yet met); TV marginal distance bounded by P(not met)
    checks.append(float(np.mean(~meet)) < 0.02)  # meets fast: geo mean 2p
    # marginals: from x0=0, P(X_n = 1) -> 0.5; disagree prob bounds |P_X - P_Y|
    checks.append(abs(float(np.mean(final_same)) - 1.0) < 0.05)
    # TV distance decays like (1-2p)^n: |P(X_n=1|X_0=0) - 0.5| = 0.5|1-2p|^n
    exact = 0.5 * abs(1 - 2 * p) ** n
    checks.append(exact < 1e-9)
    # coupling inequality holds: TV <= P(tau > n); both ~ (1-2p)^n/2
    checks.append(float(np.mean(~meet)) >= float(np.mean(~final_same)) - 1e-6)
    return float(sum(checks) / len(checks))


def bench_coupling_arg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_coupling_arg": _bench_coupling_arg(seed)}
