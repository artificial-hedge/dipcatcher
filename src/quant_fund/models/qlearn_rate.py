"""Q-learning convergence on a tiny stochastic-action MDP.

Random-walk exploration with alpha_t = 1/visits(s,a) satisfies
Robbins-Monro and converges to Q*. On a deterministic-reward MDP the
constant-alpha variant is effectively async VI and can converge faster
finite-horizon — the bench reports both curves honestly, whichever
wins.
"""

import numpy as np

from quant_fund.models._rlt_synth import MDP_GAMMA
from quant_fund.models.pi_contraction import S_R, _trans, _vi


def _qlearn(seed: int, steps: int, decay: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, na = len(S_R), 2
    q = np.zeros((n, na))
    visits = np.zeros((n, na))
    v_star = _vi()
    errs = np.zeros(steps)
    for t in range(steps):
        s = int(rng.integers(n))
        a = int(rng.integers(na))
        tgt = _trans(s, a)
        visits[s, a] += 1
        alpha = 1.0 / visits[s, a] if decay else 0.05
        q[s, a] += alpha * (S_R[s] + MDP_GAMMA * q[tgt].max() - q[s, a])
        errs[t] = float(np.max(np.abs(q.max(axis=1) - v_star)))
    return errs


def bench_qlearn_rate(seed: int = 4611) -> dict[str, float]:
    steps = 60000
    e_dec = _qlearn(seed, steps, True)
    e_con = _qlearn(seed + 1, steps, False)
    return {
        "synthetic_ql_err_early": float(e_dec[steps // 10 - 1]),
        "synthetic_ql_err_late": float(e_dec[-1]),
        "synthetic_ql_const_floor": float(e_con[-1]),
        "synthetic_ql_decay_gain": float(e_con[-1] - e_dec[-1]),
        "synthetic_ql_decay_wins": float(e_dec[-1] < e_con[-1]),
        "synthetic_ql_ratio": float(e_dec[steps // 10 - 1] / max(e_dec[-1], 1e-9)),
    }
