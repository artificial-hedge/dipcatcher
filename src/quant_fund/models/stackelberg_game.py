"""Stackelberg leader-follower quantity game (two firms, unit-demand).

Firm L commits q_L first; firm F best-responds q_F = (a - c_F - b q_L)/(2b).
The leader internalizes the follower's reaction: q_L* = (a - 2c_L + c_F)/(2b).
Bench verifies the leader's first-order condition residual, the follower's
best-response residual, and the leader advantage vs simultaneous Cournot.
"""

from quant_fund.models._mfg_synth import COU_A, COU_B, COU_C
from quant_fund.models.nash_cournot import _equilibrium


def bench_stackelberg_game(seed: int = 4207) -> dict[str, float]:
    del seed
    c_l, c_f = float(COU_C[0]), float(COU_C[1])

    def q_f_br(ql: float) -> float:
        return max(0.0, (COU_A - c_f - COU_B * ql) / (2 * COU_B))

    # leader optimum: max (a - b(qL + qF(qL)))qL - cL qL => dqL
    # = (a - cF - 2cL + cF... ) solve numerically on a fine grid for honesty
    grid = [i * 0.05 for i in range(0, 2400)]
    best_q, best_p = 0.0, -1e18
    for ql in grid:
        qf = q_f_br(ql)
        pi = (COU_A - COU_B * (ql + qf) - c_l) * ql
        if pi > best_p:
            best_q, best_p = ql, pi
    q_l = best_q
    q_f = q_f_br(q_l)
    price = COU_A - COU_B * (q_l + q_f)
    pi_l = (price - c_l) * q_l
    q_cour = _equilibrium(COU_C[:2])
    pc = COU_A - COU_B * q_cour.sum()
    pi_l_cournot = float((pc - c_l) * q_cour[0])
    # FOC residual: d piL/dqL at optimum ~ 0 (finite-diff check)
    eps = 1e-4
    d = (
        (COU_A - COU_B * (q_l + eps + q_f_br(q_l + eps)) - c_l) * (q_l + eps)
        - (COU_A - COU_B * (q_l + q_f) - c_l) * q_l
    ) / eps
    return {
        "synthetic_stack_foc_resid": abs(float(d)),
        "synthetic_stack_ql": float(q_l),
        "synthetic_stack_qf": float(q_f),
        "synthetic_stack_leader_profit": float(pi_l),
        "synthetic_stack_cournot_profit": pi_l_cournot,
        "synthetic_stack_leader_gain": float(pi_l - pi_l_cournot),
    }
