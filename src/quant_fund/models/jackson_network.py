"""Open Jackson network: product-form steady-state (SYNTHETIC).

Traffic equations lam = gamma + P^T lam give per-node arrival rates; under
exponential service the network factorizes into independent M/M/1 marginals,
N_i = rho_i/(1 - rho_i). Bench: total mean population vs theory and vs the
single-node M/M/1 sim sanity at node 0.
"""

import numpy as np

from quant_fund.models._queue_synth import GAMMA, MU, jackson_lam, mm1_sim


def bench_jackson_network(seed: int = 4401) -> dict[str, float]:
    lam = jackson_lam()
    rho = lam / MU
    n_theory = float(np.sum(rho / (1.0 - rho)))
    # discrete-event check on node 0 alone (it sees Poisson gamma by Burke)
    n0_sim = mm1_sim(seed, GAMMA[0], MU[0])
    n0_theory = float(rho[0] / (1.0 - rho[0]))
    return {
        "synthetic_jn_n_theory": n_theory,
        "synthetic_jn_node0_theory": n0_theory,
        "synthetic_jn_node0_sim": n0_sim,
        "synthetic_jn_node0_err": abs(n0_sim - n0_theory),
        "synthetic_jn_stable": float(np.max(rho) < 1.0),
        "synthetic_jn_max_rho": float(np.max(rho)),
    }
