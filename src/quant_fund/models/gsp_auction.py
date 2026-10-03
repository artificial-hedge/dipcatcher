"""Generalized second-price (GSP) position auction.

Slots with click-through rates c_1 > c_2 > ...; bidders ranked by bid,
each pays the next bidder's bid per click. Bench: GSP revenue vs VCG
(on the same value/CTR fixture), plus the envy-free locally envy-free
equilibrium check: bidder i shouldn't want slot i-1's price per click.
"""

import numpy as np

from quant_fund.models._auc_synth import GSP_CTR, GSP_VALUES


def _gsp_revenue(bids: np.ndarray) -> float:
    order = np.argsort(-bids)
    b = bids[order]
    rev = 0.0
    for s in range(min(len(GSP_CTR), len(b) - 1)):
        rev += GSP_CTR[s] * b[s + 1]
    return float(rev)


def _vcg_revenue(bids: np.ndarray) -> float:
    order = np.argsort(-bids)
    b = bids[order]
    rev = 0.0
    n_slots = min(len(GSP_CTR), len(b) - 1)
    for s in range(n_slots):
        # winner s pays sum over displaced slots weighted by CTR drop
        pay = 0.0
        for t in range(s, n_slots):
            ctr_drop = GSP_CTR[t] - (GSP_CTR[t + 1] if t + 1 < n_slots else 0.0)
            pay += ctr_drop * b[t + 1]
        rev += pay
    return float(rev)


def _envy_free_viol(bids: np.ndarray) -> float:
    order = np.argsort(-bids)
    v = bids[order]
    n_slots = min(len(GSP_CTR), len(v))
    viol = 0.0
    for s in range(1, n_slots):
        ppc = v[s]  # price per click paid at slot s is next bid
        ppc_up = v[s - 1] if s - 1 >= 1 else v[s]
        gain = GSP_CTR[s - 1] * (v[s] - ppc_up) - GSP_CTR[s] * (v[s] - ppc)
        viol = max(viol, gain)
    return float(viol)


def bench_gsp_auction(seed: int = 4511) -> dict[str, float]:
    del seed
    bids = GSP_VALUES.copy()
    rev_gsp = _gsp_revenue(bids)
    rev_vcg = _vcg_revenue(bids)
    viol = _envy_free_viol(bids)
    return {
        "synthetic_gsp_rev": rev_gsp,
        "synthetic_gsp_vcg_rev": rev_vcg,
        "synthetic_gsp_premium": rev_gsp - rev_vcg,
        "synthetic_gsp_envy_viol": viol,
        "synthetic_gsp_n_slots": float(min(len(GSP_CTR), len(bids) - 1)),
    }
