"""Vickrey (second-price) auction — revenue equivalence check (SYNTHETIC).

With truthful bidding the winner pays the second-highest value; revenue
equivalence says E[revenue] = E[max of order statistic 2] which for
U[0,1] with n bidders equals (n-1)/(n+1). Bench: simulated second-price
revenue vs theory and vs first-price BNE revenue (equivalence).
"""

import numpy as np

from quant_fund.models._auc_synth import N_BID, iid_values, uniform_bne_bid


def bench_vickrey_auction(seed: int = 4501) -> dict[str, float]:
    v = iid_values(seed)
    order = np.sort(v, axis=1)
    rev_sp = order[:, -2].mean()  # second-highest value
    bids = uniform_bne_bid(v)
    rev_fp = bids.max(axis=1).mean()  # winner pays own bid
    theory = (N_BID - 1.0) / (N_BID + 1.0)
    return {
        "synthetic_vickrey_rev": float(rev_sp),
        "synthetic_vickrey_theory": theory,
        "synthetic_vickrey_err": abs(float(rev_sp) - theory),
        "synthetic_vickrey_fp_rev": float(rev_fp),
        "synthetic_vickrey_equiv_gap": abs(float(rev_sp) - float(rev_fp)),
    }
