"""Ascending-clock (English) auction with dropout tracking (SYNTHETIC).

Price rises until one bidder remains; each bidder drops at their value.
The clearing price equals the second-highest value — same allocation and
revenue as Vickrey (weak dominance). Bench: simulated clock revenue vs
Vickrey revenue and the price path's dropout pattern.
"""

import numpy as np

from quant_fund.models._auc_synth import N_BID, iid_values


def bench_ascending_clock(seed: int = 4507) -> dict[str, float]:
    v = iid_values(seed)
    order = np.sort(v, axis=1)
    clock_price = order[:, -2]  # last dropout clears
    rev_vick = order[:, -2]
    # dropout spread: price at which half the field dropped
    median_drop = order[:, -2] - order[:, -3]
    eff_gap = order[:, -1] - order[:, -2]  # winner surplus
    return {
        "synthetic_clock_rev": float(clock_price.mean()),
        "synthetic_clock_vickrey_gap": float(np.abs(clock_price - rev_vick).mean()),
        "synthetic_clock_median_drop": float(median_drop.mean()),
        "synthetic_clock_winner_surplus": float(eff_gap.mean()),
        "synthetic_clock_theory_gap": abs(
            float(clock_price.mean()) - (N_BID - 1.0) / (N_BID + 1.0)
        ),
    }
