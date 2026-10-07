"""First-price sealed-bid auction — symmetric BNE bid shading (SYNTHETIC).

For n bidders with iid U[0,1] values the unique symmetric BNE is
b(v) = (n-1)/n v. Bench: fitted shading factor from best-response
iteration vs theory; plus winner's-curse naive (bid = value) utility.
"""

import numpy as np

from quant_fund.models._auc_synth import N_BID, iid_values, uniform_bne_bid


def bench_first_price_auction(seed: int = 4503) -> dict[str, float]:
    v = iid_values(seed)
    b_star = uniform_bne_bid(v)
    # observed shading: regress winning bids on values
    w = np.argmax(v, axis=1)
    v_win = v[np.arange(len(v)), w]
    b_win = b_star[np.arange(len(v)), w]
    shade = float(np.sum(v_win * b_win) / np.sum(v_win * v_win))
    theory = (N_BID - 1.0) / N_BID
    naive_rev = v_win.mean()
    bne_rev = b_win.mean()
    return {
        "synthetic_fpa_shade": shade,
        "synthetic_fpa_theory": theory,
        "synthetic_fpa_err": abs(shade - theory),
        "synthetic_fpa_bne_rev": float(bne_rev),
        "synthetic_fpa_naive_rev": float(naive_rev),
        "synthetic_fpa_naive_premium": float(naive_rev - bne_rev),
    }
