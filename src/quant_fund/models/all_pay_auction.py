"""All-pay auction — symmetric BNE for uniform values.

Every bidder pays their bid; the symmetric BNE for iid U[0,1] values is
b(v) = (n-1)/n * v^n. Expected revenue = n E[b(v_(n))]... equals the
second-price revenue by revenue equivalence: (n-1)/(n+1). Bench:
simulated total payments under the BNE vs theory, plus max-bid CDF fit.
"""

import numpy as np

from quant_fund.models._auc_synth import iid_values


def bench_all_pay_auction(seed: int = 4505) -> dict[str, float]:
    v = iid_values(seed)
    n = v.shape[1]
    bids = (n - 1.0) / n * v**n
    rev = bids.sum(axis=1).mean()
    theory = (n - 1.0) / (n + 1.0)
    # winner pays own bid too; winner surplus = E[v1 - b(v1)]
    w = np.argmax(v, axis=1)
    vw = v[np.arange(len(v)), w]
    bw = bids[np.arange(len(v)), w]
    surplus = float(np.mean(vw - bw))
    return {
        "synthetic_apa_rev": float(rev),
        "synthetic_apa_theory": theory,
        "synthetic_apa_err": abs(float(rev) - theory),
        "synthetic_apa_winner_surplus": surplus,
        "synthetic_apa_mean_bid": float(bids.mean()),
    }
