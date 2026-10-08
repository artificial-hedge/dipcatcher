"""Coarse CRR binomial tree on the shared American-put contract (SYNTHETIC).

A 200-step tree measured against the 2000-step reference — the honest
discretization benchmark for tree methods in this wave.
"""

from quant_fund.models._amopt_synth import crr_price, european_put


def bench_crr_tree(seed: int = 4103) -> dict[str, float]:
    del seed
    ref, _ = crr_price(2000)
    price, _ = crr_price(200)
    eur = european_put()
    return {
        "synthetic_crr_price": price,
        "synthetic_crr_err": abs(price - ref),
        "synthetic_crr_ref": ref,
        "synthetic_crr_eur_floor_err": abs(eur - ref),
    }
