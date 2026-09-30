"""Empirically verify the CKS buy/sell decomposition against the OFI identity.

Cont-Kukanov-Stoikov (2014) define order-flow imbalance as
    OFI_n = e^B_n - e^A_n
    e^B_n = 1[P^B_n >= P^B_{n-1}] q^B_n   - 1[P^B_n <= P^B_{n-1}] q^B_{n-1}
    e^A_n = 1[P^A_n <= P^A_{n-1}] q^A_n   - 1[P^A_n >= P^A_{n-1}] q^A_{n-1}
so the buy-side leg is  1[bid up] q^B_n + 1[ask up] q^A_{n-1}
and the sell-side leg is 1[bid dn] q^B_{n-1} + 1[ask dn] q^A_n,
and buy - sell must equal OFI exactly.
"""

import numpy as np
import polars as pl

ROWS = [
    (10.00, 11.0, 100.0, 100.0),
    (10.10, 11.0, 200.0, 100.0),
    (10.05, 10.9, 150.0, 300.0),
    (10.20, 11.2, 400.0, 50.0),
    (10.20, 11.3, 100.0, 60.0),
]
TEST_EXPECTED_TOX = [0.5, 1.0 / 3.0, 7.0 / 9.0, 9.0 / 14.0]

frame = pl.DataFrame(
    {
        "best_bid": [r[0] for r in ROWS],
        "best_ask": [r[1] for r in ROWS],
        "top_bid_size": [r[2] for r in ROWS],
        "top_ask_size": [r[3] for r in ROWS],
    }
).with_columns(
    pl.col("best_bid").shift(1).alias("_pb"),
    pl.col("best_ask").shift(1).alias("_pa"),
    pl.col("top_bid_size").shift(1).alias("_ptb"),
    pl.col("top_ask_size").shift(1).alias("_pta"),
)

# Implementation's legs (estimators.py vpin_proxy).
impl_buy = np.where(
    frame["best_bid"].to_numpy() >= frame["_pb"].to_numpy(), frame["top_bid_size"].to_numpy(), 0.0
) + np.where(frame["best_ask"].to_numpy() >= frame["_pa"].to_numpy(), frame["_pta"].to_numpy(), 0.0)
impl_sell = np.where(
    frame["best_bid"].to_numpy() <= frame["_pb"].to_numpy(), frame["_ptb"].to_numpy(), 0.0
) + np.where(frame["best_ask"].to_numpy() <= frame["_pa"].to_numpy(), frame["top_ask_size"].to_numpy(), 0.0)

# Textbook OFI computed independently from the definition.
bid, ask = frame["best_bid"].to_numpy(), frame["best_ask"].to_numpy()
qb, qa = frame["top_bid_size"].to_numpy(), frame["top_ask_size"].to_numpy()
pb, pa = frame["_pb"].to_numpy(), frame["_pa"].to_numpy()
ptb, pta = frame["_ptb"].to_numpy(), frame["_pta"].to_numpy()
with np.errstate(invalid="ignore"):
    e_b = np.where(bid >= pb, qb, 0.0) - np.where(bid <= pb, ptb, 0.0)
    e_a = np.where(ask <= pa, qa, 0.0) - np.where(ask >= pa, pta, 0.0)
ofi = e_b - e_a

print("idx | impl_buy impl_sell tox_impl | OFI  buy-sell | tox_expected(test)")
for i in range(len(ROWS)):
    vol = impl_buy[i] + impl_sell[i]
    tox = abs(impl_buy[i] - impl_sell[i]) / vol if vol > 0 else float("nan")
    ident = impl_buy[i] - impl_sell[i]
    exp = TEST_EXPECTED_TOX[i - 1] if i >= 1 else float("nan")
    print(f"{i:3d} | {impl_buy[i]:8.1f} {impl_sell[i]:9.1f} {tox:8.4f} | {ofi[i]:5.1f} {ident:10.1f} | {exp:.4f}")

with np.errstate(invalid="ignore"):
    match_ofi = np.allclose(impl_buy[1:] - impl_sell[1:], ofi[1:])
print(f"\nbuy-sell == OFI exactly (rows 1..n): {match_ofi}")
vols = impl_buy + impl_sell
tox = np.where(vols > 0, np.abs(impl_buy - impl_sell) / np.maximum(vols, 1e-12), np.nan)
print(f"impl tox[1:]   : {np.round(tox[1:], 6).tolist()}")
print(f"test tox       : {np.round(TEST_EXPECTED_TOX, 6).tolist()}")
print(f"impl mean(t0:t2) window at idx2 = {np.mean(tox[1:3]):.6f}   (test expects {np.mean(TEST_EXPECTED_TOX[0:2]):.6f})")
