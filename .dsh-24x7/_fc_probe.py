"""Probe: do the catalog verifiers fail closed when their deferred import dies?"""

import sys

from quant_fund.research.catalog import (
    candle_feature_cols_ic_completeness_honesty_errors as f2,
    candle_feature_cols_ic_implies_mean_honesty_errors as f1,
    northset_metrics_required_keys_finite_when_present_honesty_errors as f3,
)

blob = {"family": "candle_order_book", "ic_imbalance_depth": 0.1, "n_scored": 5}
print("healthy f1:", f1(blob))
print("healthy f2:", f2(blob))
print("healthy f3:", f3({"mid": 1.0}))

# Simulate the dependency being unavailable (None in sys.modules -> ImportError).
sys.modules["quant_fund.microstructure.bench"] = None
sys.modules["quant_fund.microstructure.book_metrics"] = None

print("BROKEN f1:", f1(blob))
print("BROKEN f2:", f2(blob))
print("BROKEN f3:", f3({"mid": 1.0}))
