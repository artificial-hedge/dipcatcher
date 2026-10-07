"""Realized variance estimators (scaffolding).

Currently exposes :func:`quant_fund.realized.hf_rv.compute_hf_rv` (5-min
realized variance scaffolding — see :mod:`docs.HF_RV_DESIGN`).
"""

from __future__ import annotations

from quant_fund.realized.hf_rv import HFRVResult, compute_hf_rv

__all__ = ["HFRVResult", "compute_hf_rv"]
