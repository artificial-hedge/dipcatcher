"""Wave-200 adapter tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research import benches_w208 as b
from quant_fund.research.catalog.registry import OPTIONAL_BENCHMARK_FAMILIES

FAMILIES = [
    "cwt_ridge",
    "cepstrum_pitch",
    "mvdr_beamformer",
    "hilbert_instant",
    "lpc_formant",
    "goertzel_detect",
]


@pytest.mark.parametrize("name", FAMILIES)
def test_family_registered(name: str) -> None:
    assert name in OPTIONAL_BENCHMARK_FAMILIES


@pytest.mark.parametrize("name", FAMILIES)
def test_family_runs(name: str) -> None:
    out = getattr(b, f"bench_{name}_family")()
    assert out
    for k, v in out.items():
        assert np.isfinite(v), k
        assert not any(bad in k.lower() for bad in ("sharpe", "sortino", "calmar", "pnl", "nav"))
