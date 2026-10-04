"""Wave-1253 immune-mediators bench adapters (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.antibody_studies import bench_antibody_studies
from quant_fund.models.chemokine_studies import bench_chemokine_studies
from quant_fund.models.complement_studies import bench_complement_studies
from quant_fund.models.cytokine_studies import bench_cytokine_studies
from quant_fund.models.interferon_studies import bench_interferon_studies
from quant_fund.models.lymphocyte_studies import bench_lymphocyte_studies

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(blob: dict) -> dict:
    out = {}
    for k, v in blob.items():
        assert k not in _FORBIDDEN
        assert k.startswith("synthetic_")
        assert isinstance(v, float) and 0.0 <= v <= 1.0
        out[k] = v
    return out


def _floats(blob: dict) -> dict:
    return _finite_blob(blob)


def bench_cytokine_studies_family(seed: int = _SEED + 64100) -> dict:
    return _floats(bench_cytokine_studies(seed))


def bench_chemokine_studies_family(seed: int = _SEED + 64101) -> dict:
    return _floats(bench_chemokine_studies(seed))


def bench_interferon_studies_family(seed: int = _SEED + 64102) -> dict:
    return _floats(bench_interferon_studies(seed))


def bench_complement_studies_family(seed: int = _SEED + 64103) -> dict:
    return _floats(bench_complement_studies(seed))


def bench_antibody_studies_family(seed: int = _SEED + 64104) -> dict:
    return _floats(bench_antibody_studies(seed))


def bench_lymphocyte_studies_family(seed: int = _SEED + 64105) -> dict:
    return _floats(bench_lymphocyte_studies(seed))
