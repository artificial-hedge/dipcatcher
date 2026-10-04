"""Wave-1528 bench adapters: mineral canon (SYNTHETIC only)."""

from quant_fund.models import (
    calcite_qa_studies,
    feldspar_qa_studies,
    fluorite_qa_studies,
    gypsum_qa_studies,
    olivine_qa_studies,
    quartz_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15280


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_calcite_qa_studies_family(seed: int = _SEED + 0):
    """calcite_qa_studies: synthetic correctness bench."""
    return _finite_blob(calcite_qa_studies.bench_calcite_qa_studies(seed))


def bench_feldspar_qa_studies_family(seed: int = _SEED + 1):
    """feldspar_qa_studies: synthetic correctness bench."""
    return _finite_blob(feldspar_qa_studies.bench_feldspar_qa_studies(seed))


def bench_fluorite_qa_studies_family(seed: int = _SEED + 2):
    """fluorite_qa_studies: synthetic correctness bench."""
    return _finite_blob(fluorite_qa_studies.bench_fluorite_qa_studies(seed))


def bench_gypsum_qa_studies_family(seed: int = _SEED + 3):
    """gypsum_qa_studies: synthetic correctness bench."""
    return _finite_blob(gypsum_qa_studies.bench_gypsum_qa_studies(seed))


def bench_olivine_qa_studies_family(seed: int = _SEED + 4):
    """olivine_qa_studies: synthetic correctness bench."""
    return _finite_blob(olivine_qa_studies.bench_olivine_qa_studies(seed))


def bench_quartz_qa_studies_family(seed: int = _SEED + 5):
    """quartz_qa_studies: synthetic correctness bench."""
    return _finite_blob(quartz_qa_studies.bench_quartz_qa_studies(seed))
