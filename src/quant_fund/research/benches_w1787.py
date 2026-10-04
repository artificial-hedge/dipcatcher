"""Wave-1787 bench adapters: norse-myth-11 canon (SYNTHETIC only)."""

from quant_fund.models import (
    eir_qa_studies,
    heimdal_qa_studies,
    norns_qa_studies,
    odin_qa_studies,
    thor_qa_studies,
    valkyrie_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17870


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_eir_qa_studies_family(seed: int = _SEED + 0):
    """eir_qa_studies: synthetic correctness bench."""
    return _finite_blob(eir_qa_studies.bench_eir_qa_studies(seed))


def bench_heimdal_qa_studies_family(seed: int = _SEED + 1):
    """heimdal_qa_studies: synthetic correctness bench."""
    return _finite_blob(heimdal_qa_studies.bench_heimdal_qa_studies(seed))


def bench_norns_qa_studies_family(seed: int = _SEED + 2):
    """norns_qa_studies: synthetic correctness bench."""
    return _finite_blob(norns_qa_studies.bench_norns_qa_studies(seed))


def bench_odin_qa_studies_family(seed: int = _SEED + 3):
    """odin_qa_studies: synthetic correctness bench."""
    return _finite_blob(odin_qa_studies.bench_odin_qa_studies(seed))


def bench_thor_qa_studies_family(seed: int = _SEED + 4):
    """thor_qa_studies: synthetic correctness bench."""
    return _finite_blob(thor_qa_studies.bench_thor_qa_studies(seed))


def bench_valkyrie_qa_studies_family(seed: int = _SEED + 5):
    """valkyrie_qa_studies: synthetic correctness bench."""
    return _finite_blob(valkyrie_qa_studies.bench_valkyrie_qa_studies(seed))
