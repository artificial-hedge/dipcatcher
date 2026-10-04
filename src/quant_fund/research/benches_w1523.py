"""Wave-1523 bench adapters: grass canon (SYNTHETIC only)."""

from quant_fund.models import (
    bluegrass_qa_studies,
    fescue_qa_studies,
    miscanthus_qa_studies,
    pampas_qa_studies,
    ryegrass_qa_studies,
    switchgrass_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15230


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bluegrass_qa_studies_family(seed: int = _SEED + 0):
    """bluegrass_qa_studies: synthetic correctness bench."""
    return _finite_blob(bluegrass_qa_studies.bench_bluegrass_qa_studies(seed))


def bench_fescue_qa_studies_family(seed: int = _SEED + 1):
    """fescue_qa_studies: synthetic correctness bench."""
    return _finite_blob(fescue_qa_studies.bench_fescue_qa_studies(seed))


def bench_miscanthus_qa_studies_family(seed: int = _SEED + 2):
    """miscanthus_qa_studies: synthetic correctness bench."""
    return _finite_blob(miscanthus_qa_studies.bench_miscanthus_qa_studies(seed))


def bench_pampas_qa_studies_family(seed: int = _SEED + 3):
    """pampas_qa_studies: synthetic correctness bench."""
    return _finite_blob(pampas_qa_studies.bench_pampas_qa_studies(seed))


def bench_ryegrass_qa_studies_family(seed: int = _SEED + 4):
    """ryegrass_qa_studies: synthetic correctness bench."""
    return _finite_blob(ryegrass_qa_studies.bench_ryegrass_qa_studies(seed))


def bench_switchgrass_qa_studies_family(seed: int = _SEED + 5):
    """switchgrass_qa_studies: synthetic correctness bench."""
    return _finite_blob(switchgrass_qa_studies.bench_switchgrass_qa_studies(seed))
