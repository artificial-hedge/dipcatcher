"""Wave-1819 bench adapters: norse-myth-15 canon (SYNTHETIC only)."""

from quant_fund.models import (
    bragi2_qa_studies,
    forseti2_qa_studies,
    heimdall2_qa_studies,
    norna2_qa_studies,
    ve2_qa_studies,
    vili2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18190


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bragi2_qa_studies_family(seed: int = _SEED + 0):
    """bragi2_qa_studies: synthetic correctness bench."""
    return _finite_blob(bragi2_qa_studies.bench_bragi2_qa_studies(seed))


def bench_forseti2_qa_studies_family(seed: int = _SEED + 1):
    """forseti2_qa_studies: synthetic correctness bench."""
    return _finite_blob(forseti2_qa_studies.bench_forseti2_qa_studies(seed))


def bench_heimdall2_qa_studies_family(seed: int = _SEED + 2):
    """heimdall2_qa_studies: synthetic correctness bench."""
    return _finite_blob(heimdall2_qa_studies.bench_heimdall2_qa_studies(seed))


def bench_norna2_qa_studies_family(seed: int = _SEED + 3):
    """norna2_qa_studies: synthetic correctness bench."""
    return _finite_blob(norna2_qa_studies.bench_norna2_qa_studies(seed))


def bench_ve2_qa_studies_family(seed: int = _SEED + 4):
    """ve2_qa_studies: synthetic correctness bench."""
    return _finite_blob(ve2_qa_studies.bench_ve2_qa_studies(seed))


def bench_vili2_qa_studies_family(seed: int = _SEED + 5):
    """vili2_qa_studies: synthetic correctness bench."""
    return _finite_blob(vili2_qa_studies.bench_vili2_qa_studies(seed))
