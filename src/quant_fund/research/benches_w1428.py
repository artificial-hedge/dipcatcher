"""Wave-1428 bench adapters: governance canon (SYNTHETIC only)."""

from quant_fund.models import (
    agency_qa_studies,
    bureau_qa_studies,
    cabinet_qa_studies,
    election_qa_studies,
    government_qa_studies,
    ministry_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14280


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_agency_qa_studies_family(seed: int = _SEED + 0):
    """agency_qa_studies: synthetic correctness bench."""
    return _finite_blob(agency_qa_studies.bench_agency_qa_studies(seed))


def bench_bureau_qa_studies_family(seed: int = _SEED + 1):
    """bureau_qa_studies: synthetic correctness bench."""
    return _finite_blob(bureau_qa_studies.bench_bureau_qa_studies(seed))


def bench_cabinet_qa_studies_family(seed: int = _SEED + 2):
    """cabinet_qa_studies: synthetic correctness bench."""
    return _finite_blob(cabinet_qa_studies.bench_cabinet_qa_studies(seed))


def bench_election_qa_studies_family(seed: int = _SEED + 3):
    """election_qa_studies: synthetic correctness bench."""
    return _finite_blob(election_qa_studies.bench_election_qa_studies(seed))


def bench_government_qa_studies_family(seed: int = _SEED + 4):
    """government_qa_studies: synthetic correctness bench."""
    return _finite_blob(government_qa_studies.bench_government_qa_studies(seed))


def bench_ministry_qa_studies_family(seed: int = _SEED + 5):
    """ministry_qa_studies: synthetic correctness bench."""
    return _finite_blob(ministry_qa_studies.bench_ministry_qa_studies(seed))
