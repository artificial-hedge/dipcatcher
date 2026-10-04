"""Wave-1796 bench adapters: norse-myth-13 canon (SYNTHETIC only)."""

from quant_fund.models import (
    baldr2_qa_studies,
    forseti2_qa_studies,
    hermodr_qa_studies,
    idun2_qa_studies,
    nanna3_qa_studies,
    ullr2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17960


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_baldr2_qa_studies_family(seed: int = _SEED + 0):
    """baldr2_qa_studies: synthetic correctness bench."""
    return _finite_blob(baldr2_qa_studies.bench_baldr2_qa_studies(seed))


def bench_forseti2_qa_studies_family(seed: int = _SEED + 1):
    """forseti2_qa_studies: synthetic correctness bench."""
    return _finite_blob(forseti2_qa_studies.bench_forseti2_qa_studies(seed))


def bench_hermodr_qa_studies_family(seed: int = _SEED + 2):
    """hermodr_qa_studies: synthetic correctness bench."""
    return _finite_blob(hermodr_qa_studies.bench_hermodr_qa_studies(seed))


def bench_idun2_qa_studies_family(seed: int = _SEED + 3):
    """idun2_qa_studies: synthetic correctness bench."""
    return _finite_blob(idun2_qa_studies.bench_idun2_qa_studies(seed))


def bench_nanna3_qa_studies_family(seed: int = _SEED + 4):
    """nanna3_qa_studies: synthetic correctness bench."""
    return _finite_blob(nanna3_qa_studies.bench_nanna3_qa_studies(seed))


def bench_ullr2_qa_studies_family(seed: int = _SEED + 5):
    """ullr2_qa_studies: synthetic correctness bench."""
    return _finite_blob(ullr2_qa_studies.bench_ullr2_qa_studies(seed))
