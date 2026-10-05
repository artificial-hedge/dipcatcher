"""Wave-1821 bench adapters: japanese-myth-9 canon (SYNTHETIC only)."""

from quant_fund.models import (
    benzaiten2_qa_studies,
    daikoku2_qa_studies,
    ebisu2_qa_studies,
    fukurokuju2_qa_studies,
    hotei2_qa_studies,
    juroujin2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18210


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_benzaiten2_qa_studies_family(seed: int = _SEED + 0):
    """benzaiten2_qa_studies: synthetic correctness bench."""
    return _finite_blob(benzaiten2_qa_studies.bench_benzaiten2_qa_studies(seed))


def bench_daikoku2_qa_studies_family(seed: int = _SEED + 1):
    """daikoku2_qa_studies: synthetic correctness bench."""
    return _finite_blob(daikoku2_qa_studies.bench_daikoku2_qa_studies(seed))


def bench_ebisu2_qa_studies_family(seed: int = _SEED + 2):
    """ebisu2_qa_studies: synthetic correctness bench."""
    return _finite_blob(ebisu2_qa_studies.bench_ebisu2_qa_studies(seed))


def bench_fukurokuju2_qa_studies_family(seed: int = _SEED + 3):
    """fukurokuju2_qa_studies: synthetic correctness bench."""
    return _finite_blob(fukurokuju2_qa_studies.bench_fukurokuju2_qa_studies(seed))


def bench_hotei2_qa_studies_family(seed: int = _SEED + 4):
    """hotei2_qa_studies: synthetic correctness bench."""
    return _finite_blob(hotei2_qa_studies.bench_hotei2_qa_studies(seed))


def bench_juroujin2_qa_studies_family(seed: int = _SEED + 5):
    """juroujin2_qa_studies: synthetic correctness bench."""
    return _finite_blob(juroujin2_qa_studies.bench_juroujin2_qa_studies(seed))
