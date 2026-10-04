"""Wave-1895 bench adapters: mesopotamian-demon-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    allatu_qa_studies,
    belili_qa_studies,
    dimme_qa_studies,
    gallu_qa_studies,
    lilu_qa_studies,
    sulak_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18950


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_allatu_qa_studies_family(seed: int = _SEED + 0):
    """allatu_qa_studies: synthetic correctness bench."""
    return _finite_blob(allatu_qa_studies.bench_allatu_qa_studies(seed))


def bench_belili_qa_studies_family(seed: int = _SEED + 1):
    """belili_qa_studies: synthetic correctness bench."""
    return _finite_blob(belili_qa_studies.bench_belili_qa_studies(seed))


def bench_dimme_qa_studies_family(seed: int = _SEED + 2):
    """dimme_qa_studies: synthetic correctness bench."""
    return _finite_blob(dimme_qa_studies.bench_dimme_qa_studies(seed))


def bench_gallu_qa_studies_family(seed: int = _SEED + 3):
    """gallu_qa_studies: synthetic correctness bench."""
    return _finite_blob(gallu_qa_studies.bench_gallu_qa_studies(seed))


def bench_lilu_qa_studies_family(seed: int = _SEED + 4):
    """lilu_qa_studies: synthetic correctness bench."""
    return _finite_blob(lilu_qa_studies.bench_lilu_qa_studies(seed))


def bench_sulak_qa_studies_family(seed: int = _SEED + 5):
    """sulak_qa_studies: synthetic correctness bench."""
    return _finite_blob(sulak_qa_studies.bench_sulak_qa_studies(seed))
