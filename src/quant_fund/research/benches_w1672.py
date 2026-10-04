"""Wave-1672 bench adapters: slavic-domestic canon (SYNTHETIC only)."""

from quant_fund.models import (
    domovoi_qa_studies,
    kikimora_qa_studies,
    leshy_qa_studies,
    polevik_qa_studies,
    rusalka_qa_studies,
    vodianoi_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16720


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_domovoi_qa_studies_family(seed: int = _SEED + 0):
    """domovoi_qa_studies: synthetic correctness bench."""
    return _finite_blob(domovoi_qa_studies.bench_domovoi_qa_studies(seed))


def bench_kikimora_qa_studies_family(seed: int = _SEED + 1):
    """kikimora_qa_studies: synthetic correctness bench."""
    return _finite_blob(kikimora_qa_studies.bench_kikimora_qa_studies(seed))


def bench_leshy_qa_studies_family(seed: int = _SEED + 2):
    """leshy_qa_studies: synthetic correctness bench."""
    return _finite_blob(leshy_qa_studies.bench_leshy_qa_studies(seed))


def bench_polevik_qa_studies_family(seed: int = _SEED + 3):
    """polevik_qa_studies: synthetic correctness bench."""
    return _finite_blob(polevik_qa_studies.bench_polevik_qa_studies(seed))


def bench_rusalka_qa_studies_family(seed: int = _SEED + 4):
    """rusalka_qa_studies: synthetic correctness bench."""
    return _finite_blob(rusalka_qa_studies.bench_rusalka_qa_studies(seed))


def bench_vodianoi_qa_studies_family(seed: int = _SEED + 5):
    """vodianoi_qa_studies: synthetic correctness bench."""
    return _finite_blob(vodianoi_qa_studies.bench_vodianoi_qa_studies(seed))
