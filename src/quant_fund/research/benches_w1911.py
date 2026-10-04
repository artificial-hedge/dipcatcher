"""Wave-1911 bench adapters: persian-spirit canon (SYNTHETIC only)."""

from quant_fund.models import (
    divsalar_qa_studies,
    khrafstra_qa_studies,
    leshenka_qa_studies,
    pari_vatra_qa_studies,
    srosh_demon_qa_studies,
    urvan_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19110


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_divsalar_qa_studies_family(seed: int = _SEED + 0):
    """divsalar_qa_studies: synthetic correctness bench."""
    return _finite_blob(divsalar_qa_studies.bench_divsalar_qa_studies(seed))


def bench_khrafstra_qa_studies_family(seed: int = _SEED + 1):
    """khrafstra_qa_studies: synthetic correctness bench."""
    return _finite_blob(khrafstra_qa_studies.bench_khrafstra_qa_studies(seed))


def bench_leshenka_qa_studies_family(seed: int = _SEED + 2):
    """leshenka_qa_studies: synthetic correctness bench."""
    return _finite_blob(leshenka_qa_studies.bench_leshenka_qa_studies(seed))


def bench_pari_vatra_qa_studies_family(seed: int = _SEED + 3):
    """pari_vatra_qa_studies: synthetic correctness bench."""
    return _finite_blob(pari_vatra_qa_studies.bench_pari_vatra_qa_studies(seed))


def bench_srosh_demon_qa_studies_family(seed: int = _SEED + 4):
    """srosh_demon_qa_studies: synthetic correctness bench."""
    return _finite_blob(srosh_demon_qa_studies.bench_srosh_demon_qa_studies(seed))


def bench_urvan_qa_studies_family(seed: int = _SEED + 5):
    """urvan_qa_studies: synthetic correctness bench."""
    return _finite_blob(urvan_qa_studies.bench_urvan_qa_studies(seed))
