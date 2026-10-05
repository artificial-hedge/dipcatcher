"""Wave-1853 bench adapters: kushite-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    amesemi_qa_studies,
    apedemak_qa_studies,
    aresnuphis_qa_studies,
    dedun_qa_studies,
    sabios_qa_studies,
    sebiumeker_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18530


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_amesemi_qa_studies_family(seed: int = _SEED + 0):
    """amesemi_qa_studies: synthetic correctness bench."""
    return _finite_blob(amesemi_qa_studies.bench_amesemi_qa_studies(seed))


def bench_apedemak_qa_studies_family(seed: int = _SEED + 1):
    """apedemak_qa_studies: synthetic correctness bench."""
    return _finite_blob(apedemak_qa_studies.bench_apedemak_qa_studies(seed))


def bench_aresnuphis_qa_studies_family(seed: int = _SEED + 2):
    """aresnuphis_qa_studies: synthetic correctness bench."""
    return _finite_blob(aresnuphis_qa_studies.bench_aresnuphis_qa_studies(seed))


def bench_dedun_qa_studies_family(seed: int = _SEED + 3):
    """dedun_qa_studies: synthetic correctness bench."""
    return _finite_blob(dedun_qa_studies.bench_dedun_qa_studies(seed))


def bench_sabios_qa_studies_family(seed: int = _SEED + 4):
    """sabios_qa_studies: synthetic correctness bench."""
    return _finite_blob(sabios_qa_studies.bench_sabios_qa_studies(seed))


def bench_sebiumeker_qa_studies_family(seed: int = _SEED + 5):
    """sebiumeker_qa_studies: synthetic correctness bench."""
    return _finite_blob(sebiumeker_qa_studies.bench_sebiumeker_qa_studies(seed))
