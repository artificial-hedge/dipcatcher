"""Wave-1545 bench adapters: parrot canon (SYNTHETIC only)."""

from quant_fund.models import (
    amazon_qa_studies,
    cockatoo_qa_studies,
    conure_qa_studies,
    kakapo_qa_studies,
    kea_qa_studies,
    lorikeet_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15450


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_amazon_qa_studies_family(seed: int = _SEED + 0):
    """amazon_qa_studies: synthetic correctness bench."""
    return _finite_blob(amazon_qa_studies.bench_amazon_qa_studies(seed))


def bench_cockatoo_qa_studies_family(seed: int = _SEED + 1):
    """cockatoo_qa_studies: synthetic correctness bench."""
    return _finite_blob(cockatoo_qa_studies.bench_cockatoo_qa_studies(seed))


def bench_conure_qa_studies_family(seed: int = _SEED + 2):
    """conure_qa_studies: synthetic correctness bench."""
    return _finite_blob(conure_qa_studies.bench_conure_qa_studies(seed))


def bench_kakapo_qa_studies_family(seed: int = _SEED + 3):
    """kakapo_qa_studies: synthetic correctness bench."""
    return _finite_blob(kakapo_qa_studies.bench_kakapo_qa_studies(seed))


def bench_kea_qa_studies_family(seed: int = _SEED + 4):
    """kea_qa_studies: synthetic correctness bench."""
    return _finite_blob(kea_qa_studies.bench_kea_qa_studies(seed))


def bench_lorikeet_qa_studies_family(seed: int = _SEED + 5):
    """lorikeet_qa_studies: synthetic correctness bench."""
    return _finite_blob(lorikeet_qa_studies.bench_lorikeet_qa_studies(seed))
