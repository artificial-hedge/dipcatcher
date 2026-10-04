"""Wave-1560 bench adapters: reef-fish canon (SYNTHETIC only)."""

from quant_fund.models import (
    butterflyfish_qa_studies,
    damselfish_qa_studies,
    grouper_qa_studies,
    parrotfish_qa_studies,
    snapper_qa_studies,
    wrasse_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15600


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_butterflyfish_qa_studies_family(seed: int = _SEED + 0):
    """butterflyfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(butterflyfish_qa_studies.bench_butterflyfish_qa_studies(seed))


def bench_damselfish_qa_studies_family(seed: int = _SEED + 1):
    """damselfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(damselfish_qa_studies.bench_damselfish_qa_studies(seed))


def bench_grouper_qa_studies_family(seed: int = _SEED + 2):
    """grouper_qa_studies: synthetic correctness bench."""
    return _finite_blob(grouper_qa_studies.bench_grouper_qa_studies(seed))


def bench_parrotfish_qa_studies_family(seed: int = _SEED + 3):
    """parrotfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(parrotfish_qa_studies.bench_parrotfish_qa_studies(seed))


def bench_snapper_qa_studies_family(seed: int = _SEED + 4):
    """snapper_qa_studies: synthetic correctness bench."""
    return _finite_blob(snapper_qa_studies.bench_snapper_qa_studies(seed))


def bench_wrasse_qa_studies_family(seed: int = _SEED + 5):
    """wrasse_qa_studies: synthetic correctness bench."""
    return _finite_blob(wrasse_qa_studies.bench_wrasse_qa_studies(seed))
