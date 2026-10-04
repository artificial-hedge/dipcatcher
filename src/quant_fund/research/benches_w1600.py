"""Wave-1600 bench adapters: mollusk canon (SYNTHETIC only)."""

from quant_fund.models import (
    abalone_qa_studies,
    chiton_qa_studies,
    cockle_qa_studies,
    cowrie_qa_studies,
    limpet_qa_studies,
    periwinkle_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16000


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_abalone_qa_studies_family(seed: int = _SEED + 0):
    """abalone_qa_studies: synthetic correctness bench."""
    return _finite_blob(abalone_qa_studies.bench_abalone_qa_studies(seed))


def bench_chiton_qa_studies_family(seed: int = _SEED + 1):
    """chiton_qa_studies: synthetic correctness bench."""
    return _finite_blob(chiton_qa_studies.bench_chiton_qa_studies(seed))


def bench_cockle_qa_studies_family(seed: int = _SEED + 2):
    """cockle_qa_studies: synthetic correctness bench."""
    return _finite_blob(cockle_qa_studies.bench_cockle_qa_studies(seed))


def bench_cowrie_qa_studies_family(seed: int = _SEED + 3):
    """cowrie_qa_studies: synthetic correctness bench."""
    return _finite_blob(cowrie_qa_studies.bench_cowrie_qa_studies(seed))


def bench_limpet_qa_studies_family(seed: int = _SEED + 4):
    """limpet_qa_studies: synthetic correctness bench."""
    return _finite_blob(limpet_qa_studies.bench_limpet_qa_studies(seed))


def bench_periwinkle_qa_studies_family(seed: int = _SEED + 5):
    """periwinkle_qa_studies: synthetic correctness bench."""
    return _finite_blob(periwinkle_qa_studies.bench_periwinkle_qa_studies(seed))
