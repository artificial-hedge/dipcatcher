"""Wave-1283 bench adapters: agent-memory canon (SYNTHETIC only)."""

from quant_fund.models import (
    context_compression_studies,
    episodic_memory_studies,
    memory_bank_studies,
    retrieval_memory_studies,
    semantic_memory_studies,
    working_memory_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12830


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_context_compression_studies_family(seed: int = _SEED + 0):
    """context_compression_studies: synthetic correctness bench."""
    return _finite_blob(context_compression_studies.bench_context_compression_studies(seed))


def bench_episodic_memory_studies_family(seed: int = _SEED + 1):
    """episodic_memory_studies: synthetic correctness bench."""
    return _finite_blob(episodic_memory_studies.bench_episodic_memory_studies(seed))


def bench_memory_bank_studies_family(seed: int = _SEED + 2):
    """memory_bank_studies: synthetic correctness bench."""
    return _finite_blob(memory_bank_studies.bench_memory_bank_studies(seed))


def bench_retrieval_memory_studies_family(seed: int = _SEED + 3):
    """retrieval_memory_studies: synthetic correctness bench."""
    return _finite_blob(retrieval_memory_studies.bench_retrieval_memory_studies(seed))


def bench_semantic_memory_studies_family(seed: int = _SEED + 4):
    """semantic_memory_studies: synthetic correctness bench."""
    return _finite_blob(semantic_memory_studies.bench_semantic_memory_studies(seed))


def bench_working_memory_studies_family(seed: int = _SEED + 5):
    """working_memory_studies: synthetic correctness bench."""
    return _finite_blob(working_memory_studies.bench_working_memory_studies(seed))
