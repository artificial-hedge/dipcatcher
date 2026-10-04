"""Wave-1677 bench adapters: greco-roman canon (SYNTHETIC only)."""

from quant_fund.models import (
    antheia_qa_studies,
    aurae_qa_studies,
    camenae_qa_studies,
    fauns_qa_studies,
    limoniad_qa_studies,
    numina_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16770


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_antheia_qa_studies_family(seed: int = _SEED + 0):
    """antheia_qa_studies: synthetic correctness bench."""
    return _finite_blob(antheia_qa_studies.bench_antheia_qa_studies(seed))


def bench_aurae_qa_studies_family(seed: int = _SEED + 1):
    """aurae_qa_studies: synthetic correctness bench."""
    return _finite_blob(aurae_qa_studies.bench_aurae_qa_studies(seed))


def bench_camenae_qa_studies_family(seed: int = _SEED + 2):
    """camenae_qa_studies: synthetic correctness bench."""
    return _finite_blob(camenae_qa_studies.bench_camenae_qa_studies(seed))


def bench_fauns_qa_studies_family(seed: int = _SEED + 3):
    """fauns_qa_studies: synthetic correctness bench."""
    return _finite_blob(fauns_qa_studies.bench_fauns_qa_studies(seed))


def bench_limoniad_qa_studies_family(seed: int = _SEED + 4):
    """limoniad_qa_studies: synthetic correctness bench."""
    return _finite_blob(limoniad_qa_studies.bench_limoniad_qa_studies(seed))


def bench_numina_qa_studies_family(seed: int = _SEED + 5):
    """numina_qa_studies: synthetic correctness bench."""
    return _finite_blob(numina_qa_studies.bench_numina_qa_studies(seed))
