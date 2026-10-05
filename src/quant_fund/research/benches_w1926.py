"""Wave-1926 bench adapters: caucasus-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    albasty_qa_studies,
    albi_qa_studies,
    chinka_qa_studies,
    furts_qa_studies,
    gorgogosh_qa_studies,
    rukhi_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19260


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_albasty_qa_studies_family(seed: int = _SEED + 0):
    """albasty_qa_studies: synthetic correctness bench."""
    return _finite_blob(albasty_qa_studies.bench_albasty_qa_studies(seed))


def bench_albi_qa_studies_family(seed: int = _SEED + 1):
    """albi_qa_studies: synthetic correctness bench."""
    return _finite_blob(albi_qa_studies.bench_albi_qa_studies(seed))


def bench_chinka_qa_studies_family(seed: int = _SEED + 2):
    """chinka_qa_studies: synthetic correctness bench."""
    return _finite_blob(chinka_qa_studies.bench_chinka_qa_studies(seed))


def bench_furts_qa_studies_family(seed: int = _SEED + 3):
    """furts_qa_studies: synthetic correctness bench."""
    return _finite_blob(furts_qa_studies.bench_furts_qa_studies(seed))


def bench_gorgogosh_qa_studies_family(seed: int = _SEED + 4):
    """gorgogosh_qa_studies: synthetic correctness bench."""
    return _finite_blob(gorgogosh_qa_studies.bench_gorgogosh_qa_studies(seed))


def bench_rukhi_qa_studies_family(seed: int = _SEED + 5):
    """rukhi_qa_studies: synthetic correctness bench."""
    return _finite_blob(rukhi_qa_studies.bench_rukhi_qa_studies(seed))
