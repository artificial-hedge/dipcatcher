"""Wave-1756 bench adapters: hindu-myth-5 canon (SYNTHETIC only)."""

from quant_fund.models import (
    apsara_qa_studies,
    gandharva_qa_studies,
    kinnara_qa_studies,
    ratri_qa_studies,
    rudra_qa_studies,
    ushas_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17560


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_apsara_qa_studies_family(seed: int = _SEED + 0):
    """apsara_qa_studies: synthetic correctness bench."""
    return _finite_blob(apsara_qa_studies.bench_apsara_qa_studies(seed))


def bench_gandharva_qa_studies_family(seed: int = _SEED + 1):
    """gandharva_qa_studies: synthetic correctness bench."""
    return _finite_blob(gandharva_qa_studies.bench_gandharva_qa_studies(seed))


def bench_kinnara_qa_studies_family(seed: int = _SEED + 2):
    """kinnara_qa_studies: synthetic correctness bench."""
    return _finite_blob(kinnara_qa_studies.bench_kinnara_qa_studies(seed))


def bench_ratri_qa_studies_family(seed: int = _SEED + 3):
    """ratri_qa_studies: synthetic correctness bench."""
    return _finite_blob(ratri_qa_studies.bench_ratri_qa_studies(seed))


def bench_rudra_qa_studies_family(seed: int = _SEED + 4):
    """rudra_qa_studies: synthetic correctness bench."""
    return _finite_blob(rudra_qa_studies.bench_rudra_qa_studies(seed))


def bench_ushas_qa_studies_family(seed: int = _SEED + 5):
    """ushas_qa_studies: synthetic correctness bench."""
    return _finite_blob(ushas_qa_studies.bench_ushas_qa_studies(seed))
