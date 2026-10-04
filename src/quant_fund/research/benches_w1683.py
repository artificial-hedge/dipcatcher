"""Wave-1683 bench adapters: hindu-myth-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    danava_qa_studies,
    gana_qa_studies,
    gandharva_qa_studies,
    kalakeya_qa_studies,
    kimpurusha_qa_studies,
    rakshasa_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16830


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_danava_qa_studies_family(seed: int = _SEED + 0):
    """danava_qa_studies: synthetic correctness bench."""
    return _finite_blob(danava_qa_studies.bench_danava_qa_studies(seed))


def bench_gana_qa_studies_family(seed: int = _SEED + 1):
    """gana_qa_studies: synthetic correctness bench."""
    return _finite_blob(gana_qa_studies.bench_gana_qa_studies(seed))


def bench_gandharva_qa_studies_family(seed: int = _SEED + 2):
    """gandharva_qa_studies: synthetic correctness bench."""
    return _finite_blob(gandharva_qa_studies.bench_gandharva_qa_studies(seed))


def bench_kalakeya_qa_studies_family(seed: int = _SEED + 3):
    """kalakeya_qa_studies: synthetic correctness bench."""
    return _finite_blob(kalakeya_qa_studies.bench_kalakeya_qa_studies(seed))


def bench_kimpurusha_qa_studies_family(seed: int = _SEED + 4):
    """kimpurusha_qa_studies: synthetic correctness bench."""
    return _finite_blob(kimpurusha_qa_studies.bench_kimpurusha_qa_studies(seed))


def bench_rakshasa_qa_studies_family(seed: int = _SEED + 5):
    """rakshasa_qa_studies: synthetic correctness bench."""
    return _finite_blob(rakshasa_qa_studies.bench_rakshasa_qa_studies(seed))
