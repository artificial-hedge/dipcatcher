"""Wave-1869 bench adapters: breton-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    ankou_qa_studies,
    gwalarn_qa_studies,
    korrigan_qa_studies,
    mari_morgen_qa_studies,
    tangi_qa_studies,
    yeun_elez_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18690


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ankou_qa_studies_family(seed: int = _SEED + 0):
    """ankou_qa_studies: synthetic correctness bench."""
    return _finite_blob(ankou_qa_studies.bench_ankou_qa_studies(seed))


def bench_gwalarn_qa_studies_family(seed: int = _SEED + 1):
    """gwalarn_qa_studies: synthetic correctness bench."""
    return _finite_blob(gwalarn_qa_studies.bench_gwalarn_qa_studies(seed))


def bench_korrigan_qa_studies_family(seed: int = _SEED + 2):
    """korrigan_qa_studies: synthetic correctness bench."""
    return _finite_blob(korrigan_qa_studies.bench_korrigan_qa_studies(seed))


def bench_mari_morgen_qa_studies_family(seed: int = _SEED + 3):
    """mari_morgen_qa_studies: synthetic correctness bench."""
    return _finite_blob(mari_morgen_qa_studies.bench_mari_morgen_qa_studies(seed))


def bench_tangi_qa_studies_family(seed: int = _SEED + 4):
    """tangi_qa_studies: synthetic correctness bench."""
    return _finite_blob(tangi_qa_studies.bench_tangi_qa_studies(seed))


def bench_yeun_elez_qa_studies_family(seed: int = _SEED + 5):
    """yeun_elez_qa_studies: synthetic correctness bench."""
    return _finite_blob(yeun_elez_qa_studies.bench_yeun_elez_qa_studies(seed))
