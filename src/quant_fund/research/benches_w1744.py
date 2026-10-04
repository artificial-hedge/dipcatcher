"""Wave-1744 bench adapters: inuit-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    agloolik_qa_studies,
    aumanil_qa_studies,
    nuktessien_qa_studies,
    sedna_qa_studies,
    tekkeitsertok_qa_studies,
    torngarsuk_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17440


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_agloolik_qa_studies_family(seed: int = _SEED + 0):
    """agloolik_qa_studies: synthetic correctness bench."""
    return _finite_blob(agloolik_qa_studies.bench_agloolik_qa_studies(seed))


def bench_aumanil_qa_studies_family(seed: int = _SEED + 1):
    """aumanil_qa_studies: synthetic correctness bench."""
    return _finite_blob(aumanil_qa_studies.bench_aumanil_qa_studies(seed))


def bench_nuktessien_qa_studies_family(seed: int = _SEED + 2):
    """nuktessien_qa_studies: synthetic correctness bench."""
    return _finite_blob(nuktessien_qa_studies.bench_nuktessien_qa_studies(seed))


def bench_sedna_qa_studies_family(seed: int = _SEED + 3):
    """sedna_qa_studies: synthetic correctness bench."""
    return _finite_blob(sedna_qa_studies.bench_sedna_qa_studies(seed))


def bench_tekkeitsertok_qa_studies_family(seed: int = _SEED + 4):
    """tekkeitsertok_qa_studies: synthetic correctness bench."""
    return _finite_blob(tekkeitsertok_qa_studies.bench_tekkeitsertok_qa_studies(seed))


def bench_torngarsuk_qa_studies_family(seed: int = _SEED + 5):
    """torngarsuk_qa_studies: synthetic correctness bench."""
    return _finite_blob(torngarsuk_qa_studies.bench_torngarsuk_qa_studies(seed))
