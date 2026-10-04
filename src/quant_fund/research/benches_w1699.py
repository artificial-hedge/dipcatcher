"""Wave-1699 bench adapters: japanese-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    amaterasu_qa_studies,
    hachiman_qa_studies,
    inari_qa_studies,
    raijin_qa_studies,
    susanoo_qa_studies,
    tsukuyomi_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16990


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_amaterasu_qa_studies_family(seed: int = _SEED + 0):
    """amaterasu_qa_studies: synthetic correctness bench."""
    return _finite_blob(amaterasu_qa_studies.bench_amaterasu_qa_studies(seed))


def bench_hachiman_qa_studies_family(seed: int = _SEED + 1):
    """hachiman_qa_studies: synthetic correctness bench."""
    return _finite_blob(hachiman_qa_studies.bench_hachiman_qa_studies(seed))


def bench_inari_qa_studies_family(seed: int = _SEED + 2):
    """inari_qa_studies: synthetic correctness bench."""
    return _finite_blob(inari_qa_studies.bench_inari_qa_studies(seed))


def bench_raijin_qa_studies_family(seed: int = _SEED + 3):
    """raijin_qa_studies: synthetic correctness bench."""
    return _finite_blob(raijin_qa_studies.bench_raijin_qa_studies(seed))


def bench_susanoo_qa_studies_family(seed: int = _SEED + 4):
    """susanoo_qa_studies: synthetic correctness bench."""
    return _finite_blob(susanoo_qa_studies.bench_susanoo_qa_studies(seed))


def bench_tsukuyomi_qa_studies_family(seed: int = _SEED + 5):
    """tsukuyomi_qa_studies: synthetic correctness bench."""
    return _finite_blob(tsukuyomi_qa_studies.bench_tsukuyomi_qa_studies(seed))
