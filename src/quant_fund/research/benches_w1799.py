"""Wave-1799 bench adapters: japanese-myth-7 canon (SYNTHETIC only)."""

from quant_fund.models import (
    amaterasu2_qa_studies,
    hachiman2_qa_studies,
    kaguya2_qa_studies,
    sarutahiko2_qa_studies,
    susanoo2_qa_studies,
    tsukuyomi2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17990


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_amaterasu2_qa_studies_family(seed: int = _SEED + 0):
    """amaterasu2_qa_studies: synthetic correctness bench."""
    return _finite_blob(amaterasu2_qa_studies.bench_amaterasu2_qa_studies(seed))


def bench_hachiman2_qa_studies_family(seed: int = _SEED + 1):
    """hachiman2_qa_studies: synthetic correctness bench."""
    return _finite_blob(hachiman2_qa_studies.bench_hachiman2_qa_studies(seed))


def bench_kaguya2_qa_studies_family(seed: int = _SEED + 2):
    """kaguya2_qa_studies: synthetic correctness bench."""
    return _finite_blob(kaguya2_qa_studies.bench_kaguya2_qa_studies(seed))


def bench_sarutahiko2_qa_studies_family(seed: int = _SEED + 3):
    """sarutahiko2_qa_studies: synthetic correctness bench."""
    return _finite_blob(sarutahiko2_qa_studies.bench_sarutahiko2_qa_studies(seed))


def bench_susanoo2_qa_studies_family(seed: int = _SEED + 4):
    """susanoo2_qa_studies: synthetic correctness bench."""
    return _finite_blob(susanoo2_qa_studies.bench_susanoo2_qa_studies(seed))


def bench_tsukuyomi2_qa_studies_family(seed: int = _SEED + 5):
    """tsukuyomi2_qa_studies: synthetic correctness bench."""
    return _finite_blob(tsukuyomi2_qa_studies.bench_tsukuyomi2_qa_studies(seed))
