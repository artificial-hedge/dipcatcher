"""Wave-1775 bench adapters: japanese-myth-5 canon (SYNTHETIC only)."""

from quant_fund.models import (
    inari_qa_studies,
    kaguya_qa_studies,
    momotaro_qa_studies,
    shichifukujin_qa_studies,
    takemikazuchi_qa_studies,
    urashima_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17750


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_inari_qa_studies_family(seed: int = _SEED + 0):
    """inari_qa_studies: synthetic correctness bench."""
    return _finite_blob(inari_qa_studies.bench_inari_qa_studies(seed))


def bench_kaguya_qa_studies_family(seed: int = _SEED + 1):
    """kaguya_qa_studies: synthetic correctness bench."""
    return _finite_blob(kaguya_qa_studies.bench_kaguya_qa_studies(seed))


def bench_momotaro_qa_studies_family(seed: int = _SEED + 2):
    """momotaro_qa_studies: synthetic correctness bench."""
    return _finite_blob(momotaro_qa_studies.bench_momotaro_qa_studies(seed))


def bench_shichifukujin_qa_studies_family(seed: int = _SEED + 3):
    """shichifukujin_qa_studies: synthetic correctness bench."""
    return _finite_blob(shichifukujin_qa_studies.bench_shichifukujin_qa_studies(seed))


def bench_takemikazuchi_qa_studies_family(seed: int = _SEED + 4):
    """takemikazuchi_qa_studies: synthetic correctness bench."""
    return _finite_blob(takemikazuchi_qa_studies.bench_takemikazuchi_qa_studies(seed))


def bench_urashima_qa_studies_family(seed: int = _SEED + 5):
    """urashima_qa_studies: synthetic correctness bench."""
    return _finite_blob(urashima_qa_studies.bench_urashima_qa_studies(seed))
