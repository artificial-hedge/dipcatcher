"""Wave-1890 bench adapters: zoroastrian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    daeva_qa_studies,
    div_qa_studies,
    fravashi_qa_studies,
    khshathra_qa_studies,
    pairika_qa_studies,
    spenta_mainyu_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18900


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_daeva_qa_studies_family(seed: int = _SEED + 0):
    """daeva_qa_studies: synthetic correctness bench."""
    return _finite_blob(daeva_qa_studies.bench_daeva_qa_studies(seed))


def bench_div_qa_studies_family(seed: int = _SEED + 1):
    """div_qa_studies: synthetic correctness bench."""
    return _finite_blob(div_qa_studies.bench_div_qa_studies(seed))


def bench_fravashi_qa_studies_family(seed: int = _SEED + 2):
    """fravashi_qa_studies: synthetic correctness bench."""
    return _finite_blob(fravashi_qa_studies.bench_fravashi_qa_studies(seed))


def bench_khshathra_qa_studies_family(seed: int = _SEED + 3):
    """khshathra_qa_studies: synthetic correctness bench."""
    return _finite_blob(khshathra_qa_studies.bench_khshathra_qa_studies(seed))


def bench_pairika_qa_studies_family(seed: int = _SEED + 4):
    """pairika_qa_studies: synthetic correctness bench."""
    return _finite_blob(pairika_qa_studies.bench_pairika_qa_studies(seed))


def bench_spenta_mainyu_qa_studies_family(seed: int = _SEED + 5):
    """spenta_mainyu_qa_studies: synthetic correctness bench."""
    return _finite_blob(spenta_mainyu_qa_studies.bench_spenta_mainyu_qa_studies(seed))
