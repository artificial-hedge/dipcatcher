"""Wave-1682 bench adapters: slavic-wild canon (SYNTHETIC only)."""

from quant_fund.models import (
    alkonost_qa_studies,
    gamayun_qa_studies,
    sirin_qa_studies,
    veles_qa_studies,
    zhaba_qa_studies,
    zmei_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16820


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_alkonost_qa_studies_family(seed: int = _SEED + 0):
    """alkonost_qa_studies: synthetic correctness bench."""
    return _finite_blob(alkonost_qa_studies.bench_alkonost_qa_studies(seed))


def bench_gamayun_qa_studies_family(seed: int = _SEED + 1):
    """gamayun_qa_studies: synthetic correctness bench."""
    return _finite_blob(gamayun_qa_studies.bench_gamayun_qa_studies(seed))


def bench_sirin_qa_studies_family(seed: int = _SEED + 2):
    """sirin_qa_studies: synthetic correctness bench."""
    return _finite_blob(sirin_qa_studies.bench_sirin_qa_studies(seed))


def bench_veles_qa_studies_family(seed: int = _SEED + 3):
    """veles_qa_studies: synthetic correctness bench."""
    return _finite_blob(veles_qa_studies.bench_veles_qa_studies(seed))


def bench_zhaba_qa_studies_family(seed: int = _SEED + 4):
    """zhaba_qa_studies: synthetic correctness bench."""
    return _finite_blob(zhaba_qa_studies.bench_zhaba_qa_studies(seed))


def bench_zmei_qa_studies_family(seed: int = _SEED + 5):
    """zmei_qa_studies: synthetic correctness bench."""
    return _finite_blob(zmei_qa_studies.bench_zmei_qa_studies(seed))
