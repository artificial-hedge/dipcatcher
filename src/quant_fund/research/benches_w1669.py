"""Wave-1669 bench adapters: filipino-creature-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    ghouling_qa_studies,
    ikugan_qa_studies,
    kataw_qa_studies,
    lambana_qa_studies,
    sarimanok_qa_studies,
    tamahaling_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16690


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ghouling_qa_studies_family(seed: int = _SEED + 0):
    """ghouling_qa_studies: synthetic correctness bench."""
    return _finite_blob(ghouling_qa_studies.bench_ghouling_qa_studies(seed))


def bench_ikugan_qa_studies_family(seed: int = _SEED + 1):
    """ikugan_qa_studies: synthetic correctness bench."""
    return _finite_blob(ikugan_qa_studies.bench_ikugan_qa_studies(seed))


def bench_kataw_qa_studies_family(seed: int = _SEED + 2):
    """kataw_qa_studies: synthetic correctness bench."""
    return _finite_blob(kataw_qa_studies.bench_kataw_qa_studies(seed))


def bench_lambana_qa_studies_family(seed: int = _SEED + 3):
    """lambana_qa_studies: synthetic correctness bench."""
    return _finite_blob(lambana_qa_studies.bench_lambana_qa_studies(seed))


def bench_sarimanok_qa_studies_family(seed: int = _SEED + 4):
    """sarimanok_qa_studies: synthetic correctness bench."""
    return _finite_blob(sarimanok_qa_studies.bench_sarimanok_qa_studies(seed))


def bench_tamahaling_qa_studies_family(seed: int = _SEED + 5):
    """tamahaling_qa_studies: synthetic correctness bench."""
    return _finite_blob(tamahaling_qa_studies.bench_tamahaling_qa_studies(seed))
