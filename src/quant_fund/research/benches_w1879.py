"""Wave-1879 bench adapters: tuareg canon (SYNTHETIC only)."""

from quant_fund.models import (
    afriye_qa_studies,
    almajira_qa_studies,
    djinnet_qa_studies,
    kel_essuf_qa_studies,
    tanit_lok_qa_studies,
    tin_hinan_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18790


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_afriye_qa_studies_family(seed: int = _SEED + 0):
    """afriye_qa_studies: synthetic correctness bench."""
    return _finite_blob(afriye_qa_studies.bench_afriye_qa_studies(seed))


def bench_almajira_qa_studies_family(seed: int = _SEED + 1):
    """almajira_qa_studies: synthetic correctness bench."""
    return _finite_blob(almajira_qa_studies.bench_almajira_qa_studies(seed))


def bench_djinnet_qa_studies_family(seed: int = _SEED + 2):
    """djinnet_qa_studies: synthetic correctness bench."""
    return _finite_blob(djinnet_qa_studies.bench_djinnet_qa_studies(seed))


def bench_kel_essuf_qa_studies_family(seed: int = _SEED + 3):
    """kel_essuf_qa_studies: synthetic correctness bench."""
    return _finite_blob(kel_essuf_qa_studies.bench_kel_essuf_qa_studies(seed))


def bench_tanit_lok_qa_studies_family(seed: int = _SEED + 4):
    """tanit_lok_qa_studies: synthetic correctness bench."""
    return _finite_blob(tanit_lok_qa_studies.bench_tanit_lok_qa_studies(seed))


def bench_tin_hinan_qa_studies_family(seed: int = _SEED + 5):
    """tin_hinan_qa_studies: synthetic correctness bench."""
    return _finite_blob(tin_hinan_qa_studies.bench_tin_hinan_qa_studies(seed))
