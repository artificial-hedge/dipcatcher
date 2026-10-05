"""Wave-1880 bench adapters: tuareg-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    achimi_qa_studies,
    iyezid_qa_studies,
    mazer_qa_studies,
    milkart_qa_studies,
    tamgak_qa_studies,
    tesfit_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18800


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_achimi_qa_studies_family(seed: int = _SEED + 0):
    """achimi_qa_studies: synthetic correctness bench."""
    return _finite_blob(achimi_qa_studies.bench_achimi_qa_studies(seed))


def bench_iyezid_qa_studies_family(seed: int = _SEED + 1):
    """iyezid_qa_studies: synthetic correctness bench."""
    return _finite_blob(iyezid_qa_studies.bench_iyezid_qa_studies(seed))


def bench_mazer_qa_studies_family(seed: int = _SEED + 2):
    """mazer_qa_studies: synthetic correctness bench."""
    return _finite_blob(mazer_qa_studies.bench_mazer_qa_studies(seed))


def bench_milkart_qa_studies_family(seed: int = _SEED + 3):
    """milkart_qa_studies: synthetic correctness bench."""
    return _finite_blob(milkart_qa_studies.bench_milkart_qa_studies(seed))


def bench_tamgak_qa_studies_family(seed: int = _SEED + 4):
    """tamgak_qa_studies: synthetic correctness bench."""
    return _finite_blob(tamgak_qa_studies.bench_tamgak_qa_studies(seed))


def bench_tesfit_qa_studies_family(seed: int = _SEED + 5):
    """tesfit_qa_studies: synthetic correctness bench."""
    return _finite_blob(tesfit_qa_studies.bench_tesfit_qa_studies(seed))
