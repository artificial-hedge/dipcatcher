"""Wave-1722 bench adapters: ossetian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    barastir_qa_studies,
    donbettyr_qa_studies,
    nart_qa_studies,
    safa_qa_studies,
    styr_qa_studies,
    tulur_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17220


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_barastir_qa_studies_family(seed: int = _SEED + 0):
    """barastir_qa_studies: synthetic correctness bench."""
    return _finite_blob(barastir_qa_studies.bench_barastir_qa_studies(seed))


def bench_donbettyr_qa_studies_family(seed: int = _SEED + 1):
    """donbettyr_qa_studies: synthetic correctness bench."""
    return _finite_blob(donbettyr_qa_studies.bench_donbettyr_qa_studies(seed))


def bench_nart_qa_studies_family(seed: int = _SEED + 2):
    """nart_qa_studies: synthetic correctness bench."""
    return _finite_blob(nart_qa_studies.bench_nart_qa_studies(seed))


def bench_safa_qa_studies_family(seed: int = _SEED + 3):
    """safa_qa_studies: synthetic correctness bench."""
    return _finite_blob(safa_qa_studies.bench_safa_qa_studies(seed))


def bench_styr_qa_studies_family(seed: int = _SEED + 4):
    """styr_qa_studies: synthetic correctness bench."""
    return _finite_blob(styr_qa_studies.bench_styr_qa_studies(seed))


def bench_tulur_qa_studies_family(seed: int = _SEED + 5):
    """tulur_qa_studies: synthetic correctness bench."""
    return _finite_blob(tulur_qa_studies.bench_tulur_qa_studies(seed))
