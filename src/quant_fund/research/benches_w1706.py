"""Wave-1706 bench adapters: siberian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    akana_qa_studies,
    erlik_qa_studies,
    kayra_qa_studies,
    perysh_qa_studies,
    tengri_qa_studies,
    ulgen_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17060


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_akana_qa_studies_family(seed: int = _SEED + 0):
    """akana_qa_studies: synthetic correctness bench."""
    return _finite_blob(akana_qa_studies.bench_akana_qa_studies(seed))


def bench_erlik_qa_studies_family(seed: int = _SEED + 1):
    """erlik_qa_studies: synthetic correctness bench."""
    return _finite_blob(erlik_qa_studies.bench_erlik_qa_studies(seed))


def bench_kayra_qa_studies_family(seed: int = _SEED + 2):
    """kayra_qa_studies: synthetic correctness bench."""
    return _finite_blob(kayra_qa_studies.bench_kayra_qa_studies(seed))


def bench_perysh_qa_studies_family(seed: int = _SEED + 3):
    """perysh_qa_studies: synthetic correctness bench."""
    return _finite_blob(perysh_qa_studies.bench_perysh_qa_studies(seed))


def bench_tengri_qa_studies_family(seed: int = _SEED + 4):
    """tengri_qa_studies: synthetic correctness bench."""
    return _finite_blob(tengri_qa_studies.bench_tengri_qa_studies(seed))


def bench_ulgen_qa_studies_family(seed: int = _SEED + 5):
    """ulgen_qa_studies: synthetic correctness bench."""
    return _finite_blob(ulgen_qa_studies.bench_ulgen_qa_studies(seed))
