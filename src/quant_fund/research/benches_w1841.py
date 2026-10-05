"""Wave-1841 bench adapters: moabite-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    ashtar2_qa_studies,
    baalpeor_qa_studies,
    chemosh3_qa_studies,
    dibon2_qa_studies,
    kiriath2_qa_studies,
    nebo2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18410


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ashtar2_qa_studies_family(seed: int = _SEED + 0):
    """ashtar2_qa_studies: synthetic correctness bench."""
    return _finite_blob(ashtar2_qa_studies.bench_ashtar2_qa_studies(seed))


def bench_baalpeor_qa_studies_family(seed: int = _SEED + 1):
    """baalpeor_qa_studies: synthetic correctness bench."""
    return _finite_blob(baalpeor_qa_studies.bench_baalpeor_qa_studies(seed))


def bench_chemosh3_qa_studies_family(seed: int = _SEED + 2):
    """chemosh3_qa_studies: synthetic correctness bench."""
    return _finite_blob(chemosh3_qa_studies.bench_chemosh3_qa_studies(seed))


def bench_dibon2_qa_studies_family(seed: int = _SEED + 3):
    """dibon2_qa_studies: synthetic correctness bench."""
    return _finite_blob(dibon2_qa_studies.bench_dibon2_qa_studies(seed))


def bench_kiriath2_qa_studies_family(seed: int = _SEED + 4):
    """kiriath2_qa_studies: synthetic correctness bench."""
    return _finite_blob(kiriath2_qa_studies.bench_kiriath2_qa_studies(seed))


def bench_nebo2_qa_studies_family(seed: int = _SEED + 5):
    """nebo2_qa_studies: synthetic correctness bench."""
    return _finite_blob(nebo2_qa_studies.bench_nebo2_qa_studies(seed))
