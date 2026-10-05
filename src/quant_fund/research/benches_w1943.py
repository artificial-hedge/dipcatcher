"""Wave-1943 bench adapters: javanese-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    genderuwo_qa_studies,
    jenglot_qa_studies,
    kuntilanak_qa_studies,
    leyak_qa_studies,
    pocong_qa_studies,
    tuyul_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19430


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_genderuwo_qa_studies_family(seed: int = _SEED + 0):
    """genderuwo_qa_studies: synthetic correctness bench."""
    return _finite_blob(genderuwo_qa_studies.bench_genderuwo_qa_studies(seed))


def bench_jenglot_qa_studies_family(seed: int = _SEED + 1):
    """jenglot_qa_studies: synthetic correctness bench."""
    return _finite_blob(jenglot_qa_studies.bench_jenglot_qa_studies(seed))


def bench_kuntilanak_qa_studies_family(seed: int = _SEED + 2):
    """kuntilanak_qa_studies: synthetic correctness bench."""
    return _finite_blob(kuntilanak_qa_studies.bench_kuntilanak_qa_studies(seed))


def bench_leyak_qa_studies_family(seed: int = _SEED + 3):
    """leyak_qa_studies: synthetic correctness bench."""
    return _finite_blob(leyak_qa_studies.bench_leyak_qa_studies(seed))


def bench_pocong_qa_studies_family(seed: int = _SEED + 4):
    """pocong_qa_studies: synthetic correctness bench."""
    return _finite_blob(pocong_qa_studies.bench_pocong_qa_studies(seed))


def bench_tuyul_qa_studies_family(seed: int = _SEED + 5):
    """tuyul_qa_studies: synthetic correctness bench."""
    return _finite_blob(tuyul_qa_studies.bench_tuyul_qa_studies(seed))
