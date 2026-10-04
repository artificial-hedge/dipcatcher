"""Wave-1773 bench adapters: norse-myth-9 canon (SYNTHETIC only)."""

from quant_fund.models import (
    balder_qa_studies,
    frigg_qa_studies,
    loki_qa_studies,
    sif_qa_studies,
    vali_qa_studies,
    vitharr_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17730


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_balder_qa_studies_family(seed: int = _SEED + 0):
    """balder_qa_studies: synthetic correctness bench."""
    return _finite_blob(balder_qa_studies.bench_balder_qa_studies(seed))


def bench_frigg_qa_studies_family(seed: int = _SEED + 1):
    """frigg_qa_studies: synthetic correctness bench."""
    return _finite_blob(frigg_qa_studies.bench_frigg_qa_studies(seed))


def bench_loki_qa_studies_family(seed: int = _SEED + 2):
    """loki_qa_studies: synthetic correctness bench."""
    return _finite_blob(loki_qa_studies.bench_loki_qa_studies(seed))


def bench_sif_qa_studies_family(seed: int = _SEED + 3):
    """sif_qa_studies: synthetic correctness bench."""
    return _finite_blob(sif_qa_studies.bench_sif_qa_studies(seed))


def bench_vali_qa_studies_family(seed: int = _SEED + 4):
    """vali_qa_studies: synthetic correctness bench."""
    return _finite_blob(vali_qa_studies.bench_vali_qa_studies(seed))


def bench_vitharr_qa_studies_family(seed: int = _SEED + 5):
    """vitharr_qa_studies: synthetic correctness bench."""
    return _finite_blob(vitharr_qa_studies.bench_vitharr_qa_studies(seed))
