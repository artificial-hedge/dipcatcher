"""Wave-1668 bench adapters: norse-warrior canon (SYNTHETIC only)."""

from quant_fund.models import (
    berserkr_qa_studies,
    fafnir_qa_studies,
    jotun_qa_studies,
    regin_qa_studies,
    ulfhednar_qa_studies,
    vargr_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16680


def _finite_blob(blob):
    if not (isinstance(blob, dict) and blob):
        raise ValueError("bench blob must be a non-empty dict")
    for k, v in blob.items():
        if not k.startswith("synthetic_"):
            raise ValueError(f"non-synthetic metric key {k}")
        if k in _FORBIDDEN:
            raise ValueError(f"forbidden metric key {k}")
        if not (isinstance(v, float) and 0.0 <= v <= 1.0):
            raise ValueError(f"metric {k} is not a [0,1] float")
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_berserkr_qa_studies_family(seed: int = _SEED + 0):
    """berserkr_qa_studies: synthetic correctness bench."""
    return _finite_blob(berserkr_qa_studies.bench_berserkr_qa_studies(seed))


def bench_fafnir_qa_studies_family(seed: int = _SEED + 1):
    """fafnir_qa_studies: synthetic correctness bench."""
    return _finite_blob(fafnir_qa_studies.bench_fafnir_qa_studies(seed))


def bench_jotun_qa_studies_family(seed: int = _SEED + 2):
    """jotun_qa_studies: synthetic correctness bench."""
    return _finite_blob(jotun_qa_studies.bench_jotun_qa_studies(seed))


def bench_regin_qa_studies_family(seed: int = _SEED + 3):
    """regin_qa_studies: synthetic correctness bench."""
    return _finite_blob(regin_qa_studies.bench_regin_qa_studies(seed))


def bench_ulfhednar_qa_studies_family(seed: int = _SEED + 4):
    """ulfhednar_qa_studies: synthetic correctness bench."""
    return _finite_blob(ulfhednar_qa_studies.bench_ulfhednar_qa_studies(seed))


def bench_vargr_qa_studies_family(seed: int = _SEED + 5):
    """vargr_qa_studies: synthetic correctness bench."""
    return _finite_blob(vargr_qa_studies.bench_vargr_qa_studies(seed))
