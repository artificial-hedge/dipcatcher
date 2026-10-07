"""Wave-1702 bench adapters: mayan-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    chac_qa_studies,
    hunab_qa_studies,
    itzamna_qa_studies,
    ixchel_qa_studies,
    kukulcan_qa_studies,
    yumkaax_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17020


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


def bench_chac_qa_studies_family(seed: int = _SEED + 0):
    """chac_qa_studies: synthetic correctness bench."""
    return _finite_blob(chac_qa_studies.bench_chac_qa_studies(seed))


def bench_hunab_qa_studies_family(seed: int = _SEED + 1):
    """hunab_qa_studies: synthetic correctness bench."""
    return _finite_blob(hunab_qa_studies.bench_hunab_qa_studies(seed))


def bench_itzamna_qa_studies_family(seed: int = _SEED + 2):
    """itzamna_qa_studies: synthetic correctness bench."""
    return _finite_blob(itzamna_qa_studies.bench_itzamna_qa_studies(seed))


def bench_ixchel_qa_studies_family(seed: int = _SEED + 3):
    """ixchel_qa_studies: synthetic correctness bench."""
    return _finite_blob(ixchel_qa_studies.bench_ixchel_qa_studies(seed))


def bench_kukulcan_qa_studies_family(seed: int = _SEED + 4):
    """kukulcan_qa_studies: synthetic correctness bench."""
    return _finite_blob(kukulcan_qa_studies.bench_kukulcan_qa_studies(seed))


def bench_yumkaax_qa_studies_family(seed: int = _SEED + 5):
    """yumkaax_qa_studies: synthetic correctness bench."""
    return _finite_blob(yumkaax_qa_studies.bench_yumkaax_qa_studies(seed))
