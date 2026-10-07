"""Wave-1521 bench adapters: orchid canon (SYNTHETIC only)."""

from quant_fund.models import (
    cattleya_qa_studies,
    cymbidium_qa_studies,
    dendrobium_qa_studies,
    oncidium_qa_studies,
    paphiopedilum_qa_studies,
    phalaenopsis_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15210


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


def bench_cattleya_qa_studies_family(seed: int = _SEED + 0):
    """cattleya_qa_studies: synthetic correctness bench."""
    return _finite_blob(cattleya_qa_studies.bench_cattleya_qa_studies(seed))


def bench_cymbidium_qa_studies_family(seed: int = _SEED + 1):
    """cymbidium_qa_studies: synthetic correctness bench."""
    return _finite_blob(cymbidium_qa_studies.bench_cymbidium_qa_studies(seed))


def bench_dendrobium_qa_studies_family(seed: int = _SEED + 2):
    """dendrobium_qa_studies: synthetic correctness bench."""
    return _finite_blob(dendrobium_qa_studies.bench_dendrobium_qa_studies(seed))


def bench_oncidium_qa_studies_family(seed: int = _SEED + 3):
    """oncidium_qa_studies: synthetic correctness bench."""
    return _finite_blob(oncidium_qa_studies.bench_oncidium_qa_studies(seed))


def bench_paphiopedilum_qa_studies_family(seed: int = _SEED + 4):
    """paphiopedilum_qa_studies: synthetic correctness bench."""
    return _finite_blob(paphiopedilum_qa_studies.bench_paphiopedilum_qa_studies(seed))


def bench_phalaenopsis_qa_studies_family(seed: int = _SEED + 5):
    """phalaenopsis_qa_studies: synthetic correctness bench."""
    return _finite_blob(phalaenopsis_qa_studies.bench_phalaenopsis_qa_studies(seed))
