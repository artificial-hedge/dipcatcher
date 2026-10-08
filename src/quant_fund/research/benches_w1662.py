"""Wave-1662 bench adapters: celtic-beast canon (SYNTHETIC only)."""

from quant_fund.models import (
    banshee_qa_studies,
    dullahan_qa_studies,
    kelpie_qa_studies,
    leprechaun_qa_studies,
    puca_qa_studies,
    selkie_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16620


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


def bench_banshee_qa_studies_family(seed: int = _SEED + 0):
    """banshee_qa_studies: synthetic correctness bench."""
    return _finite_blob(banshee_qa_studies.bench_banshee_qa_studies(seed))


def bench_dullahan_qa_studies_family(seed: int = _SEED + 1):
    """dullahan_qa_studies: synthetic correctness bench."""
    return _finite_blob(dullahan_qa_studies.bench_dullahan_qa_studies(seed))


def bench_kelpie_qa_studies_family(seed: int = _SEED + 2):
    """kelpie_qa_studies: synthetic correctness bench."""
    return _finite_blob(kelpie_qa_studies.bench_kelpie_qa_studies(seed))


def bench_leprechaun_qa_studies_family(seed: int = _SEED + 3):
    """leprechaun_qa_studies: synthetic correctness bench."""
    return _finite_blob(leprechaun_qa_studies.bench_leprechaun_qa_studies(seed))


def bench_puca_qa_studies_family(seed: int = _SEED + 4):
    """puca_qa_studies: synthetic correctness bench."""
    return _finite_blob(puca_qa_studies.bench_puca_qa_studies(seed))


def bench_selkie_qa_studies_family(seed: int = _SEED + 5):
    """selkie_qa_studies: synthetic correctness bench."""
    return _finite_blob(selkie_qa_studies.bench_selkie_qa_studies(seed))
