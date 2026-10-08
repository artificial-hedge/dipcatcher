"""Wave-1504 bench adapters: seabird-4 canon (SYNTHETIC only)."""

from quant_fund.models import (
    murre_qa_studies,
    noddie_qa_studies,
    prion_qa_studies,
    shag_qa_studies,
    skimmer_qa_studies,
    storm_petrel_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15040


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


def bench_murre_qa_studies_family(seed: int = _SEED + 0):
    """murre_qa_studies: synthetic correctness bench."""
    return _finite_blob(murre_qa_studies.bench_murre_qa_studies(seed))


def bench_noddie_qa_studies_family(seed: int = _SEED + 1):
    """noddie_qa_studies: synthetic correctness bench."""
    return _finite_blob(noddie_qa_studies.bench_noddie_qa_studies(seed))


def bench_prion_qa_studies_family(seed: int = _SEED + 2):
    """prion_qa_studies: synthetic correctness bench."""
    return _finite_blob(prion_qa_studies.bench_prion_qa_studies(seed))


def bench_shag_qa_studies_family(seed: int = _SEED + 3):
    """shag_qa_studies: synthetic correctness bench."""
    return _finite_blob(shag_qa_studies.bench_shag_qa_studies(seed))


def bench_skimmer_qa_studies_family(seed: int = _SEED + 4):
    """skimmer_qa_studies: synthetic correctness bench."""
    return _finite_blob(skimmer_qa_studies.bench_skimmer_qa_studies(seed))


def bench_storm_petrel_qa_studies_family(seed: int = _SEED + 5):
    """storm_petrel_qa_studies: synthetic correctness bench."""
    return _finite_blob(storm_petrel_qa_studies.bench_storm_petrel_qa_studies(seed))
