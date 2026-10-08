"""Wave-1629 bench adapters: cryptid canon (SYNTHETIC only)."""

from quant_fund.models import (
    chupacabra_qa_studies,
    jersey_devil_qa_studies,
    kraken_2_qa_studies,
    mothman_qa_studies,
    thunderbird_qa_studies,
    yeti_2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16290


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


def bench_chupacabra_qa_studies_family(seed: int = _SEED + 0):
    """chupacabra_qa_studies: synthetic correctness bench."""
    return _finite_blob(chupacabra_qa_studies.bench_chupacabra_qa_studies(seed))


def bench_jersey_devil_qa_studies_family(seed: int = _SEED + 1):
    """jersey_devil_qa_studies: synthetic correctness bench."""
    return _finite_blob(jersey_devil_qa_studies.bench_jersey_devil_qa_studies(seed))


def bench_kraken_2_qa_studies_family(seed: int = _SEED + 2):
    """kraken_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(kraken_2_qa_studies.bench_kraken_2_qa_studies(seed))


def bench_mothman_qa_studies_family(seed: int = _SEED + 3):
    """mothman_qa_studies: synthetic correctness bench."""
    return _finite_blob(mothman_qa_studies.bench_mothman_qa_studies(seed))


def bench_thunderbird_qa_studies_family(seed: int = _SEED + 4):
    """thunderbird_qa_studies: synthetic correctness bench."""
    return _finite_blob(thunderbird_qa_studies.bench_thunderbird_qa_studies(seed))


def bench_yeti_2_qa_studies_family(seed: int = _SEED + 5):
    """yeti_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(yeti_2_qa_studies.bench_yeti_2_qa_studies(seed))
