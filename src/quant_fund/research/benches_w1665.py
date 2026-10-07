"""Wave-1665 bench adapters: norse-realm canon (SYNTHETIC only)."""

from quant_fund.models import (
    einherjar_qa_studies,
    hati_qa_studies,
    lindworm_qa_studies,
    skoll_qa_studies,
    vargbroder_qa_studies,
    vedrfolnir_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16650


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


def bench_einherjar_qa_studies_family(seed: int = _SEED + 0):
    """einherjar_qa_studies: synthetic correctness bench."""
    return _finite_blob(einherjar_qa_studies.bench_einherjar_qa_studies(seed))


def bench_hati_qa_studies_family(seed: int = _SEED + 1):
    """hati_qa_studies: synthetic correctness bench."""
    return _finite_blob(hati_qa_studies.bench_hati_qa_studies(seed))


def bench_lindworm_qa_studies_family(seed: int = _SEED + 2):
    """lindworm_qa_studies: synthetic correctness bench."""
    return _finite_blob(lindworm_qa_studies.bench_lindworm_qa_studies(seed))


def bench_skoll_qa_studies_family(seed: int = _SEED + 3):
    """skoll_qa_studies: synthetic correctness bench."""
    return _finite_blob(skoll_qa_studies.bench_skoll_qa_studies(seed))


def bench_vargbroder_qa_studies_family(seed: int = _SEED + 4):
    """vargbroder_qa_studies: synthetic correctness bench."""
    return _finite_blob(vargbroder_qa_studies.bench_vargbroder_qa_studies(seed))


def bench_vedrfolnir_qa_studies_family(seed: int = _SEED + 5):
    """vedrfolnir_qa_studies: synthetic correctness bench."""
    return _finite_blob(vedrfolnir_qa_studies.bench_vedrfolnir_qa_studies(seed))
