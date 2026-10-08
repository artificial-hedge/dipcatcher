"""Wave-1581 bench adapters: pinniped canon (SYNTHETIC only)."""

from quant_fund.models import (
    elephant_seal_qa_studies,
    fur_seal_qa_studies,
    harp_seal_qa_studies,
    leopard_seal_qa_studies,
    monk_seal_qa_studies,
    weddell_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15810


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


def bench_elephant_seal_qa_studies_family(seed: int = _SEED + 0):
    """elephant_seal_qa_studies: synthetic correctness bench."""
    return _finite_blob(elephant_seal_qa_studies.bench_elephant_seal_qa_studies(seed))


def bench_fur_seal_qa_studies_family(seed: int = _SEED + 1):
    """fur_seal_qa_studies: synthetic correctness bench."""
    return _finite_blob(fur_seal_qa_studies.bench_fur_seal_qa_studies(seed))


def bench_harp_seal_qa_studies_family(seed: int = _SEED + 2):
    """harp_seal_qa_studies: synthetic correctness bench."""
    return _finite_blob(harp_seal_qa_studies.bench_harp_seal_qa_studies(seed))


def bench_leopard_seal_qa_studies_family(seed: int = _SEED + 3):
    """leopard_seal_qa_studies: synthetic correctness bench."""
    return _finite_blob(leopard_seal_qa_studies.bench_leopard_seal_qa_studies(seed))


def bench_monk_seal_qa_studies_family(seed: int = _SEED + 4):
    """monk_seal_qa_studies: synthetic correctness bench."""
    return _finite_blob(monk_seal_qa_studies.bench_monk_seal_qa_studies(seed))


def bench_weddell_qa_studies_family(seed: int = _SEED + 5):
    """weddell_qa_studies: synthetic correctness bench."""
    return _finite_blob(weddell_qa_studies.bench_weddell_qa_studies(seed))
