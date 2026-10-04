"""Wave-1485 bench adapters: shorebird canon (SYNTHETIC only)."""

from quant_fund.models import (
    avocet_qa_studies,
    egret_qa_studies,
    heron_qa_studies,
    plover_qa_studies,
    sandpiper_qa_studies,
    tern_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14850


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_avocet_qa_studies_family(seed: int = _SEED + 0):
    """avocet_qa_studies: synthetic correctness bench."""
    return _finite_blob(avocet_qa_studies.bench_avocet_qa_studies(seed))


def bench_egret_qa_studies_family(seed: int = _SEED + 1):
    """egret_qa_studies: synthetic correctness bench."""
    return _finite_blob(egret_qa_studies.bench_egret_qa_studies(seed))


def bench_heron_qa_studies_family(seed: int = _SEED + 2):
    """heron_qa_studies: synthetic correctness bench."""
    return _finite_blob(heron_qa_studies.bench_heron_qa_studies(seed))


def bench_plover_qa_studies_family(seed: int = _SEED + 3):
    """plover_qa_studies: synthetic correctness bench."""
    return _finite_blob(plover_qa_studies.bench_plover_qa_studies(seed))


def bench_sandpiper_qa_studies_family(seed: int = _SEED + 4):
    """sandpiper_qa_studies: synthetic correctness bench."""
    return _finite_blob(sandpiper_qa_studies.bench_sandpiper_qa_studies(seed))


def bench_tern_qa_studies_family(seed: int = _SEED + 5):
    """tern_qa_studies: synthetic correctness bench."""
    return _finite_blob(tern_qa_studies.bench_tern_qa_studies(seed))
