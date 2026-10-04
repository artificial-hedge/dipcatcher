"""Wave-1480 bench adapters: reptile canon (SYNTHETIC only)."""

from quant_fund.models import (
    adder_qa_studies,
    boa_qa_studies,
    krait_qa_studies,
    mamba_qa_studies,
    monitor_qa_studies,
    taipan_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14800


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_adder_qa_studies_family(seed: int = _SEED + 0):
    """adder_qa_studies: synthetic correctness bench."""
    return _finite_blob(adder_qa_studies.bench_adder_qa_studies(seed))


def bench_boa_qa_studies_family(seed: int = _SEED + 1):
    """boa_qa_studies: synthetic correctness bench."""
    return _finite_blob(boa_qa_studies.bench_boa_qa_studies(seed))


def bench_krait_qa_studies_family(seed: int = _SEED + 2):
    """krait_qa_studies: synthetic correctness bench."""
    return _finite_blob(krait_qa_studies.bench_krait_qa_studies(seed))


def bench_mamba_qa_studies_family(seed: int = _SEED + 3):
    """mamba_qa_studies: synthetic correctness bench."""
    return _finite_blob(mamba_qa_studies.bench_mamba_qa_studies(seed))


def bench_monitor_qa_studies_family(seed: int = _SEED + 4):
    """monitor_qa_studies: synthetic correctness bench."""
    return _finite_blob(monitor_qa_studies.bench_monitor_qa_studies(seed))


def bench_taipan_qa_studies_family(seed: int = _SEED + 5):
    """taipan_qa_studies: synthetic correctness bench."""
    return _finite_blob(taipan_qa_studies.bench_taipan_qa_studies(seed))
