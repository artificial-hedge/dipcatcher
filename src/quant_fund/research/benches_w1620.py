"""Wave-1620 bench adapters: venom-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    boomslang_qa_studies,
    death_adder_qa_studies,
    gaboon_qa_studies,
    inland_taipan_qa_studies,
    saw_scaled_qa_studies,
    sea_krait_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16200


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_boomslang_qa_studies_family(seed: int = _SEED + 0):
    """boomslang_qa_studies: synthetic correctness bench."""
    return _finite_blob(boomslang_qa_studies.bench_boomslang_qa_studies(seed))


def bench_death_adder_qa_studies_family(seed: int = _SEED + 1):
    """death_adder_qa_studies: synthetic correctness bench."""
    return _finite_blob(death_adder_qa_studies.bench_death_adder_qa_studies(seed))


def bench_gaboon_qa_studies_family(seed: int = _SEED + 2):
    """gaboon_qa_studies: synthetic correctness bench."""
    return _finite_blob(gaboon_qa_studies.bench_gaboon_qa_studies(seed))


def bench_inland_taipan_qa_studies_family(seed: int = _SEED + 3):
    """inland_taipan_qa_studies: synthetic correctness bench."""
    return _finite_blob(inland_taipan_qa_studies.bench_inland_taipan_qa_studies(seed))


def bench_saw_scaled_qa_studies_family(seed: int = _SEED + 4):
    """saw_scaled_qa_studies: synthetic correctness bench."""
    return _finite_blob(saw_scaled_qa_studies.bench_saw_scaled_qa_studies(seed))


def bench_sea_krait_qa_studies_family(seed: int = _SEED + 5):
    """sea_krait_qa_studies: synthetic correctness bench."""
    return _finite_blob(sea_krait_qa_studies.bench_sea_krait_qa_studies(seed))
