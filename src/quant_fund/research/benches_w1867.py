"""Wave-1867 bench adapters: celtic-remnant canon (SYNTHETIC only)."""

from quant_fund.models import (
    antenociticus_qa_studies,
    ares_lusitani_qa_studies,
    braciaca_qa_studies,
    deiba_qa_studies,
    nantosuelta_qa_studies,
    ognios_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18670


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_antenociticus_qa_studies_family(seed: int = _SEED + 0):
    """antenociticus_qa_studies: synthetic correctness bench."""
    return _finite_blob(antenociticus_qa_studies.bench_antenociticus_qa_studies(seed))


def bench_ares_lusitani_qa_studies_family(seed: int = _SEED + 1):
    """ares_lusitani_qa_studies: synthetic correctness bench."""
    return _finite_blob(ares_lusitani_qa_studies.bench_ares_lusitani_qa_studies(seed))


def bench_braciaca_qa_studies_family(seed: int = _SEED + 2):
    """braciaca_qa_studies: synthetic correctness bench."""
    return _finite_blob(braciaca_qa_studies.bench_braciaca_qa_studies(seed))


def bench_deiba_qa_studies_family(seed: int = _SEED + 3):
    """deiba_qa_studies: synthetic correctness bench."""
    return _finite_blob(deiba_qa_studies.bench_deiba_qa_studies(seed))


def bench_nantosuelta_qa_studies_family(seed: int = _SEED + 4):
    """nantosuelta_qa_studies: synthetic correctness bench."""
    return _finite_blob(nantosuelta_qa_studies.bench_nantosuelta_qa_studies(seed))


def bench_ognios_qa_studies_family(seed: int = _SEED + 5):
    """ognios_qa_studies: synthetic correctness bench."""
    return _finite_blob(ognios_qa_studies.bench_ognios_qa_studies(seed))
