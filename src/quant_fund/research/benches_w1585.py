"""Wave-1585 bench adapters: insectivore canon (SYNTHETIC only)."""

from quant_fund.models import (
    aardvark_qa_studies,
    elephant_shrew_qa_studies,
    golden_mole_qa_studies,
    gymnure_qa_studies,
    solenodon_qa_studies,
    tenrec_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15850


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_aardvark_qa_studies_family(seed: int = _SEED + 0):
    """aardvark_qa_studies: synthetic correctness bench."""
    return _finite_blob(aardvark_qa_studies.bench_aardvark_qa_studies(seed))


def bench_elephant_shrew_qa_studies_family(seed: int = _SEED + 1):
    """elephant_shrew_qa_studies: synthetic correctness bench."""
    return _finite_blob(elephant_shrew_qa_studies.bench_elephant_shrew_qa_studies(seed))


def bench_golden_mole_qa_studies_family(seed: int = _SEED + 2):
    """golden_mole_qa_studies: synthetic correctness bench."""
    return _finite_blob(golden_mole_qa_studies.bench_golden_mole_qa_studies(seed))


def bench_gymnure_qa_studies_family(seed: int = _SEED + 3):
    """gymnure_qa_studies: synthetic correctness bench."""
    return _finite_blob(gymnure_qa_studies.bench_gymnure_qa_studies(seed))


def bench_solenodon_qa_studies_family(seed: int = _SEED + 4):
    """solenodon_qa_studies: synthetic correctness bench."""
    return _finite_blob(solenodon_qa_studies.bench_solenodon_qa_studies(seed))


def bench_tenrec_qa_studies_family(seed: int = _SEED + 5):
    """tenrec_qa_studies: synthetic correctness bench."""
    return _finite_blob(tenrec_qa_studies.bench_tenrec_qa_studies(seed))
