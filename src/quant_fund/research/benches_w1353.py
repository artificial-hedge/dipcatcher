"""Wave-1353 bench adapters: NLI-eval-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    creak_lite_studies,
    entailment_bn_studies,
    hans_lite_studies,
    prove_it_studies,
    strategy_qa_studies,
    sup_nli_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13530


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_creak_lite_studies_family(seed: int = _SEED + 0):
    """creak_lite_studies: synthetic correctness bench."""
    return _finite_blob(creak_lite_studies.bench_creak_lite_studies(seed))


def bench_entailment_bn_studies_family(seed: int = _SEED + 1):
    """entailment_bn_studies: synthetic correctness bench."""
    return _finite_blob(entailment_bn_studies.bench_entailment_bn_studies(seed))


def bench_hans_lite_studies_family(seed: int = _SEED + 2):
    """hans_lite_studies: synthetic correctness bench."""
    return _finite_blob(hans_lite_studies.bench_hans_lite_studies(seed))


def bench_prove_it_studies_family(seed: int = _SEED + 3):
    """prove_it_studies: synthetic correctness bench."""
    return _finite_blob(prove_it_studies.bench_prove_it_studies(seed))


def bench_strategy_qa_studies_family(seed: int = _SEED + 4):
    """strategy_qa_studies: synthetic correctness bench."""
    return _finite_blob(strategy_qa_studies.bench_strategy_qa_studies(seed))


def bench_sup_nli_studies_family(seed: int = _SEED + 5):
    """sup_nli_studies: synthetic correctness bench."""
    return _finite_blob(sup_nli_studies.bench_sup_nli_studies(seed))
