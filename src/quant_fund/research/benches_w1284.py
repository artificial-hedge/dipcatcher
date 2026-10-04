"""Wave-1284 bench adapters: grounding/hallucination canon (SYNTHETIC only)."""

from quant_fund.models import (
    citation_check_studies,
    claim_verifier_studies,
    entailment_studies,
    factuality_score_studies,
    grounding_verify_studies,
    self_reflect_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12840


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_citation_check_studies_family(seed: int = _SEED + 0):
    """citation_check_studies: synthetic correctness bench."""
    return _finite_blob(citation_check_studies.bench_citation_check_studies(seed))


def bench_claim_verifier_studies_family(seed: int = _SEED + 1):
    """claim_verifier_studies: synthetic correctness bench."""
    return _finite_blob(claim_verifier_studies.bench_claim_verifier_studies(seed))


def bench_entailment_studies_family(seed: int = _SEED + 2):
    """entailment_studies: synthetic correctness bench."""
    return _finite_blob(entailment_studies.bench_entailment_studies(seed))


def bench_factuality_score_studies_family(seed: int = _SEED + 3):
    """factuality_score_studies: synthetic correctness bench."""
    return _finite_blob(factuality_score_studies.bench_factuality_score_studies(seed))


def bench_grounding_verify_studies_family(seed: int = _SEED + 4):
    """grounding_verify_studies: synthetic correctness bench."""
    return _finite_blob(grounding_verify_studies.bench_grounding_verify_studies(seed))


def bench_self_reflect_studies_family(seed: int = _SEED + 5):
    """self_reflect_studies: synthetic correctness bench."""
    return _finite_blob(self_reflect_studies.bench_self_reflect_studies(seed))
