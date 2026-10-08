"""Wave-1374 bench adapters: proof-entailment canon (SYNTHETIC only)."""

from quant_fund.models import (
    deduc_lite_studies,
    entail_bank_studies,
    folio_lite_studies,
    logic_nli_studies,
    proof_writer_studies,
    rule_taker_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13740


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


def bench_deduc_lite_studies_family(seed: int = _SEED + 0):
    """deduc_lite_studies: synthetic correctness bench."""
    return _finite_blob(deduc_lite_studies.bench_deduc_lite_studies(seed))


def bench_entail_bank_studies_family(seed: int = _SEED + 1):
    """entail_bank_studies: synthetic correctness bench."""
    return _finite_blob(entail_bank_studies.bench_entail_bank_studies(seed))


def bench_folio_lite_studies_family(seed: int = _SEED + 2):
    """folio_lite_studies: synthetic correctness bench."""
    return _finite_blob(folio_lite_studies.bench_folio_lite_studies(seed))


def bench_logic_nli_studies_family(seed: int = _SEED + 3):
    """logic_nli_studies: synthetic correctness bench."""
    return _finite_blob(logic_nli_studies.bench_logic_nli_studies(seed))


def bench_proof_writer_studies_family(seed: int = _SEED + 4):
    """proof_writer_studies: synthetic correctness bench."""
    return _finite_blob(proof_writer_studies.bench_proof_writer_studies(seed))


def bench_rule_taker_studies_family(seed: int = _SEED + 5):
    """rule_taker_studies: synthetic correctness bench."""
    return _finite_blob(rule_taker_studies.bench_rule_taker_studies(seed))
