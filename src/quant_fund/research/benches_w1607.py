"""Wave-1607 bench adapters: primate-4 canon (SYNTHETIC only)."""

from quant_fund.models import (
    bamboo_lemur_qa_studies,
    bearded_saki_qa_studies,
    owl_monkey_qa_studies,
    pale_titi_qa_studies,
    uakari_2_qa_studies,
    woolly_lemur_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16070


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


def bench_bamboo_lemur_qa_studies_family(seed: int = _SEED + 0):
    """bamboo_lemur_qa_studies: synthetic correctness bench."""
    return _finite_blob(bamboo_lemur_qa_studies.bench_bamboo_lemur_qa_studies(seed))


def bench_bearded_saki_qa_studies_family(seed: int = _SEED + 1):
    """bearded_saki_qa_studies: synthetic correctness bench."""
    return _finite_blob(bearded_saki_qa_studies.bench_bearded_saki_qa_studies(seed))


def bench_owl_monkey_qa_studies_family(seed: int = _SEED + 2):
    """owl_monkey_qa_studies: synthetic correctness bench."""
    return _finite_blob(owl_monkey_qa_studies.bench_owl_monkey_qa_studies(seed))


def bench_pale_titi_qa_studies_family(seed: int = _SEED + 3):
    """pale_titi_qa_studies: synthetic correctness bench."""
    return _finite_blob(pale_titi_qa_studies.bench_pale_titi_qa_studies(seed))


def bench_uakari_2_qa_studies_family(seed: int = _SEED + 4):
    """uakari_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(uakari_2_qa_studies.bench_uakari_2_qa_studies(seed))


def bench_woolly_lemur_qa_studies_family(seed: int = _SEED + 5):
    """woolly_lemur_qa_studies: synthetic correctness bench."""
    return _finite_blob(woolly_lemur_qa_studies.bench_woolly_lemur_qa_studies(seed))
