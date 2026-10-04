"""Wave-1391 bench adapters: QA-exotics-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    archer_qa_studies,
    argue_eval_studies,
    expert_qa_studies,
    mintaka_lite_studies,
    musique_lite_studies,
    wiki2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13910


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_archer_qa_studies_family(seed: int = _SEED + 0):
    """archer_qa_studies: synthetic correctness bench."""
    return _finite_blob(archer_qa_studies.bench_archer_qa_studies(seed))


def bench_argue_eval_studies_family(seed: int = _SEED + 1):
    """argue_eval_studies: synthetic correctness bench."""
    return _finite_blob(argue_eval_studies.bench_argue_eval_studies(seed))


def bench_expert_qa_studies_family(seed: int = _SEED + 2):
    """expert_qa_studies: synthetic correctness bench."""
    return _finite_blob(expert_qa_studies.bench_expert_qa_studies(seed))


def bench_mintaka_lite_studies_family(seed: int = _SEED + 3):
    """mintaka_lite_studies: synthetic correctness bench."""
    return _finite_blob(mintaka_lite_studies.bench_mintaka_lite_studies(seed))


def bench_musique_lite_studies_family(seed: int = _SEED + 4):
    """musique_lite_studies: synthetic correctness bench."""
    return _finite_blob(musique_lite_studies.bench_musique_lite_studies(seed))


def bench_wiki2_qa_studies_family(seed: int = _SEED + 5):
    """wiki2_qa_studies: synthetic correctness bench."""
    return _finite_blob(wiki2_qa_studies.bench_wiki2_qa_studies(seed))
