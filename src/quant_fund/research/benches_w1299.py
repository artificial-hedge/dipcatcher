"""Wave-1299 bench adapters: safety-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    agent_harm_studies,
    harm_bench_studies,
    jailbreak_bench_studies,
    prompt_inject_studies,
    safety_bench_studies,
    xstest_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12990


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_agent_harm_studies_family(seed: int = _SEED + 0):
    """agent_harm_studies: synthetic correctness bench."""
    return _finite_blob(agent_harm_studies.bench_agent_harm_studies(seed))


def bench_harm_bench_studies_family(seed: int = _SEED + 1):
    """harm_bench_studies: synthetic correctness bench."""
    return _finite_blob(harm_bench_studies.bench_harm_bench_studies(seed))


def bench_jailbreak_bench_studies_family(seed: int = _SEED + 2):
    """jailbreak_bench_studies: synthetic correctness bench."""
    return _finite_blob(jailbreak_bench_studies.bench_jailbreak_bench_studies(seed))


def bench_prompt_inject_studies_family(seed: int = _SEED + 3):
    """prompt_inject_studies: synthetic correctness bench."""
    return _finite_blob(prompt_inject_studies.bench_prompt_inject_studies(seed))


def bench_safety_bench_studies_family(seed: int = _SEED + 4):
    """safety_bench_studies: synthetic correctness bench."""
    return _finite_blob(safety_bench_studies.bench_safety_bench_studies(seed))


def bench_xstest_studies_family(seed: int = _SEED + 5):
    """xstest_studies: synthetic correctness bench."""
    return _finite_blob(xstest_studies.bench_xstest_studies(seed))
