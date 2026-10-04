"""Wave-1274 bench adapters: agent-infrastructure canon (SYNTHETIC only)."""

from quant_fund.models import (
    agent_memory_studies,
    code_agent_studies,
    computer_use_studies,
    mcp_protocol_studies,
    skill_library_studies,
    web_agent_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12740


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_agent_memory_studies_family(seed: int = _SEED + 0):
    """agent_memory_studies: synthetic correctness bench."""
    return _finite_blob(agent_memory_studies.bench_agent_memory_studies(seed))


def bench_code_agent_studies_family(seed: int = _SEED + 1):
    """code_agent_studies: synthetic correctness bench."""
    return _finite_blob(code_agent_studies.bench_code_agent_studies(seed))


def bench_computer_use_studies_family(seed: int = _SEED + 2):
    """computer_use_studies: synthetic correctness bench."""
    return _finite_blob(computer_use_studies.bench_computer_use_studies(seed))


def bench_mcp_protocol_studies_family(seed: int = _SEED + 3):
    """mcp_protocol_studies: synthetic correctness bench."""
    return _finite_blob(mcp_protocol_studies.bench_mcp_protocol_studies(seed))


def bench_skill_library_studies_family(seed: int = _SEED + 4):
    """skill_library_studies: synthetic correctness bench."""
    return _finite_blob(skill_library_studies.bench_skill_library_studies(seed))


def bench_web_agent_studies_family(seed: int = _SEED + 5):
    """web_agent_studies: synthetic correctness bench."""
    return _finite_blob(web_agent_studies.bench_web_agent_studies(seed))
