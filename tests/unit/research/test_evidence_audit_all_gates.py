"""Lock the evidence-audit fail-at-end recipe: every tail gate must run."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MAKEFILE = REPO_ROOT / "Makefile"
CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"

TAIL_GATES = (
    "crown-jewels",
    "verify-witness",
    "verify-repo",
    "checkpoint-chain",
    "verify-rotations",
    "tamper-drill",
    "fuzz-drill",
)


def _evidence_audit_recipe() -> str:
    text = MAKEFILE.read_text(encoding="utf-8")
    match = re.search(
        r"^evidence-audit:.*?(?=^[a-zA-Z0-9_-]+:|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, "Makefile must declare evidence-audit"
    return match.group(0)


def test_evidence_audit_runs_all_seven_tail_gates_then_fails_closed() -> None:
    recipe = _evidence_audit_recipe()
    for gate in TAIL_GATES:
        assert f"_ea {gate} " in recipe, gate
    assert recipe.find("_ea crown-jewels") < recipe.find("_ea fuzz-drill")
    assert "ea_fail=1" in recipe
    assert "one or more gates failed" in recipe
    assert 'if [ "$$ea_fail" -ne 0 ]' in recipe or 'if [ "$ea_fail" -ne 0 ]' in recipe


def test_ci_evidence_base_prefers_last_green_artifact() -> None:
    text = CI_YML.read_text(encoding="utf-8")
    start = text.index("  evidence-audit:")
    end = text.find("\n  pretrade-risk:", start)
    if end < 0:
        end = len(text)
    job = text[start:end]
    assert "github.event.before" in job
    assert "LAST_GREEN" in job
    assert "evidence-base-sha" in job
    assert "quant_fund.research.evidence_base" in job
    assert "persist-credentials: false" in job
