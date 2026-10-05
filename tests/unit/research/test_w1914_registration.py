"""The jinn wave must preserve existing canonical benchmark registrations."""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_CANONICAL = {
    "ifrit_qa_studies": "quant_fund.research.benches_w1634",
    "marid_qa_studies": "quant_fund.research.benches_w1634",
    "nasnas_qa_studies": "quant_fund.research.benches_w1886",
}


def test_jinn_wave_preserves_canonical_adapter_imports() -> None:
    """A repeated family must not silently switch its adapter or default seed."""
    source = (_ROOT / "src/quant_fund/research/agent.py").read_text()
    imports = [node for node in ast.parse(source).body if isinstance(node, ast.ImportFrom)]
    for family, module in _CANONICAL.items():
        symbol = f"bench_{family}_family"
        bindings = [
            node.module for node in imports if any(alias.name == symbol for alias in node.names)
        ]
        assert bindings == [module], (family, bindings)


def test_jinn_wave_registers_each_existing_family_once() -> None:
    source = (_ROOT / "src/quant_fund/research/catalog/registry.py").read_text()
    assignment = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "OPTIONAL_BENCHMARK_FAMILIES"
    )
    assert isinstance(assignment.value, ast.Call)
    families = assignment.value.args[0]
    assert isinstance(families, ast.Set)
    values = [ast.literal_eval(item) for item in families.elts]
    for family in _CANONICAL:
        assert values.count(family) == 1, family
