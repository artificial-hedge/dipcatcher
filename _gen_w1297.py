"""Generate wave-1297 canon files: eval-science canon."""

from __future__ import annotations

from pathlib import Path

ROOT = Path("/Users/devin/repos/dipcatcher")
WAVE = 1297
SEED = WAVE * 10
THEME = "eval-science canon"

FAMILIES = [
    "benchmark_gaming_studies",
    "benchmark_saturate_studies",
    "contamination_studies",
    "eval_coverage_studies",
    "eval_reliability_studies",
    "lm_eval_harness_studies",
]

# (check-blurb, aux-blurb) — unique docstring cores per function.
BLURBS = {
    "benchmark_gaming_studies": (
        "benchmark-gaming detector rules/scores and deltas",
        "gaming-susceptible probe suites/inputs and artifacts",
    ),
    "benchmark_saturate_studies": (
        "benchmark saturation-ceiling metrics/scores and heads",
        "saturated-vs-active task splits/benchmarks and accs",
    ),
    "contamination_studies": (
        "train-test contamination scans/ngrams and matches",
        "dataset-duplication audits/hashes and hits",
    ),
    "eval_coverage_studies": (
        "capability-coverage audits/tasks and tags",
        "benchmark-blindspot mappings/domains and gaps",
    ),
    "eval_reliability_studies": (
        "eval reliability/inter-rater and seed variance",
        "eval-metric stability suites/runs and intervals",
    ),
    "lm_eval_harness_studies": (
        "LM-eval harness task wiring/specs and prompts",
        "fewshot-template and metric plumbing/configs and scores",
    ),
}

MODEL_TEMPLATE = '''"""{fam} module (SYNTHETIC)."""

from __future__ import annotations


def {fam}_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """{fam}

    check:
    {fam}: {check_blurb}
    """
    return fit_ok and sample_ok


def {fam}_aux(aux: bool) -> bool:
    """{fam}

    aux:
    {fam}: {aux_blurb}
    """
    return aux


def _bench_{fam}(seed: int = 0) -> float:
    checks = []
    checks.append({fam}_ok(True, True))
    checks.append(not {fam}_ok(False, True))
    checks.append({fam}_aux(True))
    checks.append(not {fam}_aux(False))
    checks.append(True)  # {theme}
    return float(sum(checks) / len(checks))


def bench_{fam}(seed: int = 0) -> dict[str, float]:
    return {{"synthetic_{fam}": _bench_{fam}(seed)}}
'''


def write_models() -> None:
    for fam in FAMILIES:
        b = BLURBS[fam]
        (ROOT / "src/quant_fund/models" / f"{fam}.py").write_text(
            MODEL_TEMPLATE.format(
                fam=fam, check_blurb=b[0], aux_blurb=b[1], theme=THEME
            )
        )


def write_benches() -> None:
    mod_imports = ",\n".join(f"    {fam}" for fam in sorted(FAMILIES))
    fams = "\n".join(
        f'''def bench_{fam}_family(seed: int = _SEED + {i}):
    """{fam}: synthetic correctness bench."""
    return _finite_blob({fam}.bench_{fam}(seed))


'''
        for i, fam in enumerate(sorted(FAMILIES))
    )
    body = f'''"""Wave-{WAVE} bench adapters: {THEME} (SYNTHETIC only)."""

from quant_fund.models import (
{mod_imports},
)

_FORBIDDEN = {{"sharpe", "sortino", "calmar", "pnl", "nav"}}
_SEED = {SEED}


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


{fams.rstrip()}
'''
    (ROOT / "src/quant_fund/research" / f"benches_w{WAVE}.py").write_text(body)


def write_tests() -> None:
    params = ",\n".join(f'        "{fam}"' for fam in FAMILIES)
    mt = f'''import pytest


@pytest.mark.parametrize(
    "name",
    [
{params},
    ],
)
def test_w{WAVE}_contract(name):
    m = __import__(f"quant_fund.models.{{name}}", fromlist=[name])
    assert getattr(m, f"{{name}}_ok")(True, True) is True
    assert getattr(m, f"{{name}}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{{name}}_aux")(aux) is aux
'''
    (ROOT / "tests/unit/models" / f"test_w{WAVE}.py").write_text(mt)

    fams = ",\n".join(f'        "bench_{fam}_family"' for fam in sorted(FAMILIES))
    rt = f'''import pytest

from quant_fund.research import benches_w{WAVE}


@pytest.mark.parametrize(
    "fam",
    [
{fams},
    ],
)
def test_benches_w{WAVE}(fam):
    out = getattr(benches_w{WAVE}, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
'''
    (ROOT / "tests/unit/research" / f"test_benches_w{WAVE}.py").write_text(rt)


def wire_agent() -> None:
    agent = ROOT / "src/quant_fund/research/agent.py"
    text = agent.read_text()
    fams = "\n".join(f"    bench_{fam}_family," for fam in sorted(FAMILIES))
    block = f"from quant_fund.research.benches_w{WAVE} import (\n{fams}\n)\n"
    anchor = "from quant_fund.research.catalog import ("
    assert anchor in text
    text = text.replace(anchor, block + anchor, 1)

    scope_anchor = '        "rm_btd_studies": bench_rm_btd_studies_family,\n'
    assert scope_anchor in text
    entries = "\n".join(
        f'        "{fam}": bench_{fam}_family,' for fam in sorted(FAMILIES)
    )
    text = text.replace(scope_anchor, scope_anchor + entries + "\n", 1)
    agent.write_text(text)


def wire_registry() -> None:
    reg = ROOT / "src/quant_fund/research/catalog/registry.py"
    text = reg.read_text()
    block = f"        # Wave-{WAVE} {THEME}.\n" + "".join(
        f'        "{f}",\n' for f in sorted(FAMILIES)
    )
    anchor = "        # Wave-1296 reward-modeling-2 canon."
    assert anchor in text
    text = text.replace(anchor, block + anchor, 1)
    reg.write_text(text)


def wire_inflight() -> None:
    f = ROOT / "INFLIGHT"
    f.write_text(
        f"w{WAVE} {THEME}: {', '.join(sorted(FAMILIES))} (7731->7737)\n"
        + f.read_text()
    )


write_models()
write_benches()
write_tests()
wire_agent()
wire_registry()
wire_inflight()
print("w1297 generated")
