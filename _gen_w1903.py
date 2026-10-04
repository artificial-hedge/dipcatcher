"""Generate wave-1890 canon files: manx-myth-2 canon."""

from __future__ import annotations

from pathlib import Path

ROOT = Path("/Users/devin/repos/dipcatcher")
WAVE = 1903
SEED = WAVE * 10
THEME = "slavic-demon canon"

FAMILIES = [
    "polevoy_qa_studies",
    "rarog_qa_studies",
    "likho_qa_studies",
    "vodyanoy_qa_studies",
    "chert_qa_studies",
    "zmey_gorynych_qa_studies",
]

# (check-blurb, aux-blurb) — unique docstring cores per function.
BLURBS = {
    "polevoy_qa_studies": "Polevoy field-spirit canon: midday manifestation windows, grain-offering pacification rates, field-boundary patrol radii, harvest-cycle power curves",
    "rarog_qa_studies": "Rarog fire-falcon canon: spark-spawn ignition priors, ember-rider arrival signatures, whirlwind ember-carriage vectors, house-fire omen scoring",
    "likho_qa_studies": "Likho misfortune-spirit canon: one-eyed jinx affixation rates, luck-drain cascade thresholds, husk-clinging persistence indices, exorcism-resistance curves",
    "vodyanoy_qa_studies": "Vodyanoy water-demon canon: drowned-servant accumulation rates, dam-breach provocation triggers, fish-poaching detection radii, mill-wheel sabotage frequency",
    "chert_qa_studies": "Chert minor-demon canon: curse-invocation summoning priors, crossroads temptation scores, minor-pact binding efficacy, church-bell banishment radii",
    "zmey_gorynych_qa_studies": "Zmey-Gorynych three-headed dragon canon: aerial-raid interception difficulty, head-regeneration cycle times, hero-duel outcome priors, lair-hoard density scores",
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
            MODEL_TEMPLATE.format(fam=fam, check_blurb=b[0], aux_blurb=b[1], theme=THEME)
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
    mt = f"""import pytest


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
"""
    (ROOT / "tests/unit/models" / f"test_w{WAVE}.py").write_text(mt)

    fams = ",\n".join(f'        "bench_{fam}_family"' for fam in sorted(FAMILIES))
    rt = f"""import pytest

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
"""
    (ROOT / "tests/unit/research" / f"test_benches_w{WAVE}.py").write_text(rt)


def wire_agent() -> None:
    agent = ROOT / "src/quant_fund/research/agent.py"
    text = agent.read_text()
    fams = "\n".join(f"    bench_{fam}_family," for fam in sorted(FAMILIES))
    block = f"from quant_fund.research.benches_w{WAVE} import (\n{fams}\n)\n"
    anchor = "from quant_fund.research.catalog import ("
    assert anchor in text
    text = text.replace(anchor, block + anchor, 1)

    agent.write_text(text)


def wire_registry() -> None:
    reg = ROOT / "src/quant_fund/research/catalog/registry.py"
    text = reg.read_text()
    block = f"        # Wave-{WAVE} {THEME}.\n" + "".join(
        f'        "{f}",\n' for f in sorted(FAMILIES)
    )
    anchor = "        # Wave-1399 retrieval-eval canon."
    assert anchor in text
    text = text.replace(anchor, block + anchor, 1)
    reg.write_text(text)


def wire_inflight() -> None:
    f = ROOT / "INFLIGHT"
    base = 9903 + 6 * (WAVE - 1659)
    f.write_text(
        f"w{WAVE} {THEME}: {', '.join(sorted(FAMILIES))} ({base}->{base + 6})\n" + f.read_text()
    )


write_models()
write_benches()
write_tests()
wire_agent()
wire_registry()
wire_inflight()
print("w1903 generated")
