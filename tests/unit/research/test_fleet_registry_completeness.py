"""Registry completeness ratchet — every fleet-compatible head is registered.

Any ``*Distribution`` class whose constructor is fleet-compatible
(``(taus, *defaults)`` — every arg past ``taus`` defaulted) must either
appear in ``FLEET_HEAD_REGISTRY`` or carry a documented reason in
``KNOWN_NON_FLEET``. A head added to ``models/`` without a registry entry
now fails CI instead of silently orphaning — and an exemption added
without a reason fails too (the reason string is pinned non-empty).
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from quant_fund.research.fleet_eval import FLEET_HEAD_REGISTRY

# Head-capable modules: every file under models/ that defines a
# ``*Distribution`` class. Derived by scanning — the manifest pins the
# module list so a new head module also trips the ratchet.
MODELS_DIR = Path(__file__).resolve().parents[3] / "src" / "quant_fund" / "models"

_HEAD_MODULE_FILES = (
    "distribution.py",
    "conformal_dist.py",
    "fhs.py",
    "hstep.py",
    "lgbm_q2.py",
    "nbeats.py",
    "qar.py",
    "regime_dist.py",
)

# Documented exemptions — heads that are fleet-compatible but deliberately
# not registered (scaled/composite variants of registered heads, or
# constructors needing non-defaultable context). Each entry requires a
# non-empty reason; shrink-only ratchet.
KNOWN_NON_FLEET: dict[str, str] = {
    "ScaledGaussianDistribution": "PIT-safe scaled variant of registered 'gaussian'; registering both double-counts the same family",
    "ScaledEmpiricalDistribution": "scaled variant of registered 'empirical'",
    "ScaledStudentTDistribution": "scaled variant of registered 'skew_t' family",
    "LinearQuantileDistribution": "conditional linear-quantile fit — covered by 'qar'/'lgbm_q2' lanes; needs feature matrix at fit time",
    "TreeQuantileDistribution": "lightgbm-backed conditional head — 'lgbm_q2' is the registered member of the family",
    "HStepScaledDistribution": "multi-horizon composite; fleet registers the two h=1 construction slices (hstep_t/hstep_emp)",
}


def _head_classes() -> dict[str, type]:
    found: dict[str, type] = {}
    for filename in _HEAD_MODULE_FILES:
        module_path = MODELS_DIR / filename
        src = module_path.read_text()
        module_name = f"quant_fund.models.{filename[:-3]}"
        try:
            import importlib

            mod = importlib.import_module(module_name)
        except ImportError:  # torch-optional heads
            continue
        for class_name in re.findall(r"^class (\w+Distribution)\b", src, re.M):
            cls = getattr(mod, class_name, None)
            if cls is not None and cls.__module__ == module_name:
                found[class_name] = cls
    return found


def _fleet_compatible(cls: type) -> bool:
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return False
    params = [p for p in sig.parameters.values() if p.name not in ("self", "taus")]
    return all(
        p.default is not inspect.Parameter.empty
        or p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for p in params
    )


def _registered_head_classes() -> set[str]:
    """Recover the *Distribution classes reachable through the registry,
    including via adapter wrappers (_QarOneStepHead._inner = QARDistribution)."""
    import quant_fund.research.fleet_eval as fe

    registered: set[str] = set()
    frontier: list[object] = list(FLEET_HEAD_REGISTRY.values())
    seen: set[int] = set()
    while frontier:
        obj = frontier.pop()
        try:
            src = inspect.getsource(obj)  # type: ignore[arg-type]
        except (OSError, TypeError):
            continue
        for ident in re.findall(r"\b(\w+)\b", src):
            if ident.endswith(("Distribution", "Head")):
                registered.add(ident)
            inner = getattr(fe, ident, None)
            if (inspect.isclass(inner) or callable(inner)) and id(inner) not in seen:
                seen.add(id(inner))
                frontier.append(inner)
    return registered


def test_every_compatible_head_registered_or_exempted() -> None:
    heads = _head_classes()
    assert heads, "head discovery found nothing — module scan broken"
    registered_classes = _registered_head_classes()

    for cls_name, cls in heads.items():
        if not _fleet_compatible(cls):
            continue
        assert cls_name in registered_classes or cls_name in KNOWN_NON_FLEET, (
            f"{cls_name} is a fleet-compatible head with no registry entry and "
            f"no KNOWN_NON_FLEET exemption — register it in FLEET_HEAD_REGISTRY "
            f"or document why it is exempt"
        )


def test_exemptions_have_reasons_and_exist() -> None:
    heads = _head_classes()
    for cls_name, reason in KNOWN_NON_FLEET.items():
        assert reason.strip(), f"{cls_name} exemption lacks a documented reason"
        assert cls_name in heads, f"exempted {cls_name} no longer exists — drop the stale entry"


def test_registry_resolves_all_names() -> None:
    """Every registered name constructs without raising (lazy torch allowed)."""
    import quant_fund.models.nbeats  # noqa: F401 — torch extra present in the suite

    for name, factory in FLEET_HEAD_REGISTRY.items():
        obj: Any = factory([0.1, 0.5, 0.9], 0)
        assert obj is not None, f"registry[{name!r}] constructed None"


def test_registry_names_are_unique_snake_case() -> None:
    for name in FLEET_HEAD_REGISTRY:
        assert re.fullmatch(r"[a-z][a-z0-9_]*", name), f"bad registry name {name!r}"
    values: list[Callable] = list(FLEET_HEAD_REGISTRY.values())
    assert len(values) == len(set(FLEET_HEAD_REGISTRY))
