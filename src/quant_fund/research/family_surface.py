"""Read-only qualified family surface and selective runners.

``/superpower`` must choose known, qualified, non-retired families that have
an implemented runner, then execute only that subset. Catalog membership is
not permission. This module lives on the research side of the architecture
boundary so the model package never imports the catalog.

A selective run is a partial record. It is not a full research receipt and
``verified`` stays false until ``verify-research`` accepts one.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

SURFACE_SCHEMA = "dipcatcher.family_surface.v1"
RUN_SCHEMA = "dipcatcher.family_run.v1"
_FAMILY_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_GOAL_TOKEN = re.compile(r"[a-z][a-z0-9_]*")

Runner = Callable[..., Any]


class FamilySurfaceError(ValueError):
    """Fail-closed refusal: unknown, retired, unqualified, or unsafe ids."""


@dataclass(frozen=True)
class PanelBundle:
    """Panel inputs a selective runner may use. Missing market data is not filled."""

    frame: Any
    label: str
    northset: Any


@dataclass(frozen=True)
class FamilyRunner:
    """One implemented bench the selective surface is allowed to call."""

    family_id: str
    purpose: str
    tasks: tuple[str, ...]
    trigger_terms: tuple[str, ...]
    scores: tuple[str, ...]
    required_inputs: tuple[str, ...]
    needs: str
    target: str
    resource_limit_s: int


def _runner(
    family_id: str,
    purpose: str,
    tasks: tuple[str, ...],
    trigger_terms: tuple[str, ...],
    scores: tuple[str, ...],
    required_inputs: tuple[str, ...],
    needs: str,
    target: str,
    resource_limit_s: int = 120,
) -> FamilyRunner:
    return FamilyRunner(
        family_id=family_id,
        purpose=purpose,
        tasks=tasks,
        trigger_terms=trigger_terms,
        scores=scores,
        required_inputs=required_inputs,
        needs=needs,
        target=target,
        resource_limit_s=resource_limit_s,
    )


# REQUIRED scorecard families with a real bench callable. Live-optional names
# stay visible to status lookups and are refused here until a runner is added.
RUNNERS: dict[str, FamilyRunner] = {
    spec.family_id: spec
    for spec in (
        _runner(
            "ranking",
            "Cross-sectional ranking diagnostics on the research panel.",
            ("ranking",),
            ("ranking", "rankic", "cross-section", "cross sectional"),
            ("mean_ic", "rank_ic"),
            ("panel", "ranking target"),
            "panel_config_label",
            "quant_fund.research.benches:bench_ranking",
            180,
        ),
        _runner(
            "alpha",
            "Cross-sectional alpha diagnostics on the research panel.",
            ("alpha",),
            ("alpha", "factor premium"),
            ("proper scores",),
            ("panel", "ranking target"),
            "panel_config_label",
            "quant_fund.research.benches:bench_alpha",
        ),
        _runner(
            "volatility",
            "Variance-forecast diagnostics scored with QLIKE.",
            ("volatility",),
            ("volatility", "qlike", "variance forecast", "vol forecast"),
            ("qlike",),
            ("panel",),
            "panel_config",
            "quant_fund.research.benches:bench_volatility",
        ),
        _runner(
            "distribution",
            "Distributional forecast diagnostics.",
            ("distribution",),
            ("distribution", "crps", "density forecast"),
            ("crps",),
            ("panel",),
            "panel_config",
            "quant_fund.research.benches:bench_distribution",
        ),
        _runner(
            "regime",
            "Regime-state diagnostics.",
            ("regime",),
            ("regime", "hmm", "regime switch"),
            ("likelihood",),
            ("panel",),
            "panel_config",
            "quant_fund.research.benches:bench_regime",
        ),
        _runner(
            "tail",
            "Tail-risk diagnostics.",
            ("tail",),
            ("tail risk", "expected shortfall", "evt"),
            ("tail score",),
            ("panel",),
            "panel_config",
            "quant_fund.research.benches:bench_tail",
        ),
        _runner(
            "drawdown",
            "Drawdown-path diagnostics.",
            ("drawdown",),
            ("drawdown",),
            ("drawdown score",),
            ("panel",),
            "panel_config",
            "quant_fund.research.benches:bench_drawdown",
        ),
        _runner(
            "liquidity",
            "Liquidity diagnostics on the research panel.",
            ("liquidity",),
            ("liquidity", "amihud"),
            ("liquidity score",),
            ("panel",),
            "panel",
            "quant_fund.research.benches:bench_liquidity",
        ),
        _runner(
            "reinforcement",
            "Contextual-bandit policy diagnostics.",
            ("reinforcement",),
            ("reinforcement", "linucb", "contextual bandit"),
            ("regret",),
            ("panel", "ranking target"),
            "panel_label",
            "quant_fund.research.benches:bench_rl",
        ),
        _runner(
            "conformal",
            "Conformal prediction-interval diagnostics.",
            ("conformal",),
            ("conformal", "prediction interval"),
            ("coverage", "mean_width"),
            ("panel",),
            "panel_config",
            "quant_fund.research.benches:bench_conformal",
        ),
        _runner(
            "evalues",
            "E-value and e-process diagnostics.",
            ("e-values",),
            ("e-value", "evalue", "e-values"),
            ("e_process",),
            ("panel",),
            "panel_config",
            "quant_fund.research.benches:bench_evalues",
        ),
        _runner(
            "jackknife_plus",
            "Jackknife+ interval diagnostics.",
            ("jackknife+",),
            ("jackknife",),
            ("coverage", "mean_width"),
            ("panel",),
            "panel_config",
            "quant_fund.research.benches:bench_jackknife_plus",
        ),
        _runner(
            "crc",
            "Conformal risk-control diagnostics.",
            ("crc",),
            ("conformal risk", "crc"),
            ("risk",),
            ("panel",),
            "panel_config",
            "quant_fund.research.benches:bench_crc",
        ),
        _runner(
            "weighted_conformal",
            "Weighted conformal diagnostics.",
            ("weighted conformal",),
            ("weighted conformal", "covariate shift"),
            ("coverage",),
            ("panel",),
            "panel_config",
            "quant_fund.research.benches:bench_weighted_conformal",
        ),
        _runner(
            "interval_risk",
            "Interval-cap risk diagnostics.",
            ("interval risk",),
            ("interval risk", "interval cap"),
            ("mean_cap",),
            ("panel",),
            "panel_config",
            "quant_fund.research.benches:bench_interval_risk",
        ),
        _runner(
            "quantile_bandit",
            "Quantile bandit diagnostics.",
            ("quantile bandit",),
            ("quantile bandit",),
            ("regret",),
            ("panel", "ranking target"),
            "panel_label",
            "quant_fund.research.benches:bench_quantile_bandit",
        ),
        _runner(
            "cv_plus",
            "CV+ interval diagnostics.",
            ("cv+",),
            ("cv+", "cv plus"),
            ("coverage",),
            ("panel",),
            "panel_config",
            "quant_fund.research.benches:bench_cv_plus",
        ),
        _runner(
            "cpcv",
            "Combinatorial purged cross-validation audit.",
            ("cpcv",),
            ("cpcv", "purged cv", "combinatorial purged"),
            ("fold_separation",),
            (),
            "none",
            "quant_fund.research.benches:bench_cpcv_audit",
        ),
        _runner(
            "localized_conformal",
            "Localized conformal diagnostics.",
            ("localized conformal",),
            ("localized conformal",),
            ("coverage",),
            ("panel",),
            "panel_config",
            "quant_fund.research.benches:bench_localized_from_panel",
        ),
        _runner(
            "conformal_rank",
            "Conformal top-k ranking diagnostics.",
            ("conformal rank",),
            ("conformal rank", "top-k", "topk"),
            ("coverage",),
            ("panel",),
            "panel_config",
            "quant_fund.research.benches:bench_conformal_topk_from_panel",
        ),
        _runner(
            "online_crc",
            "Online conformal risk-control diagnostics.",
            ("online crc",),
            ("online crc", "online conformal"),
            ("risk",),
            ("panel",),
            "panel_config",
            "quant_fund.research.benches:bench_online_crc_from_panel",
        ),
        _runner(
            "portfolio_conformal",
            "Portfolio conformal-set diagnostics.",
            ("portfolio conformal",),
            ("portfolio conformal",),
            ("coverage",),
            ("panel",),
            "panel_config",
            "quant_fund.research.benches:bench_portfolio_from_panel",
        ),
        _runner(
            "northset",
            "Order-book and candlestick identity diagnostics.",
            ("northset",),
            ("northset", "order book", "vpin"),
            ("identity",),
            ("silver bars",),
            "northset",
            "quant_fund.northset.benches:bench_northset",
            180,
        ),
    )
}


def _catalog() -> tuple[frozenset[str], Mapping[str, Any], frozenset[str], frozenset[str]]:
    from quant_fund.research.catalog import registry

    return (
        frozenset(registry.REQUIRED_BENCHMARK_FAMILIES),
        registry.RETIRED_BENCHMARK_FAMILIES,
        frozenset(registry.LIVE_OPTIONAL_BENCHMARK_FAMILIES),
        frozenset(registry.OPTIONAL_BENCHMARK_FAMILIES),
    )


def _check_id(family_id: str) -> str:
    if not isinstance(family_id, str) or _FAMILY_ID.fullmatch(family_id) is None:
        raise FamilySurfaceError(
            f"refusing family id {family_id!r}; ids must match {_FAMILY_ID.pattern}"
        )
    return family_id


def qualification_of(family_id: str) -> str:
    """Return required, live_optional, retired, or unknown. Unknown is not eligible."""
    family_id = _check_id(family_id)
    required, retired, live, optional = _catalog()
    if family_id in required:
        return "required"
    if family_id in retired or family_id in optional - live:
        return "retired"
    if family_id in live:
        return "live_optional"
    return "unknown"


def _retirement_reason(family_id: str) -> str | None:
    _required, retired, _live, _optional = _catalog()
    record = retired.get(family_id) if isinstance(retired, Mapping) else None
    if isinstance(record, Mapping):
        reason = record.get("reason")
        return str(reason) if reason else None
    if qualification_of(family_id) == "retired":
        return "retired from the live scorecard; historical receipts remain verifiable"
    return None


def _order_index() -> dict[str, int]:
    from quant_fund.research.catalog import registry

    return {name: index for index, name in enumerate(registry.BENCHMARK_FAMILY_ORDER)}


def family_contract(family_id: str) -> dict[str, Any]:
    """Machine-readable contract. Non-executable families fail closed for runs."""
    family_id = _check_id(family_id)
    qualification = qualification_of(family_id)
    runner = RUNNERS.get(family_id)
    executable = runner is not None and qualification == "required"
    reason = None if executable else _ineligible_reason(family_id, qualification, runner)
    return {
        "schema": SURFACE_SCHEMA,
        "id": family_id,
        "version": "1",
        "purpose": runner.purpose if runner else "",
        "tasks": list(runner.tasks) if runner else [],
        "trigger_terms": list(runner.trigger_terms) if runner else [],
        "scores": list(runner.scores) if runner else [],
        "required_inputs": list(runner.required_inputs) if runner else [],
        "data_labels": ["SYNTHETIC", "market"],
        "runner": f"research --only {family_id}" if executable else "",
        "resource_limit_s": runner.resource_limit_s if runner else 0,
        "dependencies": [],
        "qualification": qualification,
        "executable": executable,
        "retirement_reason": _retirement_reason(family_id),
        "refusal": reason,
    }


def _ineligible_reason(family_id: str, qualification: str, runner: FamilyRunner | None) -> str:
    if qualification == "retired":
        return (
            f"{family_id} is retired; new selection is refused and historical receipts still verify"
        )
    if qualification == "unknown":
        return f"{family_id} is not a known benchmark family"
    if qualification == "live_optional" and runner is None:
        return (
            f"{family_id} is live-optional but has no selective runner; "
            "refusing to execute it from the family surface"
        )
    if qualification != "required":
        return f"{family_id} is not a required qualified family (qualification={qualification})"
    return f"{family_id} has no implemented selective runner"


def list_executable() -> list[dict[str, Any]]:
    """Qualified families the selective surface can actually run."""
    order = _order_index()
    ids = [family_id for family_id in RUNNERS if qualification_of(family_id) == "required"]
    ids.sort(key=lambda family_id: (order.get(family_id, 10_000), family_id))
    return [family_contract(family_id) for family_id in ids]


def _score(goal: str, runner: FamilyRunner) -> int:
    lowered = goal.lower()
    score = 0
    for term in runner.trigger_terms:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            score += 1
    if re.search(rf"\b{re.escape(runner.family_id)}\b", lowered):
        score += 5
    return score


def select_families(goal: str, *, max_families: int = 4) -> dict[str, Any]:
    """Choose eligible runners for a goal and list every material exclusion."""
    if max_families < 0:
        raise FamilySurfaceError("max_families must be nonnegative")
    if not isinstance(goal, str) or not goal.strip():
        raise FamilySurfaceError("goal must be non-empty text")
    order = _order_index()
    scored = [
        (_score(goal, runner), runner.family_id)
        for runner in RUNNERS.values()
        if qualification_of(runner.family_id) == "required"
    ]
    scored = [(score, family_id) for score, family_id in scored if score > 0]
    scored.sort(key=lambda item: (-item[0], order.get(item[1], 10_000), item[1]))
    chosen = [family_id for _score_value, family_id in scored[:max_families]]
    exclusions: list[dict[str, str]] = []
    for _score_value, family_id in scored[max_families:]:
        exclusions.append(
            {
                "id": family_id,
                "reason": f"eligible but over the max_families={max_families} cap",
            }
        )
    chosen_set = set(chosen)
    for token in dict.fromkeys(_GOAL_TOKEN.findall(goal.lower())):
        if token in chosen_set or token not in _known_ids():
            continue
        contract = family_contract(token)
        if contract["executable"]:
            continue
        exclusions.append({"id": token, "reason": str(contract["refusal"])})
    return {
        "schema": SURFACE_SCHEMA,
        "goal": goal,
        "max_families": max_families,
        "selected": [family_contract(family_id) for family_id in chosen],
        "exclusions": exclusions,
    }


def _known_ids() -> set[str]:
    required, retired, live, optional = _catalog()
    return set(required) | set(retired) | set(live) | set(optional) | set(RUNNERS)


def parse_only(text: str) -> list[str]:
    """Parse a comma-separated id list. Rejects blanks, duplicates, and unsafe ids."""
    if not isinstance(text, str) or not text.strip():
        raise FamilySurfaceError("--only requires at least one family id")
    ids: list[str] = []
    for raw in text.split(","):
        family_id = _check_id(raw.strip())
        if family_id in ids:
            raise FamilySurfaceError(f"duplicate family id {family_id}")
        ids.append(family_id)
    return ids


def _default_load_panel(config: Any) -> PanelBundle:
    import polars as pl

    from quant_fund.pipeline.dataset import build_gold, ensure_silver, panel
    from quant_fund.utils.seeds import set_global_seed

    set_global_seed(int(config.train.random_seed))
    source = str(config.data.source)
    build_gold(config, refresh=source == "synthetic")
    frame = panel(config)
    benchmark = config.data.benchmark_id
    if "security_id" in frame.columns:
        frame = frame.filter(pl.col("security_id") != benchmark)
    label = str(config.train.ranking_target)
    if label not in frame.columns:
        candidates = [c for c in frame.columns if c.startswith("future_idio_return")]
        if not candidates:
            candidates = [c for c in frame.columns if c.startswith("future_excess_return")]
        if not candidates and "future_return_1" in frame.columns:
            candidates = ["future_return_1"]
        if not candidates:
            raise FamilySurfaceError(
                "ranking target is absent from the panel; refusing to invent a label "
                "or substitute synthetic rows"
            )
        label = candidates[0]
    northset = ensure_silver(config)
    if "security_id" in northset.columns:
        northset = northset.filter(pl.col("security_id") != benchmark)
    return PanelBundle(frame=frame, label=label, northset=northset)


def _callable(target: str) -> Runner:
    module_name, _, attr = target.partition(":")
    if not module_name or not attr:
        raise FamilySurfaceError(f"runner target {target!r} is malformed")
    module = importlib.import_module(module_name)
    fn = getattr(module, attr, None)
    if not callable(fn):
        raise FamilySurfaceError(f"runner {target} is not callable")
    return cast(Runner, fn)


def _invoke(runner: FamilyRunner, fn: Runner, bundle: PanelBundle | None, config: Any) -> Any:
    if runner.needs == "none":
        return fn()
    if bundle is None:
        raise FamilySurfaceError(f"{runner.family_id} requires a panel and none was loaded")
    if runner.needs == "panel_config_label":
        return fn(bundle.frame, config, bundle.label)
    if runner.needs == "panel_config":
        return fn(bundle.frame, config)
    if runner.needs == "panel":
        return fn(bundle.frame)
    if runner.needs == "panel_label":
        return fn(bundle.frame, bundle.label)
    if runner.needs == "northset":
        return fn(bundle.northset, config)
    raise FamilySurfaceError(f"unknown runner needs {runner.needs!r}")


def _code_revision() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    revision = completed.stdout.strip()
    return revision or "unknown"


def _artifact_dir(config: Any) -> Path | None:
    data = getattr(config, "data", None)
    root = getattr(data, "root", None)
    if root is None:
        return None
    return Path(root) / "family_runs"


def _write_artifact(directory: Path, payload: dict[str, Any]) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    digest = payload["sha256"]
    target = directory / f"{digest[:16]}.json"
    temporary = directory / f".{digest[:16]}.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)
    os.chmod(target, 0o600)
    return str(target)


def run_families(
    config: Any,
    family_ids: list[str],
    *,
    runners: Mapping[str, Runner] | None = None,
    load_panel: Callable[[Any], PanelBundle] | None = None,
    write_artifact: bool = True,
    budget_s: float | None = None,
) -> dict[str, Any]:
    """Execute only the named eligible runners. Refuses the rest before any bench."""
    if not family_ids:
        raise FamilySurfaceError("refusing to run an empty family selection")
    checked = [_check_id(family_id) for family_id in family_ids]
    if len(checked) != len(set(checked)):
        raise FamilySurfaceError("duplicate family ids are refused")
    selected: list[FamilyRunner] = []
    for family_id in checked:
        contract = family_contract(family_id)
        if not contract["executable"]:
            raise FamilySurfaceError(str(contract["refusal"]))
        selected.append(RUNNERS[family_id])
    estimated = sum(runner.resource_limit_s for runner in selected)
    if budget_s is not None and estimated > budget_s:
        raise FamilySurfaceError(
            f"estimated resource_limit_s {estimated} exceeds budget_s {budget_s}; "
            "refusing before execution. This is an estimate, not a spend guarantee."
        )
    needs_panel = any(runner.needs != "none" for runner in selected)
    bundle: PanelBundle | None = None
    if needs_panel:
        loader = load_panel or _default_load_panel
        bundle = loader(config)
    started = time.monotonic()
    results: dict[str, Any] = {}
    overrides = dict(runners or {})
    for runner in selected:
        fn = overrides.get(runner.family_id, _callable(runner.target))
        results[runner.family_id] = _invoke(runner, fn, bundle, config)
    elapsed = time.monotonic() - started
    source = str(getattr(getattr(config, "data", None), "source", "unknown"))
    payload: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "partial": True,
        "verified": False,
        "evidence_class": "SYNTHETIC" if source == "synthetic" else source,
        "data_source": source,
        "family_ids": [runner.family_id for runner in selected],
        "results": results,
        "elapsed_s": elapsed,
        "estimated_limit_s": estimated,
        "within_estimate": elapsed <= estimated,
        "code_revision": _code_revision(),
        "note": (
            "Partial family run. Not a full research receipt. "
            "verified stays false until verify-research accepts a receipt."
        ),
    }
    payload["artifact"] = ""
    body = {key: value for key, value in payload.items() if key not in {"sha256", "artifact"}}
    encoded = json.dumps(body, sort_keys=True, default=str).encode("utf-8")
    payload["sha256"] = hashlib.sha256(encoded).hexdigest()
    if write_artifact:
        directory = _artifact_dir(config)
        if directory is not None:
            payload["artifact"] = str(directory / f"{payload['sha256'][:16]}.json")
            _write_artifact(directory, payload)
    return payload
