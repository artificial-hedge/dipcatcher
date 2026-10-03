"""impact — which committed evidence does this config change void?

Receipts record ``git_revision`` but a config edit voids evidence just as
surely as a code change: ``cost.half_spread_bps`` moving invalidates
every execution-cost bench; ``data.universe`` invalidates anything built
on market tape. This module makes that void computable.

``flatten`` maps a config (dict or pydantic model) to dotted leaf paths.
``config_diff`` emits the changed paths with old/new values and the
semantic *domains* they hit (``KEY_DOMAINS``). ``KIND_DOMAINS``
classifies receipt ``kind`` names into the domains they consume —
microstructure lanes consume market data + execution cost, forecast/
seq-inf lanes consume forecast + evaluation + market data, integrity
substrate consumes nothing market-side (a receipt's claims about chains
and pins don't depend on spread assumptions).

``invalidation_report(diff, receipts_dir)`` then names every committed
receipt the change touches. Unknown kinds fail *conservative* — an
unmapped market-data receipt is treated as consuming ``market_data``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# config key prefix -> evidence domains the key feeds
KEY_DOMAINS: dict[str, frozenset[str]] = {
    "data.": frozenset({"market_data"}),
    "universe.": frozenset({"market_data"}),
    "calendar.": frozenset({"market_data"}),
    "cost.": frozenset({"execution_cost"}),
    "execution.": frozenset({"execution_cost"}),
    "portfolio.": frozenset({"portfolio"}),
    "optimizer.": frozenset({"portfolio"}),
    "risk": frozenset({"risk"}),
    "train.": frozenset({"forecast"}),
    "models.": frozenset({"forecast"}),
    "kronos.": frozenset({"forecast"}),
    "features.": frozenset({"features"}),
    "missing.": frozenset({"features"}),
    "horizon.": frozenset({"labels"}),
    "quantile.": frozenset({"forecast"}),
    "validation.": frozenset({"evaluation"}),
    "runtime.": frozenset(),
}

_MICRO_TOKENS = (
    "lobster",
    "queue",
    "spread",
    "trade",
    "exec",
    "book",
    "order",
    "vpin",
    "hawkes",
    "depth",
    "cancel",
    "mid_",
    "lob_",
    "micro",
    "impact",
    "sweep",
    "stale",
    "flow",
    "side_",
    "hidden",
    "round_lot",
    "split",
    "propagator",
    "quote",
    "streak",
    "markout",
    "lifetime",
    "price_",
    "run_",
    "event_",
    "intraday",
    "tape",
    "sign_",
    "size_",
    "wait",
    "occupancy",
    "imbalance",
    "resilience",
    "burst",
    "granger",
    "revision",
    "cluster",
    "drift_bench",
    "almgren",
    "dgm",
    "advers",
    "glft",
    "fifo",
    "pin_estim",
)
_FORECAST_TOKENS = (
    "fleet",
    "rankic",
    "coverage",
    "conformal",
    "calib",
    "evalue",
    "drift_",
    "changepoint",
    "tail",
    "serial",
    "fdr",
    "spa",
    "mcs",
    "emerge",
    "honest",
    "winner",
    "selective",
    "energy",
    "quantile",
    "mosaic",
    "loss_cs",
    "shift",
    "eprocess",
    "meta_model",
    "seed_sweep",
    "vol_bench",
    "hstep",
    "stability",
    "concordance",
    "race",
    "panel_audit",
    "hmm_",
    "distribution",
)
_INTEGRITY_TOKENS = (
    "lattice",
    "corpus_epoch",
    "repo_integrity",
    "provenance",
    "artifact",
    "monitor",
    "native_conformance",
    "hmm_verify",
    "epoch",
    "gate",
    "witness",
    "bundle",
    "absence",
    "anchor",
    "ots",
    "checkpoint",
    "fuzz",
    "audit_coverage",
    "receipt_surface",
    "sota",
    "deps_",
    "admission",
)
_RISK_TOKENS = ("decay", "crash", "risk", "ruin")


def kind_domains(kind: str, data_label: str | None = None) -> frozenset[str]:
    """Domains a receipt kind consumes. Conservative for unknown kinds."""
    k = kind.lower()
    if any(t in k for t in _INTEGRITY_TOKENS):
        return frozenset()
    out: set[str] = set()
    if any(t in k for t in _MICRO_TOKENS):
        out |= {"market_data", "execution_cost"}
    if any(t in k for t in _FORECAST_TOKENS):
        out |= {"forecast", "evaluation", "market_data"}
    if any(t in k for t in _RISK_TOKENS):
        out |= {"risk", "market_data"}
    if not out and (data_label or "").upper() in {"REAL", "MIXED"}:
        out.add("market_data")
    return frozenset(out)


def flatten(cfg: Any, prefix: str = "") -> dict[str, Any]:
    """Config object/dict -> {dotted.path: leaf_value}."""
    if hasattr(cfg, "model_dump"):
        cfg = cfg.model_dump()
    if not isinstance(cfg, dict):
        return {prefix.rstrip("."): cfg}
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) or hasattr(v, "model_dump"):
            out.update(flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def _key_domains(path: str) -> frozenset[str]:
    for prefix, doms in KEY_DOMAINS.items():
        p = prefix.rstrip(".")
        if path == p or path.startswith(p + "."):
            return doms
    return frozenset()


def config_diff(before: Any, after: Any) -> list[dict[str, Any]]:
    """Changed leaves with their hit domains."""
    fa, fb = flatten(before), flatten(after)
    out: list[dict[str, Any]] = []
    for key in sorted(set(fa) | set(fb)):
        if fa.get(key) != fb.get(key):
            out.append(
                {
                    "path": key,
                    "old": fa.get(key),
                    "new": fb.get(key),
                    "domains": sorted(_key_domains(key)),
                }
            )
    return out


def invalidation_report(diff: list[dict[str, Any]], receipts_dir: Path | str) -> dict[str, Any]:
    """Which committed receipts does this diff void?"""
    touched: set[str] = set()
    for change in diff:
        touched.update(change["domains"])
    hits: list[dict[str, Any]] = []
    for f in sorted(Path(receipts_dir).glob("*.json")):
        try:
            rec = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        kind = str(rec.get("kind", f.stem))
        kd = kind_domains(kind, rec.get("data_label"))
        if kd & touched:
            hits.append({"receipt": f.name, "kind": kind, "via": sorted(kd & touched)})
    return {
        "changed_keys": len(diff),
        "touched_domains": sorted(touched),
        "n_invalidated": len(hits),
        "invalidated": hits,
        "ok": len(hits) == 0 or len(diff) == 0,
    }


__all__ = ["config_diff", "flatten", "invalidation_report", "kind_domains"]
