"""forecast_infra_audit — adversarial probes on the forecast plumbing.

Pinned contract:

- ``map_signals``: unknown mapping / missing ``score`` / bad threshold
  refuse; output bounded in [-1,1]; non-finite scores map to null;
  single-name rank is 0; every row is stamped PLACEHOLDER_NOT_A_STRATEGY.
- ``create_model``: ``fx-1``/``fx1`` without an entrypoint refuse; a
  malformed entrypoint spec refuses; a non-Fx1Model entrypoint under the
  fx-1 name refuses; built-ins resolve.
- Reference models: empty frame, missing ids, or a lookahead/label column
  refuse; momentum needs ``ret_1`` or a ``mom_*`` column; forecasts carry
  the schema columns.
- ``Fx1Model`` base raises NotImplementedError (external plugin only).

Sealed ``forecast_infra_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

from typing import Any

import polars as pl

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["forecast_infra_audit", "forecast_infra_audit_bench"]


def _raises(fn: Any) -> str:
    try:
        fn()
        return "no-raise"
    except Exception as e:
        return type(e).__name__


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": [1, 1, 2, 2],
            "security_id": ["a", "b", "a", "b"],
            "score": [0.5, -0.5, 0.1, -0.1],
            "ret_1": [0.01, -0.01, 0.02, -0.02],
        }
    )


def forecast_infra_audit() -> dict[str, Any]:
    from fx1.forecast.protocol import Fx1Model
    from fx1.forecast.registry import create_model
    from fx1.forecast.signals import map_signals

    out: dict[str, Any] = {}
    f = _frame()

    out["unknown_mapping_refuses"] = _raises(lambda: map_signals(f, "bogus")) == "ValueError"
    out["missing_score_refuses"] = _raises(lambda: map_signals(f.drop("score"), "sign")).endswith(
        "Error"
    )
    out["bad_threshold_refuses"] = (
        _raises(lambda: map_signals(f, "threshold", threshold=-1.0)) == "ValueError"
        and _raises(lambda: map_signals(f, "threshold", threshold=float("nan"))) == "ValueError"
    )
    s = map_signals(f, "sign")
    sigvals = [float(v) for v in s["signal"].drop_nulls().to_list()]
    out["bounded"] = bool(sigvals) and max(abs(v) for v in sigvals) <= 1.0
    out["role_stamped"] = (s["mapping_role"].unique().to_list()) == ["PLACEHOLDER_NOT_A_STRATEGY"]
    nf = f.with_columns(pl.lit(None, dtype=pl.Float64).alias("score"))
    out["null_maps_null"] = map_signals(nf, "sign")["signal"].null_count() == 4
    one = f.filter(pl.col("event_time") == 1).head(1)
    out["single_rank_zero"] = map_signals(one, "rank")["signal"][0] == 0.0

    out["fx1_refuses_no_entrypoint"] = (
        _raises(lambda: create_model("fx-1")) == "ModelNotRegistered"
        and _raises(lambda: create_model("fx1")) == "ModelNotRegistered"
    )
    out["malformed_entrypoint_refuses"] = (
        _raises(lambda: create_model("dummy-zero", entrypoint="no-colon")) == "ValueError"
    )
    out["unresolvable_refuses"] = (
        _raises(lambda: create_model("dummy-zero", entrypoint="fx1.forecast.dummy:Nope"))
        == "ModelNotRegistered"
    )

    class _NotFx1:
        pass

    # a non-Fx1Model class entrypoint under the fx-1 name refuses
    out["fx1_entrypoint_typechecked"] = (
        _raises(lambda: create_model("fx-1", entrypoint="builtins:int")) == "TypeError"
    )
    out["builtins_resolve"] = (
        create_model("dummy-zero").name == "dummy-zero"
        and create_model("dummy-momentum").name == "dummy-momentum"
    )
    out["unknown_builtin_refuses"] = _raises(lambda: create_model("nope")) == ("ModelNotRegistered")

    z = create_model("dummy-zero")
    out["empty_frame_refuses"] = _raises(lambda: z.predict(f.head(0))) == "ValueError"
    out["missing_ids_refuses"] = _raises(lambda: z.predict(f.drop("security_id"))).endswith("Error")
    leaked = f.with_columns(pl.lit(1.0).alias("label"))
    out["leakage_refuses"] = _raises(lambda: z.predict(leaked)).endswith("Error")
    pred = z.predict(f)
    out["forecast_schema_cols"] = {
        "event_time",
        "security_id",
        "horizon_bars",
        "predicted_return",
        "predicted_price",
        "confidence",
    } <= set(pred.columns)
    out["zero_predicts_zero"] = bool(pred["predicted_return"].abs().max() == 0.0)
    m = create_model("dummy-momentum")
    out["momentum_needs_feature"] = _raises(lambda: m.predict(f.drop("ret_1"))) == "ValueError"
    out["fx1_base_unimplemented"] = _raises(lambda: Fx1Model().load()) == "NotImplementedError"
    return out


def forecast_infra_audit_bench() -> dict[str, Any]:
    r = forecast_infra_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "forecast_infra_audit",
        "schema": "forecast_infra_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Forecast infra contract holds: signal mapping is bounded, "
            "role-stamped and fail-closed; the registry refuses unbound "
            "fx-1 and type-checks entrypoints; reference models refuse "
            "empty/missing-id/leaky frames."
            if ok
            else f"FORECAST INFRA AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
