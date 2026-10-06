"""scoring_audit — proper-score primitive contract battery.

The four ``score_*`` operations are the harness's honesty-critical math:
pinball (quantile), interval (Winkler/Gneiting–Raftery eq. 43), Brier +
log-loss, and empirical CRPS. Every research claim the lab seals bottoms
out in one of these — a silent sign flip or denominator change turns
every downstream receipt into a lie. These probes pin the formulas
against hand-computed fixtures and an independent O(n²) CRPS reference:

- *Descriptor contract* — ``Operation`` ids live in the ``skills.``
  namespace with ``kind="skill"``, schemas and handler co-located in the
  capability file, ``describe()`` emits the host-facing shape.
- *score_quantiles* — exact pinball orientation (level-weighted above,
  (level-1)-weighted below), per-level means + unweighted cross-level
  mean, crossing quantiles rejected (ties allowed), strictly increasing
  levels, shape/bounds/finite enforcement, ``extra="forbid"``.
- *score_intervals* — additive width + miss penalties at multiplier
  ``2/(1-c)``, inclusive endpoints count as covered, equal bounds
  accepted, inverted/unequal-length/out-of-range rejected.
- *score_binary_forecasts* — one-event Brier ``(p-y)²`` on the
  *original* probability even when clipped, log loss in nats, impossible
  endpoints counted and reported as ``positive_infinity`` + ``None``
  unless a clip rescues them, clip bounds and probability/outcome
  domains enforced.
- *score_empirical_crps* — exact empirical-CDF CRPS: degenerate sample
  at the outcome scores 0, singleton scores ``|s-y|``, ties and unequal
  ensemble sizes accepted, and the sorted-gap integration matches an
  independent ``E|X-y| - E|X-X'|/2`` reference on every fixture.

Probes are literal bools: ``True`` pins a contract that holds;
``False`` pins a measured divergence — the sealed receipt names every
defect it found. Everything is in-process and deterministic: no served
app, no network, no workspace files (scorers never touch the context).
"""

from __future__ import annotations

import json
import math
import tempfile
from math import fsum
from pathlib import Path
from typing import Any

import pydantic

from fx1.operations import (
    score_binary_forecasts,
    score_empirical_crps,
    score_intervals,
    score_quantiles,
)
from fx1.operations.base import Operation, OperationContext

__all__ = ["scoring_audit", "scoring_audit_bench"]


def _ctx() -> OperationContext:
    # Scorers never read the workspace; a real (validated) root keeps the
    # call shape honest anyway.
    return OperationContext(workspace_root=Path(tempfile.gettempdir()))


def _refuses(model: Any, **kwargs: Any) -> bool:
    try:
        model(**kwargs)
    except (pydantic.ValidationError, ValueError):
        return True
    return False


def _probe_descriptors() -> dict[str, bool]:
    out: dict[str, bool] = {}
    ops: tuple[Operation[Any, Any], ...] = (
        score_quantiles.OPERATION,
        score_intervals.OPERATION,
        score_binary_forecasts.OPERATION,
        score_empirical_crps.OPERATION,
    )
    expected = {
        "skills.score_quantiles",
        "skills.score_intervals",
        "skills.score_binary_forecasts",
        "skills.score_empirical_crps",
    }
    out["op_ids_namespaced"] = {o.id for o in ops} == expected
    out["op_kind_skill"] = all(o.kind == "skill" for o in ops)
    out["op_schemas_module_local"] = all(
        o.input_model.__module__ == o.handler.__module__ == o.output_model.__module__ for o in ops
    )
    described = [o.describe() for o in ops]
    out["op_describe_shape"] = all(
        d["id"] == o.id and d["kind"] == "skill" and d["description"].strip()
        for d, o in zip(described, ops, strict=True)
    )
    out["op_descriptions_nonempty"] = all(o.description.strip() for o in ops)
    return out


def _probe_quantiles() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = score_quantiles
    ctx = _ctx()

    # y=1, q=0, level=0.5: residual 1 → loss 0.5*1 = 0.5
    r = mod.execute(mod.Input(outcomes=[1.0], levels=[0.5], quantiles=[[0.0]]), ctx)
    out["q_pinball_formula"] = math.isclose(r.mean_pinball_by_level[0], 0.5)
    # level 0.9: residual 1 → 0.9; residual -1 → 0.1 (asymmetric weight)
    r2 = mod.execute(
        mod.Input(outcomes=[1.0, -1.0], levels=[0.9], quantiles=[[0.0], [0.0]]),
        ctx,
    )
    out["q_pinball_asymmetric"] = math.isclose(r2.mean_pinball_by_level[0], (0.9 + 0.1) / 2)
    r3 = mod.execute(
        mod.Input(outcomes=[1.0], levels=[0.25, 0.75], quantiles=[[0.0, 2.0]]),
        ctx,
    )
    # level .25: resid 1 → .25; level .75: resid -1 → .25*... (0.75-1)*-1 = .25
    out["q_per_level_means"] = math.isclose(r3.mean_pinball_by_level[0], 0.25) and math.isclose(
        r3.mean_pinball_by_level[1], 0.25
    )
    out["q_mean_unweighted_across_levels"] = math.isclose(
        r3.mean_pinball, sum(r3.mean_pinball_by_level) / 2
    )
    perfect = mod.execute(
        mod.Input(outcomes=[3.0], levels=[0.1, 0.5, 0.9], quantiles=[[3.0, 3.0, 3.0]]),
        ctx,
    )
    out["q_perfect_zero"] = math.isclose(perfect.mean_pinball, 0.0)
    off = mod.execute(
        mod.Input(outcomes=[3.0], levels=[0.1, 0.5, 0.9], quantiles=[[3.1, 3.1, 3.1]]),
        ctx,
    )
    out["q_proper_perfect_wins"] = perfect.mean_pinball < off.mean_pinball
    out["q_observation_count"] = r2.observation_count == 2
    out["q_levels_echoed"] = r3.levels == [0.25, 0.75]

    out["q_reject_crossing"] = _refuses(
        mod.Input, outcomes=[1.0], levels=[0.5, 0.9], quantiles=[[1.0, 0.0]]
    )
    tied = mod.Input(outcomes=[1.0], levels=[0.5, 0.9], quantiles=[[0.5, 0.5]])
    out["q_ties_allowed"] = tied.quantiles[0] == [0.5, 0.5]
    out["q_reject_unsorted_levels"] = _refuses(
        mod.Input, outcomes=[1.0], levels=[0.9, 0.5], quantiles=[[0.0, 0.0]]
    )
    out["q_reject_equal_levels"] = _refuses(
        mod.Input, outcomes=[1.0], levels=[0.5, 0.5], quantiles=[[0.0, 0.0]]
    )
    out["q_reject_row_width"] = _refuses(
        mod.Input, outcomes=[1.0], levels=[0.5, 0.9], quantiles=[[0.0]]
    )
    out["q_reject_shape_mismatch"] = _refuses(
        mod.Input, outcomes=[1.0, 2.0], levels=[0.5], quantiles=[[0.0]]
    )
    out["q_reject_nan"] = _refuses(
        mod.Input, outcomes=[float("nan")], levels=[0.5], quantiles=[[0.0]]
    )
    out["q_reject_inf"] = _refuses(
        mod.Input, outcomes=[float("inf")], levels=[0.5], quantiles=[[0.0]]
    )
    out["q_reject_oob_value"] = _refuses(
        mod.Input, outcomes=[1e101], levels=[0.5], quantiles=[[0.0]]
    )
    out["q_reject_level_zero"] = _refuses(
        mod.Input, outcomes=[1.0], levels=[0.0], quantiles=[[0.0]]
    )
    out["q_reject_level_one"] = _refuses(mod.Input, outcomes=[1.0], levels=[1.0], quantiles=[[0.0]])
    out["q_reject_empty_outcomes"] = _refuses(mod.Input, outcomes=[], levels=[0.5], quantiles=[])
    out["q_extra_forbid"] = _refuses(
        mod.Input,
        outcomes=[1.0],
        levels=[0.5],
        quantiles=[[0.0]],
        bogus=1,
    )
    return out


def _probe_intervals() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = score_intervals
    ctx = _ctx()

    # cov=0.8 → multiplier 10; y in [0,10] → score = width 10, covered
    r = mod.execute(
        mod.Input(outcomes=[5.0], lower=[0.0], upper=[10.0], nominal_coverage=0.8),
        ctx,
    )
    out["i_formula_hit"] = math.isclose(r.interval_score, 10.0)
    out["i_empirical_hit"] = math.isclose(r.empirical_coverage, 1.0)
    # y=-1 → lower penalty 10*(0-(-1)) = 10 → score 20
    r2 = mod.execute(
        mod.Input(outcomes=[-1.0], lower=[0.0], upper=[10.0], nominal_coverage=0.8),
        ctx,
    )
    out["i_miss_lower"] = math.isclose(r2.interval_score, 20.0) and math.isclose(
        r2.mean_lower_miss_penalty, 10.0
    )
    r3 = mod.execute(
        mod.Input(outcomes=[11.0], lower=[0.0], upper=[10.0], nominal_coverage=0.8),
        ctx,
    )
    out["i_miss_upper"] = math.isclose(r3.interval_score, 20.0) and math.isclose(
        r3.mean_upper_miss_penalty, 10.0
    )
    out["i_additive_decomposition"] = math.isclose(
        r2.interval_score,
        r2.mean_width + r2.mean_lower_miss_penalty + r2.mean_upper_miss_penalty,
    )
    edge = mod.execute(
        mod.Input(
            outcomes=[0.0, 10.0],
            lower=[0.0, 0.0],
            upper=[10.0, 10.0],
            nominal_coverage=0.9,
        ),
        ctx,
    )
    out["i_inclusive_endpoints"] = math.isclose(edge.empirical_coverage, 1.0)
    point = mod.execute(
        mod.Input(outcomes=[7.0], lower=[7.0], upper=[7.0], nominal_coverage=0.5),
        ctx,
    )
    out["i_zero_width_hit"] = math.isclose(point.interval_score, 0.0)
    out["i_nominal_echoed"] = math.isclose(r.nominal_coverage, 0.8)

    out["i_reject_inverted"] = _refuses(
        mod.Input, outcomes=[5.0], lower=[9.0], upper=[1.0], nominal_coverage=0.8
    )
    out["i_reject_len_mismatch"] = _refuses(
        mod.Input,
        outcomes=[5.0, 6.0],
        lower=[0.0],
        upper=[10.0],
        nominal_coverage=0.8,
    )
    out["i_reject_cov_zero"] = _refuses(
        mod.Input, outcomes=[5.0], lower=[0.0], upper=[1.0], nominal_coverage=0.0
    )
    out["i_reject_cov_one"] = _refuses(
        mod.Input, outcomes=[5.0], lower=[0.0], upper=[1.0], nominal_coverage=1.0
    )
    edge_cov = mod.Input(outcomes=[5.0], lower=[0.0], upper=[10.0], nominal_coverage=0.999999)
    out["i_cov_edge_accepted"] = math.isclose(edge_cov.nominal_coverage, 0.999999)
    out["i_reject_nan"] = _refuses(
        mod.Input, outcomes=[float("nan")], lower=[0.0], upper=[1.0], nominal_coverage=0.8
    )
    out["i_extra_forbid"] = _refuses(
        mod.Input,
        outcomes=[5.0],
        lower=[0.0],
        upper=[1.0],
        nominal_coverage=0.8,
        bogus=1,
    )
    return out


def _probe_binary() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = score_binary_forecasts
    ctx = _ctx()

    r = mod.execute(mod.Input(outcomes=[1, 0], probabilities=[0.7, 0.7]), ctx)
    out["b_brier_formula"] = math.isclose(r.brier_score, (0.09 + 0.49) / 2)
    perfect = mod.execute(mod.Input(outcomes=[1, 0], probabilities=[1.0, 0.0]), ctx)
    out["b_brier_perfect_zero"] = math.isclose(perfect.brier_score, 0.0)
    # y=1, p=1/e → -log(p) = 1 nat
    ln = mod.execute(mod.Input(outcomes=[1], probabilities=[math.exp(-1)]), ctx)
    out["b_logloss_nats"] = math.isclose(ln.mean_log_loss or 0.0, 1.0, rel_tol=1e-12)
    zero = mod.execute(mod.Input(outcomes=[0], probabilities=[0.0]), ctx)
    out["b_logloss_correct_endpoint_finite"] = (
        zero.mean_log_loss is not None
        and math.isclose(zero.mean_log_loss, 0.0)
        and zero.log_loss_status == "finite"
    )
    inf = mod.execute(mod.Input(outcomes=[1, 0], probabilities=[0.0, 0.5]), ctx)
    out["b_impossible_null_loss"] = inf.mean_log_loss is None
    out["b_impossible_status"] = inf.log_loss_status == "positive_infinity"
    out["b_impossible_counted"] = inf.impossible_event_count == 1
    out["b_impossible_brier_still_scored"] = math.isclose(inf.brier_score, (1.0 + 0.25) / 2)
    clipped = mod.execute(mod.Input(outcomes=[1], probabilities=[0.0], probability_clip=0.01), ctx)
    out["b_clip_rescues"] = (
        clipped.mean_log_loss is not None
        and math.isclose(clipped.mean_log_loss, -math.log(0.01))
        and clipped.log_loss_status == "finite"
    )
    out["b_clip_counted"] = clipped.clipped_probability_count == 1
    out["b_clip_brier_uses_raw"] = math.isclose(clipped.brier_score, 1.0)
    out["b_clip_impossible_still_counted"] = clipped.impossible_event_count == 1
    out["b_clip_echoed"] = math.isclose(clipped.probability_clip or 0.0, 0.01)
    unclipped = mod.execute(mod.Input(outcomes=[1], probabilities=[0.0]), ctx)
    out["b_no_clip_no_clipped"] = unclipped.clipped_probability_count == 0

    out["b_reject_p_below"] = _refuses(mod.Input, outcomes=[1], probabilities=[-0.1])
    out["b_reject_p_above"] = _refuses(mod.Input, outcomes=[1], probabilities=[1.1])
    out["b_reject_outcome_nonbinary"] = _refuses(mod.Input, outcomes=[2], probabilities=[0.5])
    out["b_reject_outcome_float"] = _refuses(mod.Input, outcomes=[0.5], probabilities=[0.5])
    out["b_reject_clip_half"] = _refuses(
        mod.Input, outcomes=[1], probabilities=[0.5], probability_clip=0.5
    )
    out["b_reject_clip_tiny"] = _refuses(
        mod.Input, outcomes=[1], probabilities=[0.5], probability_clip=1e-16
    )
    out["b_reject_len_mismatch"] = _refuses(mod.Input, outcomes=[1, 0], probabilities=[0.5])
    out["b_reject_nan_p"] = _refuses(mod.Input, outcomes=[1], probabilities=[float("nan")])
    out["b_extra_forbid"] = _refuses(mod.Input, outcomes=[1], probabilities=[0.5], bogus=1)
    return out


def _naive_crps(outcome: float, sample: list[float]) -> float:
    """Reference: E|X-y| - E|X-X'|/2 over the empirical sample, O(n²)."""
    n = len(sample)
    exy = fsum(abs(s - outcome) for s in sample) / n
    exx = fsum(abs(a - b) for a in sample for b in sample) / (n * n)
    return exy - exx / 2.0


def _probe_crps() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = score_empirical_crps
    ctx = _ctx()

    single = mod.execute(mod.Input(outcomes=[2.0], samples=[[5.0]]), ctx)
    out["c_singleton_abs_error"] = math.isclose(single.crps_by_observation[0], 3.0)
    degen = mod.execute(mod.Input(outcomes=[4.0], samples=[[4.0, 4.0, 4.0]]), ctx)
    out["c_degenerate_zero"] = math.isclose(degen.crps_by_observation[0], 0.0)
    two = mod.execute(mod.Input(outcomes=[0.5], samples=[[0.0, 1.0]]), ctx)
    out["c_two_point_handcheck"] = math.isclose(two.crps_by_observation[0], 0.25)

    fixtures = [
        (0.0, [0.0, 0.0, 1.0, 1.0]),
        (1.7, [-2.0, 0.5, 0.5, 3.0, 9.0]),
        (-0.25, [1.0, -1.0, 0.25, 0.25, 0.25, 2.0]),
        (100.0, [90.0, 110.0]),
    ]
    r = mod.execute(
        mod.Input(
            outcomes=[f[0] for f in fixtures],
            samples=[list(f[1]) for f in fixtures],
        ),
        ctx,
    )
    out["c_matches_naive_reference"] = all(
        math.isclose(got, _naive_crps(y, list(s)), rel_tol=1e-9, abs_tol=1e-12)
        for got, (y, s) in zip(r.crps_by_observation, fixtures, strict=True)
    )
    out["c_mean_is_obs_mean"] = math.isclose(
        r.mean_crps, fsum(r.crps_by_observation) / len(fixtures)
    )
    unsorted = mod.execute(mod.Input(outcomes=[0.5], samples=[[1.0, -1.0, 0.0]]), ctx)
    out["c_unsorted_input_ok"] = math.isclose(
        unsorted.crps_by_observation[0],
        _naive_crps(0.5, [1.0, -1.0, 0.0]),
    )
    out["c_minmax_counts"] = r.minimum_sample_count == 2 and r.maximum_sample_count == 6
    out["c_observation_count"] = r.observation_count == len(fixtures)

    out["c_reject_empty_sample"] = _refuses(mod.Input, outcomes=[1.0], samples=[[]])
    out["c_reject_shape_mismatch"] = _refuses(mod.Input, outcomes=[1.0, 2.0], samples=[[1.0]])
    out["c_reject_nan_sample"] = _refuses(mod.Input, outcomes=[1.0], samples=[[float("nan")]])
    out["c_reject_nan_outcome"] = _refuses(mod.Input, outcomes=[float("nan")], samples=[[1.0]])
    out["c_reject_oob_value"] = _refuses(mod.Input, outcomes=[1.0], samples=[[1e101]])
    out["c_extra_forbid"] = _refuses(mod.Input, outcomes=[1.0], samples=[[1.0]], bogus=1)
    return out


def scoring_audit() -> dict[str, bool]:
    """Every proper-score contract as literal booleans."""
    out: dict[str, bool] = {}
    out.update(_probe_descriptors())
    out.update(_probe_quantiles())
    out.update(_probe_intervals())
    out.update(_probe_binary())
    out.update(_probe_crps())
    return out


def scoring_audit_bench() -> dict[str, Any]:
    """Sealed receipt: contract probes True, divergences named."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = scoring_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "scoring_audit",
        "schema": "scoring_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "in-process execute() calls; no served app, no workspace I/O",
            "not_verified": [
                "registry dispatch wiring (see registry lane)",
                "plugin/feature operations (separate surfaces)",
                "numerical behavior beyond 1e100 declared bounds",
            ],
        },
        "interpretation": (
            "Proper-score primitives hold: pinball losses use the exact "
            "level-weighted orientation with per-level and unweighted "
            "cross-level means, reject crossings/out-of-order levels/"
            "nonfinite/out-of-domain inputs while allowing tied quantiles; "
            "interval scores decompose additively into width plus "
            "2/(1-c) miss penalties with inclusive endpoint coverage; "
            "binary forecasts score (p-y)^2 Brier on raw probabilities "
            "and nats log loss, with impossible endpoints counted and "
            "surfaced as positive_infinity + null unless a declared clip "
            "rescues them; empirical CRPS matches an independent "
            "E|X-y| - E|X-X'|/2 reference across sorted/unsorted/tied/"
            "unequal-ensemble fixtures, scoring degenerate perfect "
            "forecasts 0 and singletons |s-y|."
            if ok
            else f"SCORING AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(scoring_audit_bench(), indent=2, sort_keys=True))
