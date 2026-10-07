"""Composite honest verdict — one sealed claim over a tournament outcome.

The sequential-inference lanes each answer one question honestly:

- ``winner_curse`` — how much did *selecting* the winner inflate its
  reported score?
- ``evalues`` — is the winner's edge over the runner-up supported by
  anytime-valid evidence, or just noise at the horizon we stopped at?
- ``drift_alarm`` — is the winner's advantage stable, or did it change
  level inside the observation window?

``honest_verdict`` composes them into a single ``honest_verdict.v1``
receipt. The composite verdict is intentionally conservative:

- ``confirmed`` — bias handled (corrected score reported), promotion
  evidence at the stated alpha, no drift alarm.
- ``supported_with_caveats`` — edge holds but the e-process has not
  crossed 1/alpha (evidence not yet decisive), or the drift diagnostic
  fired while the e-process did not.
- ``not_supported`` — winner's-curse correction erases the edge below
  the runner-up, or a drift alarm fires mid-stream.
- ``inconclusive`` — required inputs missing, mismatched, or a
  component lane is unavailable (recorded per-lane; never silently
  skipped).

Every component is imported lazily so this module lands before its
sibling lanes; a missing lane degrades the verdict to ``inconclusive``
with the lane named — never a silent pass.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

if TYPE_CHECKING:
    from numpy.typing import NDArray

HONEST_VERDICT_SCHEMA = "honest_verdict.v1"


def _sha256_stream(scores: dict[str, NDArray[np.floating]]) -> str:
    parts = ";".join(
        f"{k}:{v.size}:{hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()[:16]}"
        for k, v in sorted(scores.items())
    )
    return hashlib.sha256(parts.encode()).hexdigest()


@dataclass
class ComponentResult:
    name: str
    available: bool
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _winner_curse_component(
    scores: dict[str, NDArray[np.floating]], seed: int, n_boot: int
) -> ComponentResult:
    try:
        from quant_fund.research.winner_curse import winner_curse_audit
    except ImportError as exc:
        return ComponentResult("winner_curse", False, error=f"unavailable:{exc.name}")
    r = winner_curse_audit(scores, seed=seed, n_boot=n_boot)
    return ComponentResult(
        "winner_curse",
        True,
        detail={
            "selected_head": r.selected_head,
            "naive_score": r.naive_score,
            "selection_bias": r.selection_bias,
            "corrected_score": r.corrected_score,
            "honest_score": r.honest_score,
            "selection_aware_ci": list(r.selection_aware_ci),
        },
    )


def _promotion_component(
    scores: dict[str, NDArray[np.floating]], winner: str, alpha: float
) -> ComponentResult:
    try:
        from quant_fund.research.evalues import LossEProcess
    except ImportError as exc:
        return ComponentResult("promotion", False, error=f"unavailable:{exc.name}")
    challenger = scores[winner]
    # Runner-up by mean loss — the strongest single alternative.
    means = {h: float(a.mean()) for h, a in scores.items() if h != winner}
    if not means:
        return ComponentResult("promotion", True, detail={"skipped": "no_runner_up"})
    runner_up = min(means, key=lambda h: means[h])
    proc = LossEProcess(
        alpha=alpha, init_scale=float(np.std(challenger - scores[runner_up])) or 1e-3
    )
    for c, r in zip(challenger, scores[runner_up], strict=True):
        proc.update(float(c), float(r))
    final = proc.states[-1]
    return ComponentResult(
        "promotion",
        True,
        detail={
            "runner_up": runner_up,
            "final_evalue": final.evalue,
            "promoted": proc.promotion_origin is not None,
            "promotion_origin": proc.promotion_origin,
            "anytime_p": final.anytime_p,
        },
    )


def _drift_component(
    scores: dict[str, NDArray[np.floating]], winner: str, alpha: float
) -> ComponentResult:
    try:
        from quant_fund.research.drift_alarm import EProcessDriftAlarm
    except ImportError as exc:
        return ComponentResult("drift", False, error=f"unavailable:{exc.name}")
    stream = np.asarray(scores[winner], dtype=float)
    diffs = np.diff(stream)
    ep = EProcessDriftAlarm(alpha=alpha)
    for x in diffs:
        ep.update(float(x))
    ph_alarm = False
    try:
        from quant_fund.research.drift_alarm import PageHinkleyAlarm

        ph = PageHinkleyAlarm()
        for x in diffs:
            ph.update(float(x))
        ph_alarm = ph.alarmed
    except ImportError:
        pass
    return ComponentResult(
        "drift",
        True,
        detail={
            "eprocess_alarmed": ep.alarmed,
            "alarm_index": ep.alarm_index,
            # capped like evalues.LossEProcess — an unbounded exp turns the
            # reported e-value into inf, which json.dumps renders as the
            # non-standard literal Infinity and breaks strict consumers.
            "final_evalue": float(np.exp(min(ep.log_e, 700.0))),
            "page_hinkley_alarmed": ph_alarm,
        },
    )


def _magnitude_component(
    scores: dict[str, NDArray[np.floating]], winner: str, runner_up: str | None, alpha: float
) -> ComponentResult:
    """Is the winner's edge materially nonzero? betting CS on the diff."""
    if not isinstance(runner_up, str):
        return ComponentResult("magnitude", True, detail={"skipped": "no_runner_up"})
    try:
        from quant_fund.research.loss_cs import cs_from_streams
    except ImportError as exc:
        return ComponentResult("magnitude", False, error=f"unavailable:{exc.name}")
    rep = cs_from_streams(
        np.asarray(scores[winner], dtype=float).tolist(),
        np.asarray(scores[runner_up], dtype=float).tolist(),
        alpha=alpha,
    )
    return ComponentResult("magnitude", True, detail=dict(rep))


def _localize_component(
    scores: dict[str, NDArray[np.floating]], winner: str, alpha: float
) -> ComponentResult:
    """When drift fires, where did the stream shift? fixed-window scan."""
    try:
        from quant_fund.research.changepoint_localize import localize_changepoint
    except ImportError as exc:
        return ComponentResult("localize", False, error=f"unavailable:{exc.name}")
    diffs = np.diff(np.asarray(scores[winner], dtype=float))
    res = localize_changepoint(diffs.tolist(), alpha=alpha)
    return ComponentResult(
        "localize",
        True,
        detail={
            "tau_hat": res.tau_hat,
            "cs_lo": res.cs_lo,
            "cs_hi": res.cs_hi,
            "alarmed": res.alarmed,
        },
    )


def _calibration_component(
    pits: dict[str, NDArray[np.floating]] | None, winner: str, alpha: float
) -> ComponentResult:
    """When PIT values are supplied: is the winner's PIT uniform?"""
    if pits is None:
        return ComponentResult("calibration", True, detail={"skipped": "pits_not_supplied"})
    try:
        from quant_fund.research.calibration_eprocess import CalibrationEProcess
    except ImportError as exc:
        return ComponentResult("calibration", False, error=f"unavailable:{exc.name}")
    u = np.asarray(pits.get(winner, np.asarray([])), dtype=float)
    if u.size == 0:
        return ComponentResult("calibration", True, detail={"skipped": "no_pits"})
    ep = CalibrationEProcess(alpha=alpha)
    for ui in u:
        ep.update(float(ui))
    return ComponentResult(
        "calibration",
        True,
        detail={
            "final_evalue": ep.wealth,
            "miscalibrated": ep.alarmed,
            "alarm_origin": ep.alarm_origin,
            "channel_wealths": dict(ep.channel_wealths),
        },
    )


def honest_verdict(
    scores: dict[str, NDArray[np.floating]],
    *,
    pits: dict[str, NDArray[np.floating]] | None = None,
    alpha: float = 0.05,
    seed: int = 0,
    n_boot: int = 2000,
    data_label: str | None = None,
) -> dict[str, Any]:
    """Composite verdict over per-head loss streams → honest_verdict.v1.

    ``pits`` is optional: per-head PIT value streams unlock the
    calibration lane (``calibration_eprocess``) in the composite.
    """
    if not scores:
        raise ValueError("scores must map at least one head")
    arrays = {h: np.asarray(v, dtype=float).ravel() for h, v in scores.items()}
    for h, a in arrays.items():
        if a.size == 0 or not np.isfinite(a).all():
            raise ValueError(f"scores[{h!r}] empty or non-finite")
    n = next(iter(arrays.values())).size
    if any(a.size != n for a in arrays.values()):
        raise ValueError("all heads must observe the same number of losses")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    pit_arrays = {h: np.asarray(v, dtype=float).ravel() for h, v in (pits or {}).items()}
    # Corpus-level fingerprint: digest over the evaluated stream content
    # only — head names are just shard identities; receipts across lanes
    # over the same streams agree, which is what the lattice edges on.
    dataset_sha256 = hash_bytes(
        canonical_json_bytes(
            {
                "shards": {
                    h: {
                        "losses_sha256": hash_bytes(np.ascontiguousarray(a).tobytes()),
                        **(
                            {
                                "pits_sha256": hash_bytes(
                                    np.ascontiguousarray(pit_arrays[h]).tobytes()
                                )
                            }
                            if h in pit_arrays
                            else {}
                        ),
                    }
                    for h, a in arrays.items()
                }
            }
        )
    )

    wc = _winner_curse_component(arrays, seed, n_boot)
    if wc.available:
        winner = str(wc.detail["selected_head"])
    else:
        winner = min(arrays, key=lambda h: float(arrays[h].mean()))

    promo = _promotion_component(arrays, winner, alpha)
    drift = _drift_component(arrays, winner, alpha)
    runner = promo.detail.get("runner_up") if promo.available else None
    magnitude = _magnitude_component(
        arrays, winner, runner if isinstance(runner, str) else None, alpha
    )
    calib = _calibration_component(pits, winner, alpha)
    # localization only fires when drift alarmed — it answers "where"
    drifted = drift.available and bool(drift.detail.get("eprocess_alarmed", False))
    localize = (
        _localize_component(arrays, winner, alpha)
        if drifted
        else ComponentResult("localize", True, detail={"skipped": "no_drift_alarm"})
    )

    components = [wc, promo, drift, magnitude, calib, localize]
    # Every component lane is load-bearing: a lane that cannot run can
    # neither vouch for nor veto the claim, so any unavailable lane
    # degrades the composite to inconclusive (recorded, never silent).
    unavailable = [c.name for c in components if not c.available]

    if unavailable:
        verdict = "inconclusive"
    else:
        promoted = bool(promo.detail.get("promoted", False))
        corrected = float(wc.detail["corrected_score"])
        # Edge erased: corrected winner score at/above runner-up mean.
        edge_erased = False
        if isinstance(runner, str):
            edge_erased = corrected >= float(arrays[runner].mean())
        miscalibrated = bool(calib.detail.get("miscalibrated", False))
        cs_excludes_zero = bool(magnitude.detail.get("excludes_zero", False))
        # PageHinkley is the diagnostic half of the drift lane: an alarm
        # there without an e-process crossing is exactly the "drift
        # diagnostic fired while the e-process did not" caveat case.
        ph_alarmed = bool(drift.detail.get("page_hinkley_alarmed", False))
        if drifted or edge_erased or miscalibrated:
            verdict = "not_supported"
        elif promoted:
            verdict = (
                "confirmed" if (cs_excludes_zero and not ph_alarmed) else "supported_with_caveats"
            )
        else:
            verdict = "supported_with_caveats"

    report: dict[str, Any] = {
        "kind": HONEST_VERDICT_SCHEMA,
        # callers that cannot name the source stamp UNKNOWN — never claim
        # SYNTHETIC for a stream whose provenance is not actually synthetic
        "data_label": str(data_label) if data_label else "UNKNOWN",
        "research_only": True,
        "live_pnl_claim": False,
        "verdict": verdict,
        "winner": winner,
        "alpha": alpha,
        "n_obs": n,
        "n_heads": len(arrays),
        "inputs_sha256": _sha256_stream(arrays),
        "dataset_sha256": dataset_sha256,
        "components": {c.name: c.detail for c in components},
        "unavailable_lanes": unavailable,
        "evidence": [
            "selection_bias_corrected",
            "anytime_valid_promotion",
            "level_shift_monitor",
            "magnitude_confidence_sequence",
            "pit_uniformity_when_supplied",
            "changepoint_localization_when_drifted",
            "proper_score_only",
        ],
    }
    return report


def honest_verdict_json(
    scores: dict[str, NDArray[np.floating]],
    *,
    pits: dict[str, NDArray[np.floating]] | None = None,
    alpha: float = 0.05,
    seed: int = 0,
    n_boot: int = 2000,
) -> str:
    """Canonical JSON of the report (sealable / diffable)."""
    return json.dumps(
        honest_verdict(scores, pits=pits, alpha=alpha, seed=seed, n_boot=n_boot),
        sort_keys=True,
        default=float,
    )
