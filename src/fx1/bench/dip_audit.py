"""dip_audit — adversarial audit of the flagship Dip Quality bench.

One real bypass fixed here: ``assert_bench_output_honest`` matched
forbidden tokens as whole ``_``-separated words, so ``unrealizedpnl``,
``mysharpe``, ``sharpeRatio`` (→ ``sharperatio``) all evaded. Matching is
now a substring check on the normalized key.

Pinned contract edges: causal detection uses only the running peak;
the recovery window is exactly ``[i+1, i+bars+1)`` — recovery at
``i+bars`` is included (boundary pinned); horizons past the data record
``None`` and are never imputed; ``unconditional_baseline`` omits horizons
without observations rather than fabricating a rate; out-of-range
probabilities and malformed inputs raise; forecasts for unknown events
fail closed; ``brier_overall`` is absent when no pairs survive.
Sealed ``dip_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["dip_audit", "dip_audit_bench"]


def _detection() -> dict[str, Any]:
    from fx1.bench.dip import detect_dip_events

    # Tape: peak 100, dip to 89 (depth 0.11 ≥ 0.10), recover at bar 3.
    closes = [100.0, 95.0, 89.0, 101.0, 105.0]
    dates = [f"d{i}" for i in range(5)]
    ev = detect_dip_events(closes, dates, "X", threshold=0.10, horizons_bars={"h1": 2, "h2": 3})
    e = ev[0]
    return {
        "n_events": len(ev),
        "depth": round(e.depth, 6),
        "h1_observable": e.recovered["h1"] is not None,
        "h1_recovered": e.recovered["h1"],  # window [3,5): bar3 recovers
        "boundary_out": _boundary(),
        "rearm": _rearm(),
        "nonfinite_raises": _raises(
            lambda: detect_dip_events([100.0, float("nan")], ["a", "b"], "X")
        ),
        "misaligned_raises": _raises(lambda: detect_dip_events([1.0], ["a", "b"], "X")),
        "bad_threshold_raises": _raises(
            lambda: detect_dip_events([1.0, 0.5], ["a", "b"], "X", threshold=1.5)
        ),
    }


def _boundary() -> dict[str, Any]:
    from fx1.bench.dip import detect_dip_events

    # Recovery exactly at i+bars is included; longer incomplete horizons
    # stay unobservable even if an earlier bar has already recovered.
    tape = [100.0, 89.0, 90.0, 101.0]
    d = [f"b{i}" for i in range(4)]
    ev = detect_dip_events(tape, d, "X", threshold=0.10, horizons_bars={"h": 1})
    inside = ev[0].recovered["h"]  # window [2,3): bar2 has not recovered
    ev2 = detect_dip_events(tape, d, "X", threshold=0.10, horizons_bars={"h": 2})
    at_edge = ev2[0].recovered["h"]  # window [2,4): bar3 recovers at the boundary
    ev3 = detect_dip_events(tape, d, "X", threshold=0.10, horizons_bars={"h": 3})
    unobservable = ev3[0].recovered["h"]  # window [2,5): bar4 is unavailable
    return {"h1": inside, "h2": at_edge, "h3": unobservable}


def _rearm() -> bool:
    from fx1.bench.dip import detect_dip_events

    # Dip, shallow recovery below peak, second dip on the same peak.
    closes = [100.0, 89.0, 92.0, 89.5, 101.0]
    dates = [f"r{i}" for i in range(5)]
    ev = detect_dip_events(closes, dates, "X", threshold=0.10, horizons_bars={"h": 10})
    return len(ev) == 2


def _raises(fn: Any) -> str:
    try:
        fn()
    except ValueError:
        return "raise:ValueError"
    return "no-raise"


def _scoring() -> dict[str, Any]:
    from fx1.bench.dip import (
        DipEvent,
        DipForecast,
        evaluate_forecasts,
        unconditional_baseline,
    )

    ev = [
        DipEvent("X", "p0", "t0", 0.12, {"h1": True, "h2": None}),
        DipEvent("X", "p1", "t1", 0.15, {"h1": False, "h2": True}),
    ]
    baseline = unconditional_baseline(ev)
    fc = [
        DipForecast("X", "t0", {"h1": 0.9, "h2": 0.5}),
        DipForecast("X", "t1", {"h1": 0.1, "h2": 0.8}),
    ]
    metrics = evaluate_forecasts(ev, fc, n_bins=4)
    out = {
        "baseline_h1": baseline["h1"],  # 1 of 2 observable
        "baseline_h2": baseline["h2"],  # 1 of 1 observable
        # t0's h2 flag is None (unobservable): only t1's forecast counts.
        "h2_skips_unobservable": metrics.get("n_h2") == 1.0,
        "ghost_rejected": _raises(
            lambda: evaluate_forecasts(ev, [*fc, DipForecast("GHOST", "t9", {"h1": 0.5})])
        ),
        "oor_prob_raises": _raises(
            lambda: evaluate_forecasts(ev, [DipForecast("X", "t0", {"h1": 1.5})])
        ),
    }
    only_none = [DipEvent("X", "p", "t", 0.1, {"h1": None})]
    m2 = evaluate_forecasts(only_none, [DipForecast("X", "t", {"h1": 0.5})])
    out["brier_overall_absent_when_empty"] = "brier_overall" not in m2
    out["baseline_nan_on_empty"] = unconditional_baseline([]) == {}
    return out


def _honesty() -> dict[str, Any]:
    from fx1.bench.dip import assert_bench_output_honest

    # Probe metric keys ride in *values*: the receipt verifier's forbidden
    # scan is on mapping keys, so the literal evasion keys live in `key`.
    cases = [
        ("ratio_bare", "sharpe_annualized"),
        ("pl_underscored", "pnl_total"),
        ("pl_embedded", "unrealizedpnl"),
        ("camel_ratio", "sharpeRatio"),
        ("navlike_suffix", "mynav"),
        ("clean", "brier_overall"),
        ("guard_no_false_positive", "panel_rmse"),
    ]
    out: dict[str, Any] = {"cases": []}
    for name, key in cases:
        try:
            assert_bench_output_honest({key: 1.0})
            outcome = "accepted"
        except ValueError:
            outcome = "raise:ValueError"
        out["cases"].append({"name": name, "key": key, "outcome": outcome})
    return out


def dip_audit() -> dict[str, Any]:
    return {"detection": _detection(), "scoring": _scoring(), "honesty": _honesty()}


def dip_audit_bench() -> dict[str, Any]:
    r = dip_audit()
    d, s, h = r["detection"], r["scoring"], r["honesty"]
    ok = (
        d["n_events"] == 1
        and d["h1_recovered"] is True
        and d["boundary_out"] == {"h1": False, "h2": True, "h3": None}
        and d["rearm"] is True
        and d["nonfinite_raises"] == "raise:ValueError"
        and d["misaligned_raises"] == "raise:ValueError"
        and d["bad_threshold_raises"] == "raise:ValueError"
        and s["baseline_h1"] == 0.5
        and s["baseline_h2"] == 1.0
        and s["h2_skips_unobservable"] is True
        and s["ghost_rejected"] == "raise:ValueError"
        and s["oor_prob_raises"] == "raise:ValueError"
        and s["brier_overall_absent_when_empty"] is True
        and s["baseline_nan_on_empty"] is True
    )
    outcomes = {c["name"]: c["outcome"] for c in h["cases"]}
    ok = (
        ok
        and all(
            outcomes[n] == "raise:ValueError"
            for n in (
                "ratio_bare",
                "pl_underscored",
                "pl_embedded",
                "camel_ratio",
                "navlike_suffix",
            )
        )
        and outcomes["clean"] == "accepted"
        and outcomes["guard_no_false_positive"] == "accepted"
    )
    payload: dict[str, Any] = {
        "kind": "dip_audit",
        "schema": "dip_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Dip bench contract holds: causal detection, exact recovery "
            "window, unobservable horizons never imputed, honesty gate now "
            "substring-matched (embedded/camel evasions closed)."
            if ok
            else f"DIP BENCH DEFECT: {r}"
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
