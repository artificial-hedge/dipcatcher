"""label_stability — how many labels flip when sigma is perturbed?

``triple_barrier`` converts a volatility estimate into barrier distances; a
mis-estimated ``sigma`` (or a knife-edge price path) flips labels, and every
flipped label silently rewrites the meta-labeling and sample-weight chain
downstream. This lane measures the flip rate under symmetric ``±eps``
perturbations of sigma — the honest fragility metric for the labeling layer.

Two arms calibrate the metric:

- a calm random-walk tape at reasonable barrier distances — low flip rate;
- a contrived knife-edge tape hovering just inside the upper barrier — most
  events must flip under a downward sigma perturbation (detection sanity).

The bench reports per-arm flip rates and an ``ok`` verdict iff the calm arm
is stable AND the knife-edge arm is detected.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.labels.barriers import triple_barrier
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["flip_rate", "label_stability_bench"]


def _observable(labels: np.ndarray) -> np.ndarray:
    return np.isfinite(labels)


def flip_rate(
    close: np.ndarray,
    sigma: np.ndarray,
    events: np.ndarray,
    pt: float,
    sl: float,
    horizon: int,
    *,
    eps: float = 0.05,
) -> dict[str, Any]:
    """Fraction of observable labels that change under ``sigma * (1 ± eps)``.

    Direction split reported separately — shrinking sigma tightens barriers
    (flips toward touches), widening loosens them (flips toward horizon).
    """
    base = triple_barrier(close, events, pt, sl, horizon, sigma)
    hi = triple_barrier(close, events, pt, sl, horizon, sigma * (1.0 + eps))
    lo = triple_barrier(close, events, pt, sl, horizon, sigma * (1.0 - eps))
    mask = _observable(base["label"])
    n_obs = int(mask.sum())
    if n_obs == 0:
        return {
            "n_events": int(events.size),
            "n_observable": 0,
            "flip_hi": np.nan,
            "flip_lo": np.nan,
        }
    flip_hi = float(np.mean(hi["label"][mask] != base["label"][mask]))
    flip_lo = float(np.mean(lo["label"][mask] != base["label"][mask]))
    touch_hi = float(np.mean(hi["touch"][mask] != base["touch"][mask]))
    touch_lo = float(np.mean(lo["touch"][mask] != base["touch"][mask]))
    return {
        "n_events": int(events.size),
        "n_observable": n_obs,
        "flip_hi": flip_hi,
        "flip_lo": flip_lo,
        "touch_flip_hi": touch_hi,
        "touch_flip_lo": touch_lo,
    }


def _calm_tape(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rets = rng.normal(0.0, 0.008, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    sigma = np.full(n, 0.008)
    events = np.arange(0, n - 60, 12, dtype=np.intp)
    return close, sigma, events


def _knife_edge_tape(
    n: int, pt: float, sigma_level: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A path that closes repeatedly just under the upper barrier."""
    close = np.full(n, 100.0)
    events = np.arange(0, n - 40, 8, dtype=np.intp)
    for t in events:
        # At each event, jump to just below (pt * sigma) then return — the
        # 5%-tighter barrier must flip the touch outcome.
        close[t + 1] = 100.0 * (1.0 + pt * sigma_level * 0.97)
        if t + 2 < n:
            close[t + 2] = 100.0
    sigma = np.full(n, sigma_level)
    return close, sigma, events


def label_stability_bench(seed: int = 0, eps: float = 0.05) -> dict[str, Any]:
    """Calm arm must be stable; knife-edge arm must flip — sealed receipt."""
    rng = np.random.default_rng(seed)
    pt, sl, horizon = 1.5, 1.0, 30
    close, sigma, events = _calm_tape(720, rng)
    calm = flip_rate(close, sigma, events, pt, sl, horizon, eps=eps)

    ke_close, ke_sigma, ke_events = _knife_edge_tape(720, pt, 0.008)
    edge = flip_rate(ke_close, ke_sigma, ke_events, pt, sl, horizon, eps=eps)

    ok = (
        calm["n_observable"] > 0
        and calm["flip_hi"] < 0.15
        and calm["flip_lo"] < 0.15
        and edge["flip_lo"] > 0.5
    )
    payload: dict[str, Any] = {
        "kind": "label_stability",
        "schema": "label_stability.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "invariant": "labels robust to small sigma perturbations; knife-edge detection works",
            "eps": eps,
            "pt": pt,
            "sl": sl,
            "horizon": horizon,
            "calm": calm,
            "knife_edge": edge,
            "ok": ok,
        },
        "interpretation": (
            f"calm arm flips {calm['flip_hi']:.1%}/{calm['flip_lo']:.1%} under "
            f"sigma±{eps:.0%}; knife-edge arm flips {edge['flip_lo']:.0%} "
            "(detected). Verdict: "
            + (
                "labels are perturbation-stable"
                if ok
                else "labeling layer is fragile — audit sigma inputs"
            )
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
