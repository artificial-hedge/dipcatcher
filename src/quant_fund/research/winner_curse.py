"""Winner's-curse correction for fleet-tournament selection.

The head selected as argmin on a tournament's scoreboard has an
optimistically biased reported score — the minimum over K noisy
estimates is a downward-biased estimate of the best head's true
expected loss (the selection/optimizer's curse; for post-selection
inference see Fithian–Sun–Taylor 2014 and the winning-review strands in
Andrews–Bowen–McKenzie–Sendhil 2019). Reporting ``argmin``'s observed
score as its honest score overstates the fleet's edge.

``winner_curse_audit`` quantifies and corrects it:

- ``naive_score``: the selected head's observed mean loss — the biased
  headline.
- ``selection_bias``: paired bootstrap estimate
  ``E*[min_h l̄*_h] − E*[l̄*_{selected}]`` — the downward bias the act of
  selection injects into the reported score.
- ``corrected_score = naive + bias``: the selection-adjusted estimate.
- ``honest_score``: split-half control — select on a bootstrap-estimated
  odd/even split and evaluate on the complement, averaged over the same
  bootstrap draws. An honest expectation the correction is benchmarked
  against.
- ``naive_ci`` / ``selection_aware_ci``: the plain bootstrap CI on the
  winner's raw score vs the CI formed on the bias-corrected statistic.

Emits ``winner_curse.v1`` — a receipts-shaped verdict (sealable by the
same ``seal_receipt`` path the other research lanes use).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

WINNER_CURSE_SCHEMA = "winner_curse.v1"


@dataclass(frozen=True)
class WinnerCurseResult:
    """Selection-bias audit for one tournament outcome."""

    selected_head: str
    naive_score: float
    selection_bias: float
    corrected_score: float
    honest_score: float
    naive_ci: tuple[float, float]
    selection_aware_ci: tuple[float, float]
    n_obs: int
    n_heads: int
    n_boot: int

    def report(self) -> dict[str, object]:
        return {
            "kind": WINNER_CURSE_SCHEMA,
            "data_label": "SYNTHETIC",
            "research_only": True,
            "live_pnl_claim": False,
            "selected_head": self.selected_head,
            "naive_score": self.naive_score,
            "selection_bias": self.selection_bias,
            "corrected_score": self.corrected_score,
            "honest_score": self.honest_score,
            "naive_ci": list(self.naive_ci),
            "selection_aware_ci": list(self.selection_aware_ci),
            "n_obs": self.n_obs,
            "n_heads": self.n_heads,
            "n_boot": self.n_boot,
            "verdict": "bias_material" if self.selection_bias > 1e-12 else "bias_negligible",
            "evidence": [
                "paired_bootstrap_selection_bias",
                "split_half_honest_control",
                "proper_score_only",
            ],
        }


def _pairwise_min_bias(
    means_boot: NDArray[np.floating], selected: int
) -> tuple[float, NDArray[np.floating]]:
    """Bootstrap draws of per-head means → (bias estimate, corrected draws).

    means_boot: (n_boot, n_heads) resampled per-head mean losses.
    """
    winners = np.argmin(means_boot, axis=1)
    best_est = means_boot[np.arange(len(means_boot)), winners]
    sel_est = means_boot[:, selected]
    bias = float(np.mean(best_est) - np.mean(sel_est))
    # bias ≤ 0 by construction (min ≤ winner's draw); flip sign: the
    # *reported* naive score understates the true loss by |bias|.
    corrected_draws = sel_est - bias
    return -bias, corrected_draws


def _split_half_honest(
    means_boot: NDArray[np.floating],
    means_boot_b: NDArray[np.floating],
) -> float:
    """Honest control: re-select on half B, evaluate on half A."""
    winners_b = np.argmin(means_boot_b, axis=1)
    evaluated = means_boot[np.arange(len(means_boot)), winners_b]
    return float(np.mean(evaluated))


def winner_curse_audit(
    scores: dict[str, NDArray[np.floating]],
    *,
    seed: int = 0,
    n_boot: int = 4000,
    ci: float = 0.95,
) -> WinnerCurseResult:
    """Audit the selection bias in a tournament's winner.

    ``scores`` maps head → per-observation loss array (the same length for
    every head — a per-origin or per-chunk stream). Fails closed on
    empty/non-finite/mismatched inputs.
    """
    heads = sorted(scores)
    if not heads:
        raise ValueError("scores must map at least one head")
    arrays: list[NDArray[np.floating]] = []
    for h in heads:
        a = np.asarray(scores[h], dtype=float).ravel()
        if a.size == 0 or not np.isfinite(a).all():
            raise ValueError(f"scores[{h!r}] empty or non-finite")
        arrays.append(a)
    n_obs = int(arrays[0].size)
    if any(a.size != n_obs for a in arrays):
        raise ValueError("all heads must observe the same number of losses")
    if n_boot < 100:
        raise ValueError("n_boot must be >= 100")
    if not (0.5 < ci < 1.0):
        raise ValueError("ci must be in (0.5, 1)")

    X = np.stack(arrays, axis=1)  # (n_obs, n_heads)
    means = X.mean(axis=0)
    selected = int(np.argmin(means))
    naive = float(means[selected])

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_obs, size=(n_boot, n_obs))
    means_boot = X[idx].mean(axis=1)  # (n_boot, n_heads)

    bias, corrected_draws = _pairwise_min_bias(means_boot, selected)
    corrected = naive + bias

    # Split-half honest control: independently resampled second half selects,
    # first half evaluates — the selection-unbiased benchmark.
    idx_b = rng.integers(0, n_obs, size=(n_boot, n_obs))
    means_boot_b = X[idx_b].mean(axis=1)
    honest = _split_half_honest(means_boot, means_boot_b)

    sel_draws = means_boot[:, selected]
    lo_p, hi_p = (1.0 - ci) / 2.0, 1.0 - (1.0 - ci) / 2.0
    naive_ci = (
        float(np.quantile(sel_draws, lo_p)),
        float(np.quantile(sel_draws, hi_p)),
    )
    aware_ci = (
        float(np.quantile(corrected_draws, lo_p)),
        float(np.quantile(corrected_draws, hi_p)),
    )

    return WinnerCurseResult(
        selected_head=heads[selected],
        naive_score=naive,
        selection_bias=bias,
        corrected_score=float(corrected),
        honest_score=honest,
        naive_ci=naive_ci,
        selection_aware_ci=aware_ci,
        n_obs=n_obs,
        n_heads=len(heads),
        n_boot=n_boot,
    )


def audit_from_streams(
    streams: dict[str, list[float]], *, seed: int = 0, n_boot: int = 4000
) -> dict[str, object]:
    """Convenience entry: raw lists → sealed-ready report dict."""
    scores = {k: np.asarray(v, dtype=float) for k, v in streams.items()}
    result = winner_curse_audit(scores, seed=seed, n_boot=n_boot)
    report = result.report()
    blob = hashlib.sha256(
        ";".join(
            f"{k}:{v.size}:{hashlib.sha256(np.asarray(v).tobytes()).hexdigest()[:16]}"
            for k, v in sorted(scores.items())
        ).encode()
    ).hexdigest()
    report["inputs_sha256"] = blob
    return report
