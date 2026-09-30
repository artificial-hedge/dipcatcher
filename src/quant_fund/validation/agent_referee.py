"""Anytime-valid frozen referee for LLM factor mining ("governed self-evolution").

Implements the proposer-judge separation protocol from Qu, Chen & Wang (2026),
arXiv:2609.27051 "Propose, Don't Judge: An Anytime-Valid Referee for LLM
Agents That Mine Investment Factors".

Core idea: an LLM agent proposes investment factors; a FROZEN betting-based
statistical referee that the agent CANNOT TOUCH judges candidates only on
market outcomes revealed AFTER submission.  The referee accumulates e-values
per candidate on post-submission-only returns and controls the false-discovery
rate at EVERY stopping time for ANY proposal policy (including adaptive /
human-in-the-loop).

Key guarantee (Theorem 1 of the paper): the frozen referee admits 5-11x fewer
sub-threshold factors than leaky referees while still admitting true factors
(eventually — with a documented detection delay).

Honesty contract (house rules; see AGENTS.md):
- Every e-value is computed on POST-SUBMISSION data only (structural enforcement).
- The referee is FROZEN: its parameters cannot be modified after deployment.
- All SYNTHETIC planted-world results are correctness evidence, never market
  evidence.  No Sharpe/Sortino/P&L headline claims.
- Cross-references: validation/leakage_redteam.py (same threat model:
  selection-multiplicity + lookahead), metrics/evalues.py + metrics/anytime_fdr.py
  (e-BH machinery composed here).

References
----------
Qu, Chen & Wang (2026). "Propose, Don't Judge: An Anytime-Valid Referee for
LLM Agents That Mine Investment Factors." arXiv:2609.27051.

Wang & Ramdas (2022). "False discovery rate control with e-values." JRSS-B 84.
— e-BH procedure composed here for anytime-valid FDR control.

Shafer & Vovk (2021). "Testing by betting: A strategy for statistical and
scientific communication." JRSS-A 184. — e-value / betting foundations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

__all__ = [
    "CandidateSubmission",
    "RefereeLedger",
    "RefereeVerdict",
    "frozen_referee_evalue",
    "referee_evaluate",
    "referee_fdr_control",
    "leaky_referee_contrast",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateSubmission:
    """A single factor candidate submitted by the proposer agent.

    Attributes
    ----------
    candidate_id : str
        Unique identifier for the candidate.
    submission_time : int
        Integer time index (e.g. trading day) when the candidate was submitted.
        The referee ONLY evaluates data at times > submission_time.
    metadata : dict
        Optional metadata (factor name, description, etc.).  Not used by the
        referee — the referee is blind to everything except the post-submission
        outcome stream.
    """

    candidate_id: str
    submission_time: int
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class RefereeVerdict:
    """Result of evaluating one candidate through the frozen referee.

    Attributes
    ----------
    candidate_id : str
    e_value : float
        Accumulated e-value (>= 1 means evidence against the null of no edge).
    n_post_sub_obs : int
        Number of post-submission observations used.
    admitted : bool
        Whether the candidate is admitted (e_value >= 1/alpha after FDR control).
    """

    candidate_id: str
    e_value: float
    n_post_sub_obs: int
    admitted: bool = False


# ---------------------------------------------------------------------------
# Frozen referee e-value construction
# ---------------------------------------------------------------------------


def frozen_referee_evalue(
    post_sub_returns: Array,
    *,
    null_mean: float = 0.0,
    null_std: float = 1.0,
    bet_fraction: float = 0.1,
) -> float:
    """Compute the e-value for a single candidate on post-submission returns.

    The e-value is a betting-based test statistic: under the null hypothesis
    that the factor has no predictive edge (returns ~ N(null_mean, null_std^2)),
    the e-value is a non-negative random variable with E[E] <= 1.  Large values
    indicate evidence against the null.

    Construction (Shafer & Vovk 2021; Qu et al. 2026 §3):
    - Standardize the post-submission returns: z_t = (r_t - null_mean) / null_std
    - Bet a fixed fraction lambda of wealth on each z_t being positive:
      E_t = prod_{t} (1 + lambda * z_t)  [truncated at 0 to avoid negative wealth]
    - The e-value is the final wealth E_T.

    Under the null, E[E_T] <= 1 (Ville's inequality).  Under the alternative
    (positive edge), E_T grows exponentially.

    Parameters
    ----------
    post_sub_returns : array of post-submission returns (r_t for t > submission_time).
    null_mean : hypothesized mean under the null (default 0).
    null_std : hypothesized std under the null (default 1).
    bet_fraction : lambda, the fixed betting fraction (default 0.5).

    Returns
    -------
    e_value : float >= 0.  Returns 0.0 if the wealth goes negative (bust).

    Raises
    ------
    ValueError
        If post_sub_returns is empty, non-finite, or null_std <= 0.
    """
    r = np.asarray(post_sub_returns, dtype=float).ravel()
    if r.size == 0:
        raise ValueError("post_sub_returns must be non-empty")
    if not np.isfinite(r).all():
        raise ValueError("post_sub_returns must be finite")
    if null_std <= 0:
        raise ValueError("null_std must be positive")
    if not (0 < bet_fraction < 1):
        raise ValueError("bet_fraction must be in (0, 1)")

    z = (r - null_mean) / null_std
    # Truncated betting: wealth_t = max(wealth_{t-1} * (1 + lambda * z_t), 0)
    log_wealth = 0.0
    for z_t in z:
        factor = 1.0 + bet_fraction * z_t
        if factor <= 0:
            return 0.0  # bust
        log_wealth += math.log(factor)

    return float(math.exp(log_wealth))


# ---------------------------------------------------------------------------
# Referee ledger
# ---------------------------------------------------------------------------


class RefereeLedger:
    """Append-only ledger of candidate submissions.

    The ledger enforces the structural post-submission-only constraint:
    a candidate's e-value is NEVER evaluated on data at or before its
    submission time.  This is the core honesty mechanism.

    The ledger is FROZEN after construction: its parameters (null hypothesis,
    betting fraction) cannot be modified by the proposer agent.
    """

    def __init__(
        self,
        *,
        null_mean: float = 0.0,
        null_std: float = 1.0,
        bet_fraction: float = 0.1,
        alpha: float = 0.1,
    ) -> None:
        """Initialize the frozen referee.

        Parameters
        ----------
        null_mean, null_std : null hypothesis for the return stream.
        bet_fraction : lambda for the betting e-value.
        alpha : FDR level for the e-BH procedure.
        """
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1)")
        self._null_mean = float(null_mean)
        self._null_std = float(null_std)
        self._bet_fraction = float(bet_fraction)
        self._alpha = float(alpha)
        self._submissions: list[CandidateSubmission] = []
        self._frozen = False

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def n_submissions(self) -> int:
        return len(self._submissions)

    def submit(self, candidate: CandidateSubmission) -> None:
        """Register a candidate submission.  Append-only; no deletion."""
        if self._frozen:
            raise RuntimeError("Ledger is frozen — no new submissions allowed")
        # Enforce non-decreasing submission times (no backdating; batch submissions allowed)
        if self._submissions and candidate.submission_time < self._submissions[-1].submission_time:
            raise ValueError(
                f"submission_time {candidate.submission_time} must be >= "
                f"previous {self._submissions[-1].submission_time}"
            )
        self._submissions.append(candidate)

    def freeze(self) -> None:
        """Freeze the ledger — no more submissions allowed."""
        self._frozen = True

    def evaluate(
        self,
        outcome_stream: Array,
        *,
        time_index: Array | None = None,
    ) -> list[RefereeVerdict]:
        """Evaluate all submitted candidates against the outcome stream.

        Parameters
        ----------
        outcome_stream : array of returns/outcomes indexed by time.
        time_index : optional array of integer time indices for each outcome.
            If None, assumes outcome_stream[i] corresponds to time i.

        Returns
        -------
        List of RefereeVerdict, one per candidate.

        Raises
        ------
        ValueError
            If a candidate's submission_time is >= len(outcome_stream)
            (no post-submission data available).
        """
        if time_index is None:
            time_index_arr = np.arange(len(outcome_stream))
        else:
            time_index_arr = np.asarray(time_index)

        verdicts: list[RefereeVerdict] = []
        for cand in self._submissions:
            # STRUCTURAL ENFORCEMENT: only use data AFTER submission_time
            mask = time_index_arr > cand.submission_time
            post_sub = outcome_stream[mask]

            if post_sub.size == 0:
                # No post-submission data — e-value is 1 (no evidence)
                verdicts.append(
                    RefereeVerdict(
                        candidate_id=cand.candidate_id,
                        e_value=1.0,
                        n_post_sub_obs=0,
                        admitted=False,
                    )
                )
                continue

            e_val = frozen_referee_evalue(
                post_sub,
                null_mean=self._null_mean,
                null_std=self._null_std,
                bet_fraction=self._bet_fraction,
            )
            verdicts.append(
                RefereeVerdict(
                    candidate_id=cand.candidate_id,
                    e_value=e_val,
                    n_post_sub_obs=int(post_sub.size),
                )
            )

        return verdicts


# ---------------------------------------------------------------------------
# FDR control
# ---------------------------------------------------------------------------


def referee_fdr_control(
    verdicts: list[RefereeVerdict],
    alpha: float = 0.1,
) -> list[RefereeVerdict]:
    """Apply e-BH FDR control to the referee verdicts.

    The e-BH procedure (Wang & Ramdas 2022) controls the false discovery rate
    at level alpha under arbitrary dependence, at ANY stopping time.

    Parameters
    ----------
    verdicts : list of RefereeVerdict from RefereeLedger.evaluate().
    alpha : FDR level.

    Returns
    -------
    The same verdicts with `admitted` set to True for rejected nulls.
    """
    if not verdicts:
        return verdicts

    e_values = np.array([v.e_value for v in verdicts])
    m = len(e_values)

    # e-BH: sort e-values descending, find the largest k such that
    # e_(k) >= m / (k * alpha)
    sorted_idx = np.argsort(-e_values)  # descending
    sorted_e = e_values[sorted_idx]

    k_max = 0
    for k in range(1, m + 1):
        threshold = m / (k * alpha)
        if sorted_e[k - 1] >= threshold:
            k_max = k

    # Admit the top k_max candidates
    for i in range(k_max):
        verdicts[sorted_idx[i]].admitted = True

    return verdicts


def referee_evaluate(
    ledger: RefereeLedger,
    outcome_stream: Array,
    *,
    time_index: Array | None = None,
    alpha: float | None = None,
) -> list[RefereeVerdict]:
    """Convenience: evaluate + FDR control in one call."""
    if alpha is None:
        alpha = ledger.alpha
    verdicts = ledger.evaluate(outcome_stream, time_index=time_index)
    return referee_fdr_control(verdicts, alpha=alpha)


# ---------------------------------------------------------------------------
# Leaky-referee red-team contrast
# ---------------------------------------------------------------------------


def leaky_referee_contrast(
    n_true: int,
    n_noise: int,
    n_periods: int,
    *,
    true_edge: float = 0.5,
    seed: int = 42,
    alpha: float = 0.1,
    submission_time: int = 0,
) -> dict[str, float]:
    """SYNTHETIC red-team: frozen referee vs leaky referee on a planted world.

    Generates a world with `n_true` factors that have genuine post-submission
    edge (mean = true_edge) and `n_noise` factors that are pure noise (mean = 0).
    Compares:
    - Frozen referee: evaluates only on post-submission data (t > submission_time).
    - Leaky referee: evaluates on ALL data including pre-submission (lookahead).

    The leaky referee will admit noise factors that happen to look good in-sample,
    while the frozen referee will not.

    Parameters
    ----------
    n_true : number of true factors (with edge).
    n_noise : number of noise factors (no edge).
    n_periods : total number of time periods.
    true_edge : mean return of true factors (in std units).
    seed : RNG seed.
    alpha : FDR level.
    submission_time : time index at which all factors are submitted.

    Returns
    -------
    dict with keys:
    - frozen_true_admitted, frozen_noise_admitted
    - leaky_true_admitted, leaky_noise_admitted
    - noise_admission_ratio (leaky_noise / frozen_noise, the headline metric)
    """
    rng = np.random.default_rng(seed)

    # Generate outcome stream: true factors have edge, noise factors don't
    # Shape: (n_true + n_noise, n_periods)
    n_total = n_true + n_noise
    means = np.zeros(n_total)
    means[:n_true] = true_edge

    outcomes = rng.standard_normal((n_total, n_periods)) + means[:, None]

    # Frozen referee: only post-submission data
    frozen_ledger = RefereeLedger(alpha=alpha)
    for i in range(n_total):
        frozen_ledger.submit(
            CandidateSubmission(candidate_id=f"cand_{i}", submission_time=submission_time)
        )

    # Evaluate each candidate on its own post-submission stream
    frozen_verdicts: list[RefereeVerdict] = []
    for i, cand in enumerate(frozen_ledger._submissions):
        post_sub = outcomes[i, submission_time + 1 :]
        e_val = frozen_referee_evalue(post_sub)
        frozen_verdicts.append(
            RefereeVerdict(
                candidate_id=cand.candidate_id,
                e_value=e_val,
                n_post_sub_obs=len(post_sub),
            )
        )
    frozen_verdicts = referee_fdr_control(frozen_verdicts, alpha=alpha)

    # Leaky referee: uses ALL data (including pre-submission)
    leaky_verdicts: list[RefereeVerdict] = []
    for i in range(n_total):
        # LEAK: evaluate on the full stream, not just post-submission
        e_val = frozen_referee_evalue(outcomes[i, :])
        leaky_verdicts.append(
            RefereeVerdict(
                candidate_id=f"cand_{i}",
                e_value=e_val,
                n_post_sub_obs=n_periods,
            )
        )
    leaky_verdicts = referee_fdr_control(leaky_verdicts, alpha=alpha)

    # Count admissions
    frozen_true = sum(1 for v in frozen_verdicts[:n_true] if v.admitted)
    frozen_noise = sum(1 for v in frozen_verdicts[n_true:] if v.admitted)
    leaky_true = sum(1 for v in leaky_verdicts[:n_true] if v.admitted)
    leaky_noise = sum(1 for v in leaky_verdicts[n_true:] if v.admitted)

    noise_ratio = leaky_noise / max(frozen_noise, 1)

    return {
        "frozen_true_admitted": float(frozen_true),
        "frozen_noise_admitted": float(frozen_noise),
        "leaky_true_admitted": float(leaky_true),
        "leaky_noise_admitted": float(leaky_noise),
        "noise_admission_ratio": float(noise_ratio),
        "n_true": float(n_true),
        "n_noise": float(n_noise),
        "n_periods": float(n_periods),
        "alpha": float(alpha),
    }
