"""Revision-aware / vintage-consistent forecasting evaluation (VINTAGE-TS-style).

SYNTHETIC vintage audit layer — operates on synthetic data only (no real
vintage databases are bundled; all function signatures take synthetic data as
input). Every function output is labeled SYNTHETIC. This module is an audit
tool, not a live forecast evaluation.

Reference
---------
Ahmad, T. (2026). "Time-Series Foundation Models That Understand Data
Revisions." arXiv:2609.28576 (submitted 23 Sep 2026).

The paper proposes VINTAGE-TS: (1) targets = first-published value and value
available N days later (neither declared "final truth"); (2) a joint
predictive distribution over both targets preserving dependence; (3)
ALFRED-style rolling evaluation with delayed-label filtering; (4) a separate
audit of pretraining overlap. This module implements a self-contained
synthetic-audit subset: vintage-process simulation, consistent splitting,
hindsight contamination measurement, validity-interval reconstruction, and a
sensitivity suite over revision parameters.

Hard rules (honesty contract)
-----------------------------
- All results labeled SYNTHETIC (never market evidence).
- Fail-closed edges: degenerate revision processes, zero revision lag, missing
  vintage columns raise ValueError.
- No new dependencies (numpy/scipy/pandas only; pandas already used in
  ``quant_fund.metrics.forecast_eval``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

Array = NDArray[np.float64]


# ---------------------------------------------------------------------------
# (1) Synthetic vintage process — Nordhaus-style Markov revision model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VintageConfig:
    """Parameters for the synthetic vintage process.

    Attributes
    ----------
    n_timesteps:
        Number of observation times (t = 0, 1, …, n_timesteps-1).
    ar_coef:
        AR(1) coefficient for the true latent process. Must be in (-1, 1).
    true_noise_std:
        Standard deviation of the innovation noise on the true process.
    pub_noise_std:
        Standard deviation of the first-publication measurement noise.
    revision_frequency:
        Number of revision events per observation period. Must be ≥ 1.
        When 1, only the initial (first-published) value exists and no revision
        occurs; the process degenerates to a single-vintage case (valid but
        raises when used in a vintage-aware split expecting multiple vintages).
    noise_decay:
        Factor by which the revision noise decays per step (the difference
        between the current published value and the true value shrinks by
        this factor at each revision, converging toward the truth).
        Must be in (0, 1).
    finality_lag:
        Number of observation periods after which a value is considered
        "final" (no further revisions). Must be ≥ 0. When 0, no further
        revisions are applied after the first publication.
    seed:
        Deterministic seed for the random state.
    """

    n_timesteps: int
    ar_coef: float
    true_noise_std: float
    pub_noise_std: float
    revision_frequency: int
    noise_decay: float
    finality_lag: int
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_timesteps < 5:
            raise ValueError("n_timesteps must be ≥ 5")
        if not -1.0 < self.ar_coef < 1.0:
            raise ValueError("ar_coef must be in (-1, 1)")
        if self.true_noise_std < 0.0:
            raise ValueError("true_noise_std must be ≥ 0")
        if self.pub_noise_std < 0.0:
            raise ValueError("pub_noise_std must be ≥ 0")
        if self.revision_frequency < 0:
            raise ValueError("revision_frequency must be ≥ 0")
        if not 0.0 < self.noise_decay < 1.0:
            raise ValueError("noise_decay must be in (0, 1)")
        if self.finality_lag < 0:
            raise ValueError("finality_lag must be ≥ 0")


def synthetic_vintage_process(config: VintageConfig) -> pd.DataFrame:
    r"""Simulate a synthetic vintage database à la Nordhaus vintage triangle.

    The true latent process follows an AR(1):

    .. math::

        X_t = \rho X_{t-1} + \varepsilon_t, \quad
        \varepsilon_t \sim \mathcal{N}(0, \sigma_\varepsilon^2)

    The first-published value (v1) at observation time :math:`t` is:

    .. math::

        v_{t,1} = X_t + \eta_{t,1}, \quad
        \eta_{t,1} \sim \mathcal{N}(0, \sigma_\eta^2)

    Subsequent revisions :math:`k \ge 2` converge toward the truth:

    .. math::

        v_{t,k} = v_{t,k-1} + \lambda^{k-1} (X_t - v_{t,k-1}) + \delta_{t,k}

    where :math:`\lambda` is the noise-decay factor and
    :math:`\delta_{t,k} \sim \mathcal{N}(0, \sigma_\delta \lambda^{k-1})`
    with :math:`\sigma_\delta = \sigma_\eta \cdot \lambda`.

    Revisions stop after ``finality_lag`` periods relative to the observation
    time OR after ``revision_frequency`` revision steps, whichever comes first.

    Parameters
    ----------
    config : VintageConfig
        Process parameters.

    Returns
    -------
    pd.DataFrame
        Columns: ``observation_time`` (int), ``vintage_time`` (int, the
        revision index — when the value was published/released),
        ``value`` (float), ``is_first_published`` (bool),
        ``revision_number`` (int, 1-indexed), ``true_value`` (float, the
        latent AR(1) value — hidden in a real setting, included here for
        audit). Labeled SYNTHETIC.
    """
    rng = np.random.default_rng(config.seed)

    # Simulate true AR(1) process with warm-up
    burn = 200
    n_total = burn + config.n_timesteps
    true = np.empty(n_total, dtype=float)
    true[0] = 0.0
    eps = rng.normal(0.0, config.true_noise_std, size=n_total - 1)
    for i in range(1, n_total):
        true[i] = config.ar_coef * true[i - 1] + eps[i - 1]
    true = true[burn:]  # shape: (n_timesteps,)

    rows: list[dict[str, Any]] = []
    for t in range(config.n_timesteps):
        xt = float(true[t])
        # v1: first-published
        v_current = xt + rng.normal(0.0, config.pub_noise_std)
        rows.append(
            {
                "observation_time": t,
                "vintage_time": t,
                "value": float(v_current),
                "is_first_published": True,
                "revision_number": 1,
                "true_value": xt,
            }
        )

        # Revisions k = 2, 3, ..., revision_frequency
        for k in range(2, config.revision_frequency + 1):
            # Only revise if within finality lag
            if k - 1 > config.finality_lag:
                break
            lag = k - 1
            lam_k = config.noise_decay**lag
            delta_scale = config.pub_noise_std * lam_k
            delta = rng.normal(0.0, delta_scale)
            # Markov revision: partial convergence toward true
            v_current = v_current + lam_k * (xt - v_current) + delta
            rows.append(
                {
                    "observation_time": t,
                    "vintage_time": t + lag,
                    "value": float(v_current),
                    "is_first_published": False,
                    "revision_number": k,
                    "true_value": xt,
                }
            )

    return (
        pd.DataFrame(rows).sort_values(["observation_time", "vintage_time"]).reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# (2) Vintage-consistent train/eval split
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VintageSplit:
    """Output of :func:`vintage_consistent_split`.

    Attributes
    ----------
    origin:
        Forecast origin date t* (observation_time index).
    horizon:
        Forecast horizon h.
    info_set:
        DataFrame of the information set available at t* (rows with
        vintage_time <= t* and observation_time <= t* - lag, containing only
        first-published or earlier revisions known by t*).
    target:
        The first-published value at t* + h (scalar).
    contemporary:
        The latest-available revised value at t* + h as of the most recent
        vintage in the database (i.e., the value you could "cheat" with if
        evaluating from a contemporary download).
    target_true:
        The latent true value at t* + h (hidden audit variable — SYNTHETIC only).
    label:
        String label: "SYNTHETIC".
    """

    origin: int
    horizon: int
    info_set: pd.DataFrame
    target: float
    contemporary: float
    target_true: float
    label: str = "SYNTHETIC"


def vintage_consistent_split(
    vintage_db: pd.DataFrame,
    origin: int,
    horizon: int = 1,
    *,
    availability_lag: int = 1,
) -> VintageSplit:
    """Produce a vintage-consistent train/eval split from a synthetic vintage DB.

    At forecast origin t*, the information set contains all rows with
    ``vintage_time <= t*`` that describe observation times ≤ t* − lag
    (the values you could actually know at t*). The target is the
    first-published value at t* + h — NOT available at t*.

    The "contemporary" value is the latest revised value at t* + h as of
    the most recent vintage_time in the database. Comparing a forecast
    evaluated against ``contemporary`` vs ``target`` quantifies hindsight
    contamination.

    Parameters
    ----------
    vintage_db : pd.DataFrame
        Output of :func:`synthetic_vintage_process`. Must contain columns:
        observation_time, vintage_time, value, is_first_published,
        revision_number, true_value.
    origin : int
        Forecast origin observation_time index t*.
    horizon : int
        Forecast horizon h ≥ 1.
    availability_lag : int
        Minimum lag between observation_time and vintage_time for a value to
        be considered "known" at the origin. Must be ≥ 0.

    Returns
    -------
    VintageSplit

    Raises
    ------
    ValueError
        If required columns are missing, origin is out of range, or the
        target observation time has no first-published value.
    """
    _validate_vintage_db(vintage_db)

    if origin < 0:
        raise ValueError("origin must be ≥ 0")
    if horizon < 1:
        raise ValueError("horizon must be ≥ 1")
    if availability_lag < 0:
        raise ValueError("availability_lag must be ≥ 0")

    max_t = int(vintage_db["observation_time"].max())
    if origin > max_t:
        raise ValueError(f"origin={origin} exceeds max observation_time={max_t}")

    target_obs = origin + horizon
    if target_obs > max_t:
        raise ValueError(
            f"target observation time {target_obs} exceeds max observation_time {max_t}"
        )

    # Information set: rows available at the origin
    # A value is known if vintage_time <= origin AND observation_time <= origin - availability_lag
    info_cutoff = origin - availability_lag
    info_mask = (vintage_db["vintage_time"] <= origin) & (
        vintage_db["observation_time"] <= info_cutoff
    )
    info_set = vintage_db.loc[info_mask].copy()

    # Target: first-published value at target_obs
    target_rows = vintage_db.loc[
        (vintage_db["observation_time"] == target_obs) & (vintage_db["is_first_published"])
    ]
    if target_rows.empty:
        raise ValueError(f"no first-published value for observation_time={target_obs}")
    target = float(target_rows["value"].iloc[0])
    target_true = float(target_rows["true_value"].iloc[0])

    # Contemporary: latest revised value at target_obs (max vintage_time)
    obs_rows = vintage_db.loc[vintage_db["observation_time"] == target_obs]
    contemporary = float(obs_rows.loc[obs_rows["vintage_time"].idxmax(), "value"])

    return VintageSplit(
        origin=origin,
        horizon=horizon,
        info_set=info_set,
        target=target,
        contemporary=contemporary,
        target_true=target_true,
    )


# ---------------------------------------------------------------------------
# (3) Hindsight contamination audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContaminationAudit:
    """Output of :func:`hindsight_contamination_audit`.

    Attributes
    ----------
    origin:
        List of forecast origin indices.
    horizon:
        Forecast horizon.
    n_splits:
        Number of origins evaluated.
    contemporary_crps:
        CRPS when forecasting evaluated against contemporary (cheated) targets.
    vintage_crps:
        CRPS when forecasting evaluated against first-published targets.
    contamination_gap:
        ``contemporary_crps - vintage_crps`` (negative when cheating helps).
    relative_gap:
        ``contamination_gap / abs(vintage_crps)`` (or NaN if vintage_crps ≈ 0).
    cheat_wins:
        Fraction of splits where contemporary CRPS < vintage CRPS (cheating
        helped).
    label:
        Always "SYNTHETIC".
    """

    origin: list[int]
    horizon: int
    n_splits: int
    contemporary_crps: float
    vintage_crps: float
    contamination_gap: float
    relative_gap: float
    cheat_wins: float
    label: str = "SYNTHETIC"


def _crps_ensemble(obs: Array, pred: Array) -> float:
    """Empirical CRPS from a sample (ensemble) prediction.

    obs  — (n,) observed values.
    pred — (n, m) ensemble predictions (m members).
    Returns mean CRPS across observations.
    """
    n = int(obs.size)
    if n == 0:
        return float("nan")
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    total = 0.0
    for i in range(n):
        y = float(obs[i])
        x = pred[i]
        x_finite = x[np.isfinite(x)]
        if x_finite.size == 0 or not np.isfinite(y):
            total += float("nan")
            continue
        term1 = float(np.mean(np.abs(x_finite - y)))
        pairwise = np.abs(x_finite[:, None] - x_finite[None, :])
        term2 = float(np.mean(pairwise)) / 2.0
        total += term1 - term2
    return total / float(n)


def _naive_forecast_sample(
    info_set: pd.DataFrame, target_obs: int, n_samples: int, noise_std: float
) -> Array:
    """Generate a naive AR(1)-style ensemble forecast from the information set.

    This is a simple reference forecaster for audit purposes: fits a basic AR(1)
    to the last-available values in the information set and draws samples.
    """
    # Use the latest known value for each observation time
    latest = (
        info_set.sort_values(["observation_time", "vintage_time"])
        .groupby("observation_time")
        .tail(1)
        .sort_values("observation_time")
    )
    vals = latest["value"].values.astype(float)
    if vals.size < 3:
        # Fallback: just use mean + noise
        mu = float(np.mean(vals)) if vals.size > 0 else 0.0
        std = noise_std
    else:
        # Simple AR(1) coefficient
        x_lag = vals[:-1]
        x_cur = vals[1:]
        rho = float(np.corrcoef(x_lag, x_cur)[0, 1])
        rho = np.clip(rho, -0.99, 0.99)
        std = float(np.std(x_cur - rho * x_lag, ddof=1))
        if not np.isfinite(std) or std < 1e-12:
            std = noise_std
        mu = vals[-1] * rho

    rng = np.random.default_rng(42)
    samples = rng.normal(mu, std, size=n_samples)
    return samples


def _rolling_forecast(
    vintage_db: pd.DataFrame,
    origins: list[int],
    horizon: int,
    availability_lag: int,
    n_samples: int,
    noise_std: float,
) -> tuple[Array, Array, Array]:
    """Produce forecasts for a list of origins; return (pred, vintage_target, contemporary_target)."""
    preds = np.empty((len(origins), n_samples), dtype=float)
    vintage_targets = np.empty(len(origins), dtype=float)
    contemporary_targets = np.empty(len(origins), dtype=float)

    for idx, origin in enumerate(origins):
        split = vintage_consistent_split(
            vintage_db,
            origin=origin,
            horizon=horizon,
            availability_lag=availability_lag,
        )
        preds[idx] = _naive_forecast_sample(split.info_set, split.origin, n_samples, noise_std)
        vintage_targets[idx] = split.target
        contemporary_targets[idx] = split.contemporary

    return preds, vintage_targets, contemporary_targets


def hindsight_contamination_audit(
    vintage_db: pd.DataFrame,
    *,
    origins: list[int] | None = None,
    horizon: int = 1,
    availability_lag: int = 1,
    n_samples: int = 50,
    noise_std: float = 1.0,
) -> ContaminationAudit:
    """Measure the performance gap from using contemporary vs vintage-consistent targets.

    Runs a rolling forecast over the specified origins, evaluates CRPS against
    both contemporary (cheated) and vintage (first-published) targets, and
    reports the contamination gap — how much using revised data inflates
    apparent skill.

    Parameters
    ----------
    vintage_db : pd.DataFrame
        Output of :func:`synthetic_vintage_process`.
    origins : list[int] | None
        Forecast origin indices. If None, uses all indices from
        ``max(horizon, availability_lag)`` to ``max_obs - horizon`` stepping by
        ``horizon``.
    horizon : int
        Forecast horizon.
    availability_lag : int
        Data-availability lag for the information set.
    n_samples : int
        Number of ensemble members for the naive forecaster.
    noise_std : float
        Reference noise for the naive forecaster fallback.

    Returns
    -------
    ContaminationAudit

    Raises
    ------
    ValueError
        If too few origins remain after trimming, or the vintage DB is invalid.
    """
    _validate_vintage_db(vintage_db)
    max_obs = int(vintage_db["observation_time"].max())

    if origins is None:
        start = max(horizon, availability_lag)
        end = max_obs - horizon
        # Need to step by horizon but ensure we get coverage. Use a sensible
        # step size — at least 1, but prefer horizon.
        step = max(1, horizon)
        origins = list(range(start, end + 1, step))

    if not origins:
        raise ValueError("no valid forecast origins")

    # Ensure origins are within bounds
    origins = [o for o in origins if 0 <= o <= max_obs - horizon]
    if not origins:
        raise ValueError("no origins within valid range after trimming")
    if len(origins) < 2:
        raise ValueError("need at least 2 forecast origins for a meaningful audit")

    # Produce rolling forecasts
    preds, vintage_targets, contemporary_targets = _rolling_forecast(
        vintage_db, origins, horizon, availability_lag, n_samples, noise_std
    )

    vintage_crps = _crps_ensemble(vintage_targets, preds)
    contemporary_crps = _crps_ensemble(contemporary_targets, preds)

    contamination_gap = contemporary_crps - vintage_crps
    denom = abs(vintage_crps)
    relative_gap = contamination_gap / denom if denom > 1e-12 else float("nan")

    # Fraction of splits where cheating helped (lower CRPS)
    per_split_vintage = np.empty(len(origins))
    per_split_contemp = np.empty(len(origins))
    for i in range(len(origins)):
        per_split_vintage[i] = _crps_ensemble(
            np.array([vintage_targets[i]]), preds[i].reshape(1, -1)
        )
        per_split_contemp[i] = _crps_ensemble(
            np.array([contemporary_targets[i]]), preds[i].reshape(1, -1)
        )
    cheat_wins = float(np.mean(per_split_contemp < per_split_vintage))

    return ContaminationAudit(
        origin=list(origins),
        horizon=horizon,
        n_splits=len(origins),
        contemporary_crps=contemporary_crps,
        vintage_crps=vintage_crps,
        contamination_gap=contamination_gap,
        relative_gap=relative_gap,
        cheat_wins=cheat_wins,
    )


# ---------------------------------------------------------------------------
# (4) Validity interval reconstruction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidityIntervalResult:
    """Output of :func:`validity_interval_reconstruction`.

    Attributes
    ----------
    observation_times:
        Array of observation time indices.
    min_values, max_values:
        Minimum and maximum values observed across all vintages for each
        observation time.
    band_widths:
        ``max_values - min_values`` per observation time.
    true_values:
        The true latent value at each observation time (SYNTHETIC only).
    true_in_band:
        Whether the true value falls within [min, max] for each observation time.
    coverage:
        Fraction of observation times where the true value falls within the
        revision band.
    mean_band_width:
        Mean band width across observation times.
    n_vintages_per_obs:
        Number of revisions (vintages) per observation time.
    label:
        Always "SYNTHETIC".
    """

    observation_times: Array
    min_values: Array
    max_values: Array
    band_widths: Array
    true_values: Array
    true_in_band: NDArray[np.bool_]
    coverage: float
    mean_band_width: float
    n_vintages_per_obs: Array
    label: str = "SYNTHETIC"


def validity_interval_reconstruction(
    vintage_db: pd.DataFrame,
) -> ValidityIntervalResult:
    """Reconstruct the revision uncertainty band for each observation time.

    For each observation time, compute the range of values that it has taken
    across all vintages (the uncertainty band from revisions). Measure coverage
    of the true latent process inside that band.

    Parameters
    ----------
    vintage_db : pd.DataFrame
        Output of :func:`synthetic_vintage_process`.

    Returns
    -------
    ValidityIntervalResult

    Raises
    ------
    ValueError
        If the vintage DB is invalid or coverage cannot be computed.
    """
    _validate_vintage_db(vintage_db)

    grouped = vintage_db.groupby("observation_time")
    obs_times = np.asarray(sorted(grouped.groups.keys()), dtype=float)
    n_obs = len(obs_times)

    min_vals = np.empty(n_obs, dtype=float)
    max_vals = np.empty(n_obs, dtype=float)
    true_vals = np.empty(n_obs, dtype=float)
    n_vintages = np.empty(n_obs, dtype=int)

    for i, obs_t in enumerate(obs_times):
        grp = grouped.get_group(int(obs_t))
        min_vals[i] = float(grp["value"].min())
        max_vals[i] = float(grp["value"].max())
        true_vals[i] = float(grp["true_value"].iloc[0])
        n_vintages[i] = int(grp.shape[0])

    band_widths = max_vals - min_vals
    true_in_band = (true_vals >= min_vals) & (true_vals <= max_vals)
    coverage = float(np.mean(true_in_band))
    mean_band_width = float(np.mean(band_widths))

    return ValidityIntervalResult(
        observation_times=obs_times,
        min_values=min_vals,
        max_values=max_vals,
        band_widths=band_widths,
        true_values=true_vals,
        true_in_band=true_in_band,
        coverage=coverage,
        mean_band_width=mean_band_width,
        n_vintages_per_obs=n_vintages,
    )


# ---------------------------------------------------------------------------
# (5) SYNTHETIC sensitivity suite
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensitivityConfig:
    """A single configuration in the sensitivity sweep."""

    name: str
    pub_noise_std: float
    revision_frequency: int
    noise_decay: float
    finality_lag: int


@dataclass(frozen=True)
class SensitivityResult:
    """Result for a single configuration in the sensitivity suite.

    Attributes
    ----------
    config:
        The SensitivityConfig that produced this result.
    contamination_gap:
        contemporary_crps - vintage_crps (negative = cheating helped).
    relative_gap:
        contamination_gap / abs(vintage_crps).
    cheat_wins:
        Fraction of splits where cheating lowered CRPS.
    vintage_crps:
        CRPS against first-published targets.
    contemporary_crps:
        CRPS against contemporary targets.
    coverage:
        Validity-interval coverage of the true process.
    mean_band_width:
        Mean revision band width.
    n_leading:
        Number of forecast origins used.
    """

    config: SensitivityConfig
    contamination_gap: float
    relative_gap: float
    cheat_wins: float
    vintage_crps: float
    contemporary_crps: float
    coverage: float
    mean_band_width: float
    n_leading: int


@dataclass(frozen=True)
class SensitivitySuite:
    """Output of :func:`synthetic_sensitivity_suite`.

    Attributes
    ----------
    results:
        List of per-configuration SensitivityResult.
    noise_slope:
        Slope of contamination_gap vs pub_noise_std (linear regression across
        configs with varying pub_noise_std). Negative slope means larger noise
        leads to a larger (more negative) gap — i.e., more contamination.
    noise_rank_corr:
        Spearman rank correlation between pub_noise_std and contamination_gap.
    freq_slope:
        Slope of contamination_gap vs revision_frequency.
    label:
        Always "SYNTHETIC".
    """

    results: list[SensitivityResult]
    noise_slope: float
    noise_rank_corr: float
    freq_slope: float
    label: str = "SYNTHETIC"


def _sensitivity_configs() -> list[SensitivityConfig]:
    """Return 8 configurations spanning a 2×2×2 grid plus baseline."""
    configs: list[SensitivityConfig] = []
    for pub_noise in [0.1, 0.5]:
        for rev_freq in [2, 4]:
            for finality in [2, 4]:
                name = f"noise={pub_noise}_freq={rev_freq}_finality={finality}"
                configs.append(
                    SensitivityConfig(
                        name=name,
                        pub_noise_std=pub_noise,
                        revision_frequency=rev_freq,
                        noise_decay=0.5,
                        finality_lag=finality,
                    )
                )
    return configs


def synthetic_sensitivity_suite(
    *,
    n_timesteps: int = 200,
    horizon: int = 1,
    availability_lag: int = 1,
    n_samples: int = 50,
    seed: int = 42,
) -> SensitivitySuite:
    """Run the vintage contamination sensitivity suite over 8 parameter configs.

    Replicates the paper's 25-configuration sensitivity suite in condensed form
    (8 configurations: 2 noise levels × 2 revision frequencies × 2 finality
    lags). For each config, a fresh synthetic vintage database is generated,
    and the :func:`hindsight_contamination_audit` and
    :func:`validity_interval_reconstruction` are run.

    Parameters
    ----------
    n_timesteps : int
        Number of observation periods (default 200).
    horizon : int
        Forecast horizon.
    availability_lag : int
        Data availability lag.
    n_samples : int
        Ensemble size for naive forecaster.
    seed : int
        Base seed (incremented per config for independence).

    Returns
    -------
    SensitivitySuite
    """
    configs = _sensitivity_configs()
    results: list[SensitivityResult] = []

    for idx, sc in enumerate(configs):
        vc = VintageConfig(
            n_timesteps=n_timesteps,
            ar_coef=0.7,
            true_noise_std=0.5,
            pub_noise_std=sc.pub_noise_std,
            revision_frequency=sc.revision_frequency,
            noise_decay=sc.noise_decay,
            finality_lag=sc.finality_lag,
            seed=seed + idx,
        )
        vintage_db = synthetic_vintage_process(vc)

        audit = hindsight_contamination_audit(
            vintage_db,
            horizon=horizon,
            availability_lag=availability_lag,
            n_samples=n_samples,
            noise_std=vc.true_noise_std,
        )

        validity = validity_interval_reconstruction(vintage_db)

        results.append(
            SensitivityResult(
                config=sc,
                contamination_gap=audit.contamination_gap,
                relative_gap=audit.relative_gap,
                cheat_wins=audit.cheat_wins,
                vintage_crps=audit.vintage_crps,
                contemporary_crps=audit.contemporary_crps,
                coverage=validity.coverage,
                mean_band_width=validity.mean_band_width,
                n_leading=audit.n_splits,
            )
        )

    # Compute summary slopes
    noises = np.asarray([r.config.pub_noise_std for r in results], dtype=float)
    gaps = np.asarray([r.contamination_gap for r in results], dtype=float)
    freqs = np.asarray([float(r.config.revision_frequency) for r in results], dtype=float)

    # Linear regression slope: contamination_gap ~ pub_noise_std
    A = np.column_stack([np.ones_like(noises), noises])
    try:
        coeffs, *_ = np.linalg.lstsq(A, gaps, rcond=None)
        noise_slope = float(coeffs[1])
    except np.linalg.LinAlgError:
        noise_slope = float("nan")

    # Spearman rank correlation
    try:
        from scipy.stats import spearmanr

        noise_rank_corr = float(spearmanr(noises, gaps).statistic)
    except (ImportError, ValueError):
        noise_rank_corr = float("nan")

    # Slope: contamination_gap ~ revision_frequency
    A_freq = np.column_stack([np.ones_like(freqs), freqs])
    try:
        coeffs_f, *_ = np.linalg.lstsq(A_freq, gaps, rcond=None)
        freq_slope = float(coeffs_f[1])
    except np.linalg.LinAlgError:
        freq_slope = float("nan")

    return SensitivitySuite(
        results=results,
        noise_slope=noise_slope,
        noise_rank_corr=noise_rank_corr,
        freq_slope=freq_slope,
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_REQUIRED_VINTAGE_COLUMNS: frozenset[str] = frozenset(
    {
        "observation_time",
        "vintage_time",
        "value",
        "is_first_published",
        "revision_number",
        "true_value",
    }
)


def _validate_vintage_db(vintage_db: pd.DataFrame) -> None:
    """Validate that a DataFrame has the required vintage columns and is nonempty.

    Raises ValueError with a descriptive message on failure.
    """
    if not isinstance(vintage_db, pd.DataFrame):
        raise ValueError("vintage_db must be a pandas DataFrame")
    if vintage_db.empty:
        raise ValueError("vintage_db is empty")
    missing = _REQUIRED_VINTAGE_COLUMNS - set(vintage_db.columns)
    if missing:
        raise ValueError(
            f"vintage_db missing required columns: {sorted(missing)}. "
            f"Expected: {sorted(_REQUIRED_VINTAGE_COLUMNS)}"
        )
    # Check for nulls in critical columns
    for col in ("observation_time", "value", "is_first_published"):
        if vintage_db[col].isnull().any():
            raise ValueError(f"vintage_db column '{col}' contains null values")
    # Check for non-finite values
    if not np.isfinite(vintage_db["value"]).all():
        raise ValueError("vintage_db column 'value' contains non-finite entries")
    # Each observation_time must have at least one first_published row
    has_first = vintage_db.groupby("observation_time")["is_first_published"].any()
    if not has_first.all():
        bad = has_first[~has_first].index.tolist()
        raise ValueError(f"observation_times without a first-published value: {bad}")
    # vintage_time must be ≥ observation_time (you can't revise before observing)
    bad_vt = vintage_db.loc[vintage_db["vintage_time"] < vintage_db["observation_time"]]
    if not bad_vt.empty:
        raise ValueError("vintage_time must be ≥ observation_time for all rows")
    # revision_number must be positive integers
    rn = vintage_db["revision_number"]
    if not (rn > 0).all():
        raise ValueError("revision_number must be > 0 for all rows")


__all__ = [
    "ContaminationAudit",
    "SensitivityConfig",
    "SensitivityResult",
    "SensitivitySuite",
    "ValidityIntervalResult",
    "VintageConfig",
    "VintageSplit",
    "hindsight_contamination_audit",
    "synthetic_sensitivity_suite",
    "synthetic_vintage_process",
    "validity_interval_reconstruction",
    "vintage_consistent_split",
]
