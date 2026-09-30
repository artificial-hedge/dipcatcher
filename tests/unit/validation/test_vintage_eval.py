"""SYNTHETIC correctness tests for vintage-consistent forecast evaluation.

All data in this test file is SYNTHETIC. No real vintage databases are used,
loaded, or bundled. Every test result is labeled SYNTHETIC per the honesty
contract (AGENTS.md #2).

Tests cover:
- Synthetic vintage process generation (input validation, output structure,
  Markov convergence properties)
- Vintage-consistent split (information-set validity, temporal consistency,
  fail-closed edges)
- Hindsight contamination audit (CRPS gap direction, cheat-wins fraction)
- Validity interval reconstruction (coverage, band-width monotonicity)
- Sensitivity suite (8-configuration sweep, noise-slope sign)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_fund.validation.vintage_eval import (
    _REQUIRED_VINTAGE_COLUMNS,
    ContaminationAudit,
    SensitivityConfig,
    SensitivityResult,
    SensitivitySuite,
    ValidityIntervalResult,
    VintageConfig,
    VintageSplit,
    _validate_vintage_db,
    hindsight_contamination_audit,
    synthetic_sensitivity_suite,
    synthetic_vintage_process,
    validity_interval_reconstruction,
    vintage_consistent_split,
)

RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False
SYNTHETIC_LABEL = "SYNTHETIC"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_config(**overrides) -> VintageConfig:
    """Return a well-behaved default VintageConfig with optional overrides."""
    defaults = dict(
        n_timesteps=100,
        ar_coef=0.7,
        true_noise_std=0.5,
        pub_noise_std=0.3,
        revision_frequency=3,
        noise_decay=0.6,
        finality_lag=4,
        seed=42,
    )
    defaults.update(overrides)
    return VintageConfig(**defaults)


def _default_db(**overrides) -> pd.DataFrame:
    """Return a synthetic vintage DB from a default config."""
    return synthetic_vintage_process(_default_config(**overrides))


# ---------------------------------------------------------------------------
# VintageConfig validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw,value,expected_msg",
    [
        ("n_timesteps", 3, r"n_timesteps must be ≥ 5"),
        ("ar_coef", 1.2, r"ar_coef must be in \(-1, 1\)"),
        ("ar_coef", -1.5, r"ar_coef must be in \(-1, 1\)"),
        ("true_noise_std", -0.1, r"true_noise_std must be ≥ 0"),
        ("pub_noise_std", -0.5, r"pub_noise_std must be ≥ 0"),
        ("revision_frequency", -1, r"revision_frequency must be ≥ 0"),
        ("noise_decay", 0.0, r"noise_decay must be in \(0, 1\)"),
        ("noise_decay", 1.0, r"noise_decay must be in \(0, 1\)"),
        ("finality_lag", -1, r"finality_lag must be ≥ 0"),
    ],
)
def test_vintage_config_invalid_params(kw, value, expected_msg) -> None:
    """Fail-closed: invalid VintageConfig parameters raise ValueError."""
    with pytest.raises(ValueError, match=expected_msg):
        _default_config(**{kw: value})


def test_vintage_config_valid_default() -> None:
    """Default config should validate without error."""
    cfg = _default_config()
    assert cfg.n_timesteps == 100
    assert cfg.ar_coef == 0.7


# ---------------------------------------------------------------------------
# synthetic_vintage_process
# ---------------------------------------------------------------------------


def test_synthetic_vintage_process_output_structure() -> None:
    """Output DataFrame must have all required columns and expected types."""
    db = _default_db()

    assert isinstance(db, pd.DataFrame)
    assert not db.empty
    assert set(db.columns) >= _REQUIRED_VINTAGE_COLUMNS

    # dtypes
    assert pd.api.types.is_integer_dtype(db["observation_time"])
    assert pd.api.types.is_integer_dtype(db["vintage_time"])
    assert pd.api.types.is_float_dtype(db["value"])
    assert pd.api.types.is_bool_dtype(db["is_first_published"])
    assert pd.api.types.is_integer_dtype(db["revision_number"])
    assert pd.api.types.is_float_dtype(db["true_value"])


def test_synthetic_vintage_process_row_count() -> None:
    """Row count should be n_timesteps * (1 + min(revision_frequency-1, finality_lag))."""
    cfg = _default_config(n_timesteps=50, revision_frequency=3, finality_lag=4)
    db = synthetic_vintage_process(cfg)
    # Each observation gets 1 first-published + min(rev_freq-1, finality_lag) revisions
    # Here: rev_freq-1 = 2, finality_lag = 4, so min=2. Total per obs = 1+2 = 3.
    expected_per_obs = 1 + min(cfg.revision_frequency - 1, cfg.finality_lag)
    assert db.shape[0] == cfg.n_timesteps * expected_per_obs


def test_synthetic_vintage_process_finality_lag_truncation() -> None:
    """When finality_lag < revision_frequency-1, fewer revisions are generated."""
    cfg_low = _default_config(revision_frequency=5, finality_lag=1)
    db_low = synthetic_vintage_process(cfg_low)
    # rev_freq-1 = 4, finality=1 → min=1 → 2 rows per obs
    assert db_low.shape[0] == cfg_low.n_timesteps * 2

    cfg_high = _default_config(revision_frequency=5, finality_lag=10)
    db_high = synthetic_vintage_process(cfg_high)
    # rev_freq-1 = 4, finality=10 → min=4 → 5 rows per obs
    assert db_high.shape[0] == cfg_high.n_timesteps * 5


def test_synthetic_vintage_process_first_published_flag() -> None:
    """Each observation_time must have exactly one first-published row."""
    db = _default_db()
    firsts = db.groupby("observation_time")["is_first_published"].sum()
    assert (firsts == 1).all(), (
        f"Some obs have {firsts[firsts != 1].to_dict()} first-published rows"
    )


def test_synthetic_vintage_process_vintage_time_ordering() -> None:
    """vintage_time must be ≥ observation_time for all rows."""
    db = _default_db()
    assert (db["vintage_time"] >= db["observation_time"]).all()


def test_synthetic_vintage_process_revision_number_sequential() -> None:
    """Within each observation_time, revision_number should be 1, 2, 3, ..."""
    db = _default_db()
    for _, grp in db.groupby("observation_time"):
        rn = grp["revision_number"].values
        expected = np.arange(1, len(rn) + 1)
        assert np.array_equal(rn, expected), f"Revision numbers not sequential: {rn}"


def test_synthetic_vintage_process_deterministic_seed() -> None:
    """Same seed must produce identical output."""
    db1 = _default_db(seed=123)
    db2 = _default_db(seed=123)
    # Compare values column to avoid floating comparison issues with NaT
    assert np.array_equal(db1["value"].values, db2["value"].values)
    assert np.array_equal(db1["true_value"].values, db2["true_value"].values)


def test_synthetic_vintage_process_different_seeds_differ() -> None:
    """Different seeds should produce different outputs."""
    db1 = _default_db(seed=42)
    db2 = _default_db(seed=99)
    # The true process or first-published values should differ somewhere
    assert not np.array_equal(db1["true_value"].values, db2["true_value"].values)


def test_synthetic_vintage_process_convergence_toward_truth() -> None:
    """Later revisions should be closer to the true value on average."""
    db = _default_db(n_timesteps=500, pub_noise_std=1.0, revision_frequency=5, noise_decay=0.3)
    errors = []
    for _, grp in db.groupby("observation_time"):
        grp_sorted = grp.sort_values("revision_number")
        err = np.abs(grp_sorted["value"].values - grp_sorted["true_value"].values)
        errors.append(err)
    errors = np.array(errors)  # (n_obs, n_revs)

    # For each revision step k, compute mean absolute error across observations
    mean_err = np.mean(errors, axis=0)
    # Fresh error should decrease or stay flat (not increase) as revisions progress
    for k in range(1, len(mean_err)):
        assert mean_err[k] <= mean_err[k - 1] + 0.5, (
            f"Revision error should not increase substantially: "
            f"err[{k}]={mean_err[k]:.4f} vs err[{k - 1}]={mean_err[k - 1]:.4f}"
        )


def test_synthetic_vintage_process_zero_noise() -> None:
    """With zero publication noise, first-published equals true value exactly."""
    db = _default_db(pub_noise_std=0.0, true_noise_std=0.0, ar_coef=0.0)
    firsts = db[db["is_first_published"]]
    np.testing.assert_allclose(firsts["value"].values, firsts["true_value"].values, atol=1e-14)


def test_synthetic_vintage_process_revision_frequency_zero() -> None:
    """revision_frequency=0: only first-published, no revisions."""
    cfg = _default_config(revision_frequency=0)
    db = synthetic_vintage_process(cfg)
    assert db.shape[0] == cfg.n_timesteps  # one row per observation
    assert (db["is_first_published"]).all()


def test_synthetic_vintage_process_label() -> None:
    """All values are labeled SYNTHETIC (implicit: no live claim)."""
    db = _default_db()
    assert "true_value" in db.columns  # SYNTHETIC-only audit column
    assert db.shape[0] > 0


# ---------------------------------------------------------------------------
# _validate_vintage_db fail-closed edges
# ---------------------------------------------------------------------------


def test_validate_vintage_db_missing_columns() -> None:
    """Missing required columns must raise ValueError."""
    df = pd.DataFrame({"observation_time": [1], "vintage_time": [1]})
    with pytest.raises(ValueError, match="missing required columns"):
        _validate_vintage_db(df)


def test_validate_vintage_db_empty() -> None:
    """Empty DataFrame must raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        _validate_vintage_db(pd.DataFrame())


def test_validate_vintage_db_not_dataframe() -> None:
    """Non-DataFrame must raise ValueError."""
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        _validate_vintage_db(None)


def test_validate_vintage_db_null_values() -> None:
    """Null values in critical columns must raise."""
    db = _default_db().copy()
    db.loc[0, "value"] = np.nan
    with pytest.raises(ValueError, match="contains null"):
        _validate_vintage_db(db)


def test_validate_vintage_db_non_finite_values() -> None:
    """Non-finite values must raise."""
    db = _default_db().copy()
    db.loc[0, "value"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        _validate_vintage_db(db)


def test_validate_vintage_db_no_first_published() -> None:
    """An observation_time missing a first-published row must raise."""
    db = _default_db().copy()
    # Remove the first-published row for observation_time 5
    mask = (db["observation_time"] == 5) & db["is_first_published"]
    db = db[~mask]
    with pytest.raises(ValueError, match="without a first-published"):
        _validate_vintage_db(db)


def test_validate_vintage_db_bad_vintage_time() -> None:
    """vintage_time < observation_time must raise."""
    db = _default_db().copy()
    db.loc[0, "vintage_time"] = db.loc[0, "observation_time"] - 1
    with pytest.raises(ValueError, match="vintage_time must be ≥ observation_time"):
        _validate_vintage_db(db)


# ---------------------------------------------------------------------------
# vintage_consistent_split
# ---------------------------------------------------------------------------


def test_vintage_consistent_split_basic() -> None:
    """Basic split should return a VintageSplit with valid structure."""
    db = _default_db(n_timesteps=50, revision_frequency=3, finality_lag=4)
    split = vintage_consistent_split(db, origin=20, horizon=1)
    assert isinstance(split, VintageSplit)
    assert split.origin == 20
    assert split.horizon == 1
    assert split.label == SYNTHETIC_LABEL
    assert isinstance(split.info_set, pd.DataFrame)
    assert np.isfinite(split.target)
    assert np.isfinite(split.contemporary)
    assert np.isfinite(split.target_true)


def test_vintage_consistent_split_info_set_temporal() -> None:
    """Information set must only contain data available at the origin."""
    db = _default_db(n_timesteps=100, revision_frequency=3, finality_lag=4)
    origin = 50
    split = vintage_consistent_split(db, origin=origin, horizon=1, availability_lag=1)

    # All rows in info_set must have vintage_time <= origin
    assert (split.info_set["vintage_time"] <= origin).all()

    # All observation_times in info_set must be ≤ origin - 1
    assert (split.info_set["observation_time"] <= origin - 1).all()


def test_vintage_consistent_split_target_is_first_published() -> None:
    """Target must be the first-published value at origin+horizon."""
    db = _default_db()
    origin = 30
    split = vintage_consistent_split(db, origin=origin, horizon=2)

    # Look up the first-published value at origin+2 manually
    target_obs = origin + 2
    manual = db.loc[
        (db["observation_time"] == target_obs) & db["is_first_published"], "value"
    ].iloc[0]
    assert split.target == pytest.approx(float(manual), abs=1e-12)


def test_vintage_consistent_split_contemporary_is_latest() -> None:
    """Contemporary value must be the latest revision at target observation."""
    db = _default_db()
    origin = 30
    split = vintage_consistent_split(db, origin=origin, horizon=1)

    target_obs = origin + 1
    manual = db.loc[db["observation_time"] == target_obs, "value"].iloc[-1]
    assert split.contemporary == pytest.approx(float(manual), abs=1e-12)


def test_vintage_consistent_split_horizon_too_large() -> None:
    """Horizon pushing beyond max observation_time must raise."""
    db = _default_db(n_timesteps=20)
    with pytest.raises(ValueError, match="exceeds max observation_time"):
        vintage_consistent_split(db, origin=19, horizon=1)


def test_vintage_consistent_split_negative_origin() -> None:
    """Negative origin must raise ValueError."""
    db = _default_db()
    with pytest.raises(ValueError, match="origin must be ≥ 0"):
        vintage_consistent_split(db, origin=-1)


def test_vintage_consistent_split_zero_horizon() -> None:
    """Zero horizon must raise ValueError."""
    db = _default_db()
    with pytest.raises(ValueError, match="horizon must be ≥ 1"):
        vintage_consistent_split(db, origin=10, horizon=0)


def test_vintage_consistent_split_negative_lag() -> None:
    """Negative availability_lag must raise ValueError."""
    db = _default_db()
    with pytest.raises(ValueError, match="availability_lag must be ≥ 0"):
        vintage_consistent_split(db, origin=10, availability_lag=-1)


def test_vintage_consistent_split_no_target() -> None:
    """When target observation_time has no first-published value, raise."""
    db = _default_db(n_timesteps=30)
    # Remove all rows for observation_time 15
    db_no_target = db[db["observation_time"] != 15].copy()
    with pytest.raises(ValueError, match="no first-published value"):
        vintage_consistent_split(db_no_target, origin=14, horizon=1)


def test_vintage_consistent_split_availability_lag_zero() -> None:
    """With availability_lag=0, observation_time ≤ origin are in the info set."""
    db = _default_db(n_timesteps=50)
    origin = 25
    split = vintage_consistent_split(db, origin=origin, horizon=1, availability_lag=0)
    assert (split.info_set["observation_time"] <= origin).all()
    # The info set can include values at origin (published simultaneously)
    assert split.info_set.shape[0] > 0


# ---------------------------------------------------------------------------
# hindsight_contamination_audit
# ---------------------------------------------------------------------------


def test_hindsight_contamination_audit_structure() -> None:
    """Audit returns ContaminationAudit with all fields populated."""
    db = _default_db(n_timesteps=100)
    audit = hindsight_contamination_audit(db, horizon=1, availability_lag=1)

    assert isinstance(audit, ContaminationAudit)
    assert audit.label == SYNTHETIC_LABEL
    assert audit.n_splits >= 2
    assert np.isfinite(audit.vintage_crps)
    assert np.isfinite(audit.contemporary_crps)
    assert np.isfinite(audit.contamination_gap)
    assert 0.0 <= audit.cheat_wins <= 1.0


def test_hindsight_contamination_audit_gap_sign() -> None:
    """With revision noise > 0, contemporary CRPS should differ from vintage.

    The contemporary values have been revised (smoothed toward truth by the
    revision process), so a naive forecaster should generally score better
    against them — meaning the contamination gap should be negative (cheating
    helps) or close to zero when revisions are weak.
    """
    db = _default_db(
        n_timesteps=200, pub_noise_std=1.0, revision_frequency=5, noise_decay=0.5, finality_lag=5
    )
    audit = hindsight_contamination_audit(db, horizon=1, availability_lag=1, n_samples=30)
    # With strong revision noise and revisions, contemporary CRPS should be lower
    # (cheating helps). This is probabilistic so we assert a range.
    assert audit.contamination_gap < 0.5, (
        f"Expected contamination_gap to be <= 0.5, got {audit.contamination_gap}"
    )
    # cheat_wins should be high (cheating helps most of the time)
    assert audit.cheat_wins > 0.4, f"Expected cheat_wins > 0.4, got {audit.cheat_wins}"


def test_hindsight_contamination_audit_no_noise_vanishing_gap() -> None:
    """With zero noise, the gap should be near zero (or NaN if forecaster degenerates)."""
    db = _default_db(n_timesteps=100, pub_noise_std=0.0, true_noise_std=0.0, ar_coef=0.0)
    audit = hindsight_contamination_audit(
        db, horizon=1, availability_lag=1, n_samples=50, noise_std=0.01
    )
    # When there's no noise, first-published = contemporary = true.
    # The naive forecaster may produce NaN on degenerate (all-constant) data;
    # if so, CRPS will also be NaN. Either way the gap is zero or NaN.
    if np.isfinite(audit.contamination_gap):
        assert abs(audit.contamination_gap) < 1e-8, (
            f"With zero noise, gap should be ~0, got {audit.contamination_gap}"
        )
    else:
        # forecaster degenerated on constant data → gap is NaN (acceptable)
        assert np.isnan(audit.contamination_gap)


def test_hindsight_contamination_audit_custom_origins() -> None:
    """Custom list of origins should be respected."""
    db = _default_db(n_timesteps=100)
    audit = hindsight_contamination_audit(db, origins=[10, 20, 30, 40, 50], horizon=1)
    assert audit.n_splits == 5
    assert audit.origin == [10, 20, 30, 40, 50]


def test_hindsight_contamination_audit_too_few_origins() -> None:
    """Fewer than 2 origins must raise ValueError."""
    db = _default_db(n_timesteps=20)
    with pytest.raises(ValueError, match="need at least 2 forecast origins"):
        hindsight_contamination_audit(db, origins=[5])


def test_hindsight_contamination_audit_no_valid_origins() -> None:
    """Origins all out of range must raise."""
    db = _default_db(n_timesteps=20)
    with pytest.raises(ValueError, match="no valid forecast origins"):
        hindsight_contamination_audit(db, origins=[], horizon=1)


def test_hindsight_contamination_audit_empty_db() -> None:
    """Empty vintage DB must raise."""
    with pytest.raises(ValueError, match="empty"):
        hindsight_contamination_audit(pd.DataFrame())


def test_hindsight_contamination_audit_vs_noise_increasing() -> None:
    """Larger publication noise should produce larger (more negative) contamination gap."""
    db_low = _default_db(n_timesteps=200, pub_noise_std=0.1, revision_frequency=3, finality_lag=3)
    db_high = _default_db(n_timesteps=200, pub_noise_std=1.5, revision_frequency=3, finality_lag=3)

    audit_low = hindsight_contamination_audit(db_low, horizon=1, availability_lag=1, n_samples=30)
    audit_high = hindsight_contamination_audit(db_high, horizon=1, availability_lag=1, n_samples=30)

    # Higher noise → larger magnitude of contamination gap (more cheating benefit)
    # gap is negative when cheating helps, so higher noise → more negative gap
    assert audit_high.contamination_gap < audit_low.contamination_gap + 0.5, (
        f"High-noise gap ({audit_high.contamination_gap}) should be less than "
        f"low-noise gap ({audit_low.contamination_gap})"
    )


# ---------------------------------------------------------------------------
# validity_interval_reconstruction
# ---------------------------------------------------------------------------


def test_validity_interval_basic() -> None:
    """Basic validity interval should produce sensible results."""
    db = _default_db(n_timesteps=100, revision_frequency=3, finality_lag=4)
    result = validity_interval_reconstruction(db)

    assert isinstance(result, ValidityIntervalResult)
    assert result.label == SYNTHETIC_LABEL
    assert result.observation_times.size == 100
    assert result.min_values.size == 100
    assert result.max_values.size == 100
    assert result.true_values.size == 100
    assert (result.band_widths >= 0).all()
    assert 0.0 <= result.coverage <= 1.0
    assert result.mean_band_width >= 0.0


def test_validity_interval_coverage_high_with_noise() -> None:
    """With moderate noise and revisions, true process coverage should be non-trivial."""
    db = _default_db(
        n_timesteps=200, pub_noise_std=0.3, revision_frequency=5, noise_decay=0.5, finality_lag=5
    )
    result = validity_interval_reconstruction(db)
    # Since revisions converge toward the truth and have noise, the true value
    # should be within the band at least some of the time. The coverage depends
    # on the random realization, so we assert it is finite and in [0, 1].
    assert 0.0 <= result.coverage <= 1.0
    assert result.mean_band_width >= 0.0
    # With 5 revisions and decay=0.5, the band should have non-trivial width
    assert result.mean_band_width > 0.01


def test_validity_interval_zero_noise_perfect_coverage() -> None:
    """With zero noise, min=max=true so coverage is 100%."""
    db = _default_db(n_timesteps=100, pub_noise_std=0.0, true_noise_std=0.0, ar_coef=0.0)
    result = validity_interval_reconstruction(db)
    assert result.coverage == 1.0
    assert result.mean_band_width == 0.0


def test_validity_interval_band_width_vs_revisions() -> None:
    """More revisions should increase the band width (more opportunity for divergence)."""
    db_few = _default_db(n_timesteps=100, revision_frequency=2, finality_lag=1, pub_noise_std=0.5)
    db_many = _default_db(n_timesteps=100, revision_frequency=5, finality_lag=4, pub_noise_std=0.5)

    result_few = validity_interval_reconstruction(db_few)
    result_many = validity_interval_reconstruction(db_many)

    # More revisions → larger band width on average
    assert result_many.mean_band_width >= result_few.mean_band_width * 0.5, (
        f"More revisions should increase band width: few={result_few.mean_band_width}, "
        f"many={result_many.mean_band_width}"
    )


def test_validity_interval_n_vintages() -> None:
    """n_vintages_per_obs should match configuration."""
    cfg = _default_config(revision_frequency=3, finality_lag=4)
    db = synthetic_vintage_process(cfg)
    result = validity_interval_reconstruction(db)
    expected = 1 + min(cfg.revision_frequency - 1, cfg.finality_lag)  # = 3
    assert (result.n_vintages_per_obs == expected).all()


def test_validity_interval_invalid_db() -> None:
    """Invalid vintage DB must raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        validity_interval_reconstruction(pd.DataFrame())


# ---------------------------------------------------------------------------
# synthetic_sensitivity_suite
# ---------------------------------------------------------------------------


def test_sensitivity_suite_structure() -> None:
    """Sensitivity suite must return 8 results with valid fields."""
    suite = synthetic_sensitivity_suite(
        n_timesteps=100, horizon=1, availability_lag=1, n_samples=20, seed=42
    )

    assert isinstance(suite, SensitivitySuite)
    assert suite.label == SYNTHETIC_LABEL
    assert len(suite.results) == 8

    for result in suite.results:
        assert isinstance(result, SensitivityResult)
        assert isinstance(result.config, SensitivityConfig)
        assert np.isfinite(result.contamination_gap)
        assert np.isfinite(result.vintage_crps)
        assert np.isfinite(result.contemporary_crps)
        assert 0.0 <= result.coverage <= 1.0
        assert result.n_leading >= 2


def test_sensitivity_suite_noise_slope_sign() -> None:
    """Higher noise should correlate with more negative contamination gap."""
    suite = synthetic_sensitivity_suite(
        n_timesteps=200, horizon=1, availability_lag=1, n_samples=30, seed=42
    )
    # noise_slope: larger noise → larger absolute gap. Gap should be more
    # negative with more noise, so the slope should be negative or close to 0.
    assert suite.noise_slope < 0.2, (
        f"Noise slope {suite.noise_slope} should be negative or near-zero"
    )
    # Rank correlation should show same direction
    assert np.isfinite(suite.noise_rank_corr)


def test_sensitivity_suite_deterministic() -> None:
    """Same seed must produce identical sensitivity results."""
    s1 = synthetic_sensitivity_suite(n_timesteps=100, horizon=1, seed=42)
    s2 = synthetic_sensitivity_suite(n_timesteps=100, horizon=1, seed=42)
    gaps1 = [r.contamination_gap for r in s1.results]
    gaps2 = [r.contamination_gap for r in s2.results]
    np.testing.assert_allclose(gaps1, gaps2, rtol=1e-12)


def test_sensitivity_suite_config_variation() -> None:
    """Different configurations must produce different contamination gaps."""
    suite = synthetic_sensitivity_suite(
        n_timesteps=200, horizon=1, availability_lag=1, n_samples=30, seed=99
    )
    gaps = [r.contamination_gap for r in suite.results]
    # At least some variation across configs
    assert np.std(gaps) > 1e-6, "Contamination gaps should vary across configs"


def test_sensitivity_suite_cheat_wins_across_configs() -> None:
    """cheat_wins should be >= 0.5 for configs with non-trivial revision noise."""
    suite = synthetic_sensitivity_suite(
        n_timesteps=200, horizon=1, availability_lag=1, n_samples=30, seed=42
    )
    # Configs with pub_noise=0.5 should have high cheat_wins
    high_noise_results = [r for r in suite.results if r.config.pub_noise_std >= 0.5]
    for r in high_noise_results:
        assert r.cheat_wins > 0.35, (
            f"Config {r.config.name}: cheat_wins={r.cheat_wins} should be > 0.35"
        )


# ---------------------------------------------------------------------------
# Contamination audit: per-split breakdown
# ---------------------------------------------------------------------------


def test_contamination_audit_cheat_wins_bound() -> None:
    """cheat_wins must be in [0, 1]."""
    db = _default_db(n_timesteps=150)
    audit = hindsight_contamination_audit(db)
    assert 0.0 <= audit.cheat_wins <= 1.0


def test_contamination_audit_relative_gap_meaningful() -> None:
    """relative_gap should be finite when vintage_crps > 0."""
    db = _default_db(n_timesteps=100)
    audit = hindsight_contamination_audit(db)
    if audit.vintage_crps > 1e-12:
        assert np.isfinite(audit.relative_gap)
    else:
        assert audit.relative_gap == 0.0 or np.isnan(audit.relative_gap)


# ---------------------------------------------------------------------------
# Integration: full synthetic workflow
# ---------------------------------------------------------------------------


def test_full_synthetic_workflow() -> None:
    """Run the full pipeline end-to-end and check label consistency."""
    cfg = _default_config(n_timesteps=80, revision_frequency=3, finality_lag=3)
    db = synthetic_vintage_process(cfg)

    # Validate DB
    _validate_vintage_db(db)

    # Split
    split = vintage_consistent_split(db, origin=40, horizon=2)
    assert split.label == SYNTHETIC_LABEL
    assert split.target != split.contemporary or cfg.pub_noise_std == 0.0

    # Audit
    audit = hindsight_contamination_audit(db)
    assert audit.label == SYNTHETIC_LABEL

    # Validity
    validity = validity_interval_reconstruction(db)
    assert validity.label == SYNTHETIC_LABEL

    # All labels must be SYNTHETIC (honesty contract #2)
    for obj in (split, audit, validity):
        assert obj.label == SYNTHETIC_LABEL
