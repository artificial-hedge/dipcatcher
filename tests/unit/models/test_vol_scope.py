"""Scope-rejection matrix for the symmetric, fail-closed volatility contract.

Every rejection path is covered in BOTH directions:

- pooled ``date_level_equal_weight_cross_section`` artifact at a per-security
  consumer -> rejected;
- ``per_security`` artifact at a pooled/date-level consumer -> rejected;
- missing scope token -> rejected;
- unknown scope token -> rejected;
- corrupted / misspelled / wrong-case / non-string scope token -> rejected.

All fixtures are seeded SYNTHETIC strings; nothing here is market evidence and
nothing here is a live-trading or profitability claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.vol_scope import (
    ARTIFACT_SCOPE_TO_CONSUMER,
    DATE_LEVEL_PORTFOLIO_CONSUMER,
    KNOWN_ARTIFACT_SCOPES,
    KNOWN_CONSUMER_SCOPES,
    PER_SECURITY_CONSUMER,
    PER_SECURITY_SCOPE,
    POOLED_DATE_LEVEL_SCOPE,
    SECURITY_LEVEL_RET_1_SCOPE,
    UNIVARIATE_RETURN_SERIES_SCOPE,
    CorruptedScopeError,
    MissingScopeError,
    ScopeMismatchError,
    UnknownScopeError,
    VolScopeError,
    allowed_consumer_for,
    artifact_scope_of,
    assert_scope_compatible,
    require_per_security,
    require_pooled,
    validate_scope_token,
)
from quant_fund.models.volatility import GARCHVol

# Admitted pairs only (the whole matrix, positive side).
_ACCEPTED_PAIRS = [
    (POOLED_DATE_LEVEL_SCOPE, DATE_LEVEL_PORTFOLIO_CONSUMER),
    (PER_SECURITY_SCOPE, PER_SECURITY_CONSUMER),
    (SECURITY_LEVEL_RET_1_SCOPE, SECURITY_LEVEL_RET_1_SCOPE),
    (UNIVARIATE_RETURN_SERIES_SCOPE, UNIVARIATE_RETURN_SERIES_SCOPE),
]

# Every unadmitted pair across the two live scopes (the whole matrix, negative side).
_CROSS_DIRECTION_PAIRS = [
    (POOLED_DATE_LEVEL_SCOPE, PER_SECURITY_CONSUMER),
    (POOLED_DATE_LEVEL_SCOPE, SECURITY_LEVEL_RET_1_SCOPE),
    (POOLED_DATE_LEVEL_SCOPE, UNIVARIATE_RETURN_SERIES_SCOPE),
    (PER_SECURITY_SCOPE, DATE_LEVEL_PORTFOLIO_CONSUMER),
    (PER_SECURITY_SCOPE, SECURITY_LEVEL_RET_1_SCOPE),
    (PER_SECURITY_SCOPE, UNIVARIATE_RETURN_SERIES_SCOPE),
]

_MISSING_TOKENS = [None, "", "   ", "\t", "\n"]
_CORRUPTED_TOKENS = [
    "PER_SECURITY",
    "per-security",
    "per_securty",
    "Per_Security",
    "per__security",
    "per_security_",
    "date_level_equal_weight_cross_sectoin",
    "Date-Level-Equal-Weight-Cross-Section",
    5,
    b"per_security",
    ["per_security"],
]
_UNKNOWN_TOKENS = ["quux_scope", "market_overlay", "portfolio_scope", "zzzz"]


@pytest.mark.parametrize(("artifact", "consumer"), _ACCEPTED_PAIRS)
def test_scope_matrix_admits_registered_pairs(artifact: str, consumer: str) -> None:
    """SYNTHETIC contract check: each registered (artifact, consumer) pair passes."""
    assert assert_scope_compatible(artifact, consumer) == artifact


@pytest.mark.parametrize(("artifact", "consumer"), _CROSS_DIRECTION_PAIRS)
def test_scope_matrix_rejects_every_cross_pair(artifact: str, consumer: str) -> None:
    """SYNTHETIC contract check: no cross-scope pair is admitted."""
    with pytest.raises(VolScopeError):
        assert_scope_compatible(artifact, consumer)


def test_pooled_artifact_rejects_per_security_consumer() -> None:
    """Direction 1: pooled date-level artifacts reject per-security consumers."""
    with pytest.raises(ScopeMismatchError, match="date_level_portfolio"):
        assert_scope_compatible(POOLED_DATE_LEVEL_SCOPE, PER_SECURITY_CONSUMER)
    # ...and only the pooled consumer is admitted.
    assert assert_scope_compatible(POOLED_DATE_LEVEL_SCOPE, DATE_LEVEL_PORTFOLIO_CONSUMER)


def test_per_security_artifact_rejects_pooled_consumer() -> None:
    """Direction 2: per_security artifacts reject pooled/date-level consumers."""
    with pytest.raises(ScopeMismatchError, match="scope mismatch"):
        assert_scope_compatible(PER_SECURITY_SCOPE, DATE_LEVEL_PORTFOLIO_CONSUMER)
    with pytest.raises(ScopeMismatchError):
        assert_scope_compatible(PER_SECURITY_SCOPE, SECURITY_LEVEL_RET_1_SCOPE)
    # ...and only the per-security consumer is admitted.
    assert assert_scope_compatible(PER_SECURITY_SCOPE, PER_SECURITY_CONSUMER)


@pytest.mark.parametrize("token", _MISSING_TOKENS)
def test_missing_scope_token_rejects_at_both_consumers(token: object) -> None:
    for consumer in (PER_SECURITY_CONSUMER, DATE_LEVEL_PORTFOLIO_CONSUMER):
        with pytest.raises(MissingScopeError):
            assert_scope_compatible(token, consumer)


@pytest.mark.parametrize("token", _UNKNOWN_TOKENS)
def test_unknown_scope_token_rejects_at_both_consumers(token: str) -> None:
    for consumer in (PER_SECURITY_CONSUMER, DATE_LEVEL_PORTFOLIO_CONSUMER):
        with pytest.raises(UnknownScopeError):
            assert_scope_compatible(token, consumer)


@pytest.mark.parametrize("token", _CORRUPTED_TOKENS)
def test_corrupted_scope_token_rejects_at_both_consumers(token: object) -> None:
    for consumer in (PER_SECURITY_CONSUMER, DATE_LEVEL_PORTFOLIO_CONSUMER):
        with pytest.raises((CorruptedScopeError, UnknownScopeError, MissingScopeError)):
            assert_scope_compatible(token, consumer)


def test_corrupted_renderings_raise_the_corrupted_error_specifically() -> None:
    for token in ("PER_SECURITY", "per-security", "per_securty", b"per_security", 5):
        with pytest.raises(CorruptedScopeError):
            validate_scope_token(token)


def test_missing_scope_token_raises_the_missing_error_specifically() -> None:
    for token in _MISSING_TOKENS:
        with pytest.raises(MissingScopeError):
            validate_scope_token(token)


def test_unknown_scope_token_raises_the_unknown_error_specifically() -> None:
    for token in _UNKNOWN_TOKENS:
        with pytest.raises(UnknownScopeError):
            validate_scope_token(token)


@pytest.mark.parametrize("bad", [None, "", "   ", 5, b"per_security"])
def test_bad_consumer_tokens_reject(bad: object) -> None:
    with pytest.raises(VolScopeError):
        assert_scope_compatible(PER_SECURITY_SCOPE, bad)


@pytest.mark.parametrize("bad", ["day_level_portfolio", "per_security_risk", ""])
def test_unknown_or_missing_consumer_tokens_reject(bad: str) -> None:
    with pytest.raises(VolScopeError):
        assert_scope_compatible(PER_SECURITY_SCOPE, bad)


def test_scope_errors_are_value_errors_for_legacy_fail_closed_call_sites() -> None:
    for exc in (MissingScopeError, CorruptedScopeError, UnknownScopeError, ScopeMismatchError):
        assert issubclass(exc, VolScopeError)
        assert issubclass(exc, ValueError)


def test_validate_scope_token_returns_exact_registered_token() -> None:
    assert validate_scope_token(PER_SECURITY_SCOPE) == PER_SECURITY_SCOPE
    assert validate_scope_token(POOLED_DATE_LEVEL_SCOPE) == POOLED_DATE_LEVEL_SCOPE
    # Surrounding whitespace is normalized the way GARCHVol normalizes it.
    assert validate_scope_token(f"  {PER_SECURITY_SCOPE}  ") == PER_SECURITY_SCOPE


def test_allowed_consumer_for_matches_the_documented_matrix() -> None:
    assert allowed_consumer_for(POOLED_DATE_LEVEL_SCOPE) == DATE_LEVEL_PORTFOLIO_CONSUMER
    assert allowed_consumer_for(PER_SECURITY_SCOPE) == PER_SECURITY_CONSUMER
    for artifact, consumer in ARTIFACT_SCOPE_TO_CONSUMER.items():
        assert allowed_consumer_for(artifact) == consumer
    assert frozenset(ARTIFACT_SCOPE_TO_CONSUMER) == KNOWN_ARTIFACT_SCOPES
    assert frozenset(ARTIFACT_SCOPE_TO_CONSUMER.values()) == KNOWN_CONSUMER_SCOPES


def test_artifact_scope_of_rejects_missing_and_corrupted_attributes() -> None:
    class _Artifact:
        series_scope = PER_SECURITY_SCOPE

    class _Missing:
        pass

    class _Corrupted:
        series_scope = "per_securty"

    assert artifact_scope_of(_Artifact()) == PER_SECURITY_SCOPE
    with pytest.raises(MissingScopeError):
        artifact_scope_of(_Missing())
    with pytest.raises(CorruptedScopeError):
        artifact_scope_of(_Corrupted())


def test_require_helpers_enforce_both_directions() -> None:
    class _Pooled:
        series_scope = POOLED_DATE_LEVEL_SCOPE

    class _PerSecurity:
        series_scope = PER_SECURITY_SCOPE

    assert require_per_security(_PerSecurity()) == PER_SECURITY_SCOPE
    assert require_pooled(_Pooled()) == POOLED_DATE_LEVEL_SCOPE
    with pytest.raises(ScopeMismatchError):
        require_per_security(_Pooled())
    with pytest.raises(ScopeMismatchError):
        require_pooled(_PerSecurity())


def test_garchvol_delegates_to_the_symmetric_contract() -> None:
    """Legacy GARCHVol entry point inherits the fail-closed matrix."""
    pooled = GARCHVol(series_scope=POOLED_DATE_LEVEL_SCOPE)
    per_security = GARCHVol(series_scope=PER_SECURITY_SCOPE)
    with pytest.raises(VolScopeError):
        pooled.assert_consumer_scope(PER_SECURITY_CONSUMER)
    with pytest.raises(VolScopeError):
        per_security.assert_consumer_scope(DATE_LEVEL_PORTFOLIO_CONSUMER)
    pooled.assert_consumer_scope(DATE_LEVEL_PORTFOLIO_CONSUMER)
    per_security.assert_consumer_scope(PER_SECURITY_CONSUMER)


def test_garchvol_rejects_corrupted_and_unknown_artifact_scopes() -> None:
    for token in ("per_securty", "PER_SECURITY", "quux_scope"):
        model = GARCHVol(series_scope=token)
        with pytest.raises(VolScopeError):
            model.assert_consumer_scope(PER_SECURITY_CONSUMER)


def test_pooled_artifacts_cannot_reach_per_security_risk_consumers_end_to_end() -> None:
    """The wrong-scope per-asset consumer path is closed with real instances.

    SYNTHETIC seeded returns; this is a wiring-correctness test only.
    """
    rng = np.random.default_rng(20261007)  # SYNTHETIC
    returns = rng.normal(0.0, 0.01, 240)
    pooled = GARCHVol(series_scope=POOLED_DATE_LEVEL_SCOPE, min_obs=20).fit_returns(returns)
    with pytest.raises(ScopeMismatchError):
        require_per_security(pooled)
    require_pooled(pooled)
