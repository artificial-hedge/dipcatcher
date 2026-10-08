"""Symmetric fail-closed scope contract — full rejection matrix, both directions.

Every rejection path is asserted: missing token, corrupted (typo / wrong case /
wrong separator / non-string) token, unknown token, and a valid-but-unadmitted
pair in BOTH directions (pooled artifact at a per-security consumer AND
per-security artifact at a pooled/date-level consumer).  Legacy ``GARCHVol``
call sites must inherit the exact same contract.

No performance or profitability claim is made here; these are wiring tests.
"""

from __future__ import annotations

import re

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

_ACCEPTED_PAIRS = (
    (POOLED_DATE_LEVEL_SCOPE, DATE_LEVEL_PORTFOLIO_CONSUMER),
    (PER_SECURITY_SCOPE, PER_SECURITY_CONSUMER),
    (SECURITY_LEVEL_RET_1_SCOPE, SECURITY_LEVEL_RET_1_SCOPE),
    (UNIVARIATE_RETURN_SERIES_SCOPE, UNIVARIATE_RETURN_SERIES_SCOPE),
)

# Every unadmitted (artifact, consumer) pair — the rejection matrix, BOTH
# directions: pooled->per_security and per_security->pooled are both here.
_CROSS_DIRECTION_PAIRS = (
    (POOLED_DATE_LEVEL_SCOPE, PER_SECURITY_CONSUMER),
    (POOLED_DATE_LEVEL_SCOPE, SECURITY_LEVEL_RET_1_SCOPE),
    (POOLED_DATE_LEVEL_SCOPE, UNIVARIATE_RETURN_SERIES_SCOPE),
    (PER_SECURITY_SCOPE, DATE_LEVEL_PORTFOLIO_CONSUMER),
    (PER_SECURITY_SCOPE, SECURITY_LEVEL_RET_1_SCOPE),
    (PER_SECURITY_SCOPE, UNIVARIATE_RETURN_SERIES_SCOPE),
    (SECURITY_LEVEL_RET_1_SCOPE, DATE_LEVEL_PORTFOLIO_CONSUMER),
    (SECURITY_LEVEL_RET_1_SCOPE, PER_SECURITY_CONSUMER),
    (SECURITY_LEVEL_RET_1_SCOPE, UNIVARIATE_RETURN_SERIES_SCOPE),
    (UNIVARIATE_RETURN_SERIES_SCOPE, DATE_LEVEL_PORTFOLIO_CONSUMER),
    (UNIVARIATE_RETURN_SERIES_SCOPE, PER_SECURITY_CONSUMER),
    (UNIVARIATE_RETURN_SERIES_SCOPE, SECURITY_LEVEL_RET_1_SCOPE),
)

_MISSING_TOKENS = (None, "", "   ", "\t", "\n")

_CORRUPTED_TOKENS = (
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
)

_UNKNOWN_TOKENS = ("quux_scope", "market_overlay", "portfolio_scope", "zzzz")


@pytest.mark.parametrize(("artifact", "consumer"), _ACCEPTED_PAIRS)
def test_accepted_pair_round_trips(artifact: str, consumer: str) -> None:
    assert assert_scope_compatible(artifact, consumer) == artifact
    assert validate_scope_token(artifact) == artifact
    assert allowed_consumer_for(artifact) == consumer


@pytest.mark.parametrize(("artifact", "consumer"), _CROSS_DIRECTION_PAIRS)
def test_rejection_matrix_covers_every_unadmitted_pair(artifact: str, consumer: str) -> None:
    with pytest.raises(ScopeMismatchError):
        assert_scope_compatible(artifact, consumer)


def test_pooled_artifact_rejects_per_security_consumer() -> None:
    with pytest.raises(
        ScopeMismatchError,
        match="consumable only by date_level_portfolio",
    ):
        assert_scope_compatible(POOLED_DATE_LEVEL_SCOPE, PER_SECURITY_CONSUMER)


def test_per_security_artifact_rejects_pooled_consumer() -> None:
    with pytest.raises(ScopeMismatchError, match="scope mismatch: artifact="):
        assert_scope_compatible(PER_SECURITY_SCOPE, DATE_LEVEL_PORTFOLIO_CONSUMER)


def test_missing_tokens_reject_with_missing_error() -> None:
    for token in _MISSING_TOKENS:
        with pytest.raises(MissingScopeError):
            validate_scope_token(token)
        with pytest.raises(MissingScopeError):
            assert_scope_compatible(token, PER_SECURITY_CONSUMER)
        with pytest.raises(MissingScopeError):
            assert_scope_compatible(PER_SECURITY_SCOPE, token)


@pytest.mark.parametrize("token", _CORRUPTED_TOKENS)
def test_corrupted_tokens_reject_with_corrupted_error(token: object) -> None:
    with pytest.raises(CorruptedScopeError):
        validate_scope_token(token)
    with pytest.raises(CorruptedScopeError):
        assert_scope_compatible(token, PER_SECURITY_CONSUMER)
    with pytest.raises(CorruptedScopeError):
        assert_scope_compatible(PER_SECURITY_SCOPE, token)


@pytest.mark.parametrize("token", _UNKNOWN_TOKENS)
def test_unknown_tokens_reject_with_unknown_error(token: str) -> None:
    with pytest.raises(UnknownScopeError):
        validate_scope_token(token)
    with pytest.raises(UnknownScopeError):
        assert_scope_compatible(token, PER_SECURITY_CONSUMER)
    with pytest.raises(UnknownScopeError):
        assert_scope_compatible(PER_SECURITY_SCOPE, token)


def test_every_error_derives_from_volscopeerror_and_valueerror() -> None:
    for error in (MissingScopeError, CorruptedScopeError, UnknownScopeError, ScopeMismatchError):
        assert issubclass(error, VolScopeError)
        assert issubclass(error, ValueError)


def test_matrix_is_symmetric_and_single_consumer_per_artifact() -> None:
    assert set(ARTIFACT_SCOPE_TO_CONSUMER) == KNOWN_ARTIFACT_SCOPES
    assert set(ARTIFACT_SCOPE_TO_CONSUMER.values()) == KNOWN_CONSUMER_SCOPES
    assert set(ARTIFACT_SCOPE_TO_CONSUMER.values()) == {
        DATE_LEVEL_PORTFOLIO_CONSUMER,
        PER_SECURITY_CONSUMER,
        SECURITY_LEVEL_RET_1_SCOPE,
        UNIVARIATE_RETURN_SERIES_SCOPE,
    }


class _Artifact:
    def __init__(self, series_scope: object) -> None:
        self.series_scope = series_scope


def test_artifact_scope_of_and_require_helpers() -> None:
    per_security = _Artifact(PER_SECURITY_SCOPE)
    pooled = _Artifact(POOLED_DATE_LEVEL_SCOPE)
    assert artifact_scope_of(per_security) == PER_SECURITY_SCOPE
    assert require_per_security(per_security) == PER_SECURITY_SCOPE
    assert require_pooled(pooled) == POOLED_DATE_LEVEL_SCOPE
    with pytest.raises(ScopeMismatchError):
        require_per_security(pooled)
    with pytest.raises(ScopeMismatchError):
        require_pooled(per_security)
    with pytest.raises(MissingScopeError):
        artifact_scope_of(object())


def test_garchvol_delegates_to_the_symmetric_contract() -> None:
    pooled = GARCHVol()
    assert pooled.series_scope == POOLED_DATE_LEVEL_SCOPE
    pooled.assert_consumer_scope(DATE_LEVEL_PORTFOLIO_CONSUMER)
    with pytest.raises(ValueError, match="consumer_scope"):
        pooled.assert_consumer_scope("")
    with pytest.raises(ValueError, match="consumer_scope"):
        pooled.assert_consumer_scope("   ")
    with pytest.raises(ValueError, match="consumer_scope"):
        pooled.assert_consumer_scope(5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="consumable only by date_level_portfolio"):
        pooled.assert_consumer_scope(PER_SECURITY_CONSUMER)
    with pytest.raises(ValueError, match="scope mismatch"):
        GARCHVol.assert_consumer_scope(
            _Artifact(PER_SECURITY_SCOPE),
            DATE_LEVEL_PORTFOLIO_CONSUMER,  # type: ignore[arg-type]
        )
    GARCHVol.assert_consumer_scope(
        _Artifact(UNIVARIATE_RETURN_SERIES_SCOPE),
        UNIVARIATE_RETURN_SERIES_SCOPE,  # type: ignore[arg-type]
    )


def test_garchvol_rejects_corrupted_and_unknown_artifact_scopes() -> None:
    with pytest.raises(CorruptedScopeError):
        GARCHVol.assert_consumer_scope(
            _Artifact("PER_SECURITY"),
            PER_SECURITY_CONSUMER,  # type: ignore[arg-type]
        )
    with pytest.raises(UnknownScopeError):
        GARCHVol.assert_consumer_scope(
            _Artifact("quux_scope"),
            PER_SECURITY_CONSUMER,  # type: ignore[arg-type]
        )


def test_pooled_artifacts_cannot_reach_per_security_risk_consumers_end_to_end() -> None:
    """Real pooled model -> real per-security risk consumer boundary."""
    rng = np.random.default_rng(20261007)
    model = GARCHVol()
    model.fit_returns(np.asarray(rng.normal(0.0, 0.01, size=400)))
    with pytest.raises(ScopeMismatchError):
        require_per_security(model)
    with pytest.raises(ValueError, match="consumable only by date_level_portfolio"):
        model.assert_consumer_scope(PER_SECURITY_CONSUMER)
    assert model.assert_consumer_scope(DATE_LEVEL_PORTFOLIO_CONSUMER) is None


def test_mismatch_messages_are_directional_and_stable() -> None:
    with pytest.raises(ScopeMismatchError) as pooled_error:
        assert_scope_compatible(POOLED_DATE_LEVEL_SCOPE, PER_SECURITY_CONSUMER)
    assert re.search("consumable only by date_level_portfolio", str(pooled_error.value))
    with pytest.raises(ScopeMismatchError) as per_security_error:
        assert_scope_compatible(PER_SECURITY_SCOPE, DATE_LEVEL_PORTFOLIO_CONSUMER)
    assert "scope mismatch" in str(per_security_error.value)
    assert PER_SECURITY_SCOPE in str(per_security_error.value)
