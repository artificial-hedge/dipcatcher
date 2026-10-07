"""Fail-closed, symmetric volatility-scope contract (SYNTHETIC).

Volatility artifacts are keyed to exactly one *scope token*.  A consumer must
declare its own consumer token and the pair must be admitted by the
compatibility matrix below.  The contract is **symmetric**: a pooled
date-level artifact rejects per-security consumers AND a per-security artifact
rejects pooled/date-level consumers.  It is **fail-closed**: a missing, unknown
or corrupted (misspelled, wrong-case, wrong-separator, non-string) scope token
is rejected loudly — never guessed, never defaulted.

Scope tokens (artifact side)
----------------------------
- ``date_level_equal_weight_cross_section`` — date-level equal-weight
  cross-sectional pooled volatility (the market overlay).  NOT an
  asset-specific forecast.
- ``per_security`` — keyed per-instrument variance forecasts with an explicit
  key axis (see ``quant_fund.models.vol_per_security``).
- ``security_level_ret_1`` — legacy per-name namespace cloned from a pooled
  specification (``garch_name_forecasts_asof`` / ``garch_name_walk_forward``).
- ``univariate_return_series`` — generic single-series legacy namespace.

Consumer tokens
---------------
- ``date_level_portfolio`` — pooled date-level risk consumers (market overlay
  covariance scaling, market-level ES/VaR, date-level vol targeting).
- ``per_security`` — per-instrument risk consumers (per-asset risk scaling,
  per-name vol targeting, per-name ES/VaR).
- ``security_level_ret_1`` / ``univariate_return_series`` — legacy consumers
  that only consume their own namespace.

Compatibility matrix (artifact -> the ONE consumer token it admits)
-------------------------------------------------------------------
- ``date_level_equal_weight_cross_section`` -> ``date_level_portfolio``
- ``per_security`` -> ``per_security``
- ``security_level_ret_1`` -> ``security_level_ret_1``
- ``univariate_return_series`` -> ``univariate_return_series``

Anything not on this matrix raises.  All error types derive from
``VolScopeError`` which derives from ``ValueError`` so legacy fail-closed call
sites keep working.  The integration surface consumed by the pipeline lane is
documented verbatim in ``docs/VOL_SCOPE_CONTRACT.md``.

This module makes no performance or profitability claim of any kind; it only
admits or rejects consumer wiring.
"""

from __future__ import annotations

from typing import Final, NoReturn

PER_SECURITY_SCOPE: Final[str] = "per_security"
POOLED_DATE_LEVEL_SCOPE: Final[str] = "date_level_equal_weight_cross_section"
SECURITY_LEVEL_RET_1_SCOPE: Final[str] = "security_level_ret_1"
UNIVARIATE_RETURN_SERIES_SCOPE: Final[str] = "univariate_return_series"

DATE_LEVEL_PORTFOLIO_CONSUMER: Final[str] = "date_level_portfolio"
PER_SECURITY_CONSUMER: Final[str] = "per_security"

ARTIFACT_SCOPE_TO_CONSUMER: Final[dict[str, str]] = {
    POOLED_DATE_LEVEL_SCOPE: DATE_LEVEL_PORTFOLIO_CONSUMER,
    PER_SECURITY_SCOPE: PER_SECURITY_CONSUMER,
    SECURITY_LEVEL_RET_1_SCOPE: SECURITY_LEVEL_RET_1_SCOPE,
    UNIVARIATE_RETURN_SERIES_SCOPE: UNIVARIATE_RETURN_SERIES_SCOPE,
}
KNOWN_ARTIFACT_SCOPES: Final[frozenset[str]] = frozenset(ARTIFACT_SCOPE_TO_CONSUMER)
KNOWN_CONSUMER_SCOPES: Final[frozenset[str]] = frozenset(ARTIFACT_SCOPE_TO_CONSUMER.values())

_MAX_TOKEN_TYPO_DISTANCE: Final[int] = 2


class VolScopeError(ValueError):
    """Base class for every fail-closed scope rejection."""


class MissingScopeError(VolScopeError):
    """The scope token is absent (None / empty / whitespace only)."""


class CorruptedScopeError(VolScopeError):
    """The scope token is malformed or a corrupted rendering of a known token."""


class UnknownScopeError(VolScopeError):
    """The scope token is well formed but not registered."""


class ScopeMismatchError(VolScopeError):
    """Both tokens are valid but the matrix does not admit this pair."""


def _canonical_form(token: str) -> str:
    """Collapse case and separator variation to compare corrupted renderings."""
    lowered = token.strip().casefold().replace("-", "_").replace(" ", "_")
    collapsed = "_".join(part for part in lowered.split("_") if part)
    return collapsed


def _edit_distance_at_most(left: str, right: str, limit: int) -> bool:
    """Bounded Levenshtein distance check used to classify typos as corrupted."""
    if abs(len(left) - len(right)) > limit:
        return False
    prev = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        cur = [i] + [0] * len(right)
        for j, right_char in enumerate(right, start=1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if left_char == right_char else 1),
            )
        if min(cur) > limit:
            return False
        prev = cur
    return prev[-1] <= limit


def _is_corrupted_rendering(token: str, known: frozenset[str]) -> bool:
    """True when the token is a near-miss of a registered token (typo/case)."""
    canonical = _canonical_form(token)
    folded = token.casefold()
    for candidate in sorted(known):
        if _canonical_form(candidate) == canonical:
            return True
        if _edit_distance_at_most(folded, candidate, _MAX_TOKEN_TYPO_DISTANCE):
            return True
    return False


def _raise_unregistered(scope: object, token: str, known: frozenset[str], kind: str) -> NoReturn:
    """Classify an unregistered token as corrupted or unknown, then reject."""
    if _is_corrupted_rendering(token, known):
        raise CorruptedScopeError(
            f"{kind} scope token {scope!r} is a corrupted rendering of a registered "
            f"token (typo, wrong case, or wrong separator); refusing to guess"
        )
    raise UnknownScopeError(
        f"{kind} scope token {scope!r} is not registered; known tokens: {sorted(known)}"
    )


def validate_scope_token(scope: object) -> str:
    """Return an artifact scope token exactly as registered, or raise fail-closed.

    Raises ``MissingScopeError`` for None/empty/whitespace tokens,
    ``CorruptedScopeError`` for non-string tokens and near-miss renderings of
    registered tokens (``"PER_SECURITY"``, ``"per-security"``, ``"per_securty"``),
    and ``UnknownScopeError`` for well-formed unregistered tokens.
    """
    if scope is None:
        raise MissingScopeError("scope token is missing; artifacts must carry an explicit scope")
    if isinstance(scope, str) and not scope.strip():
        raise MissingScopeError("scope token is missing; artifacts must carry an explicit scope")
    if not isinstance(scope, str):
        raise CorruptedScopeError(
            f"scope token must be a string, got {type(scope).__name__}; refusing to guess"
        )
    token = scope.strip()
    if token in KNOWN_ARTIFACT_SCOPES:
        return token
    _raise_unregistered(scope, token, KNOWN_ARTIFACT_SCOPES, "artifact")


def _validate_consumer_token(consumer_scope: object) -> str:
    """Return a consumer token exactly as registered, or raise fail-closed."""
    if not isinstance(consumer_scope, str):
        raise CorruptedScopeError(
            f"consumer_scope must be a string token, got {type(consumer_scope).__name__}"
        )
    if not consumer_scope.strip():
        raise MissingScopeError("consumer_scope must be a non-empty string")
    token = consumer_scope.strip()
    if token in KNOWN_CONSUMER_SCOPES:
        return token
    _raise_unregistered(consumer_scope, token, KNOWN_CONSUMER_SCOPES, "consumer")


def _mismatch_error(artifact: str, consumer: str) -> ScopeMismatchError:
    """Build the directional mismatch error for an unadmitted token pair."""
    allowed = ARTIFACT_SCOPE_TO_CONSUMER[artifact]
    if artifact == POOLED_DATE_LEVEL_SCOPE:
        return ScopeMismatchError(
            "pooled date-level volatility artifacts are consumable only by "
            f"date_level_portfolio risk consumers; got consumer {consumer!r}"
        )
    return ScopeMismatchError(
        f"scope mismatch: artifact={artifact!r}, consumer={consumer!r}; "
        f"this artifact is consumable only by {allowed!r} consumers"
    )


def assert_scope_compatible(artifact_scope: object, consumer_scope: object) -> str:
    """Admit or reject one (artifact scope, consumer scope) pair fail-closed.

    Returns the validated artifact token when the matrix admits the pair;
    raises ``MissingScopeError`` / ``CorruptedScopeError`` / ``UnknownScopeError``
    for bad tokens and ``ScopeMismatchError`` for a valid but unadmitted pair —
    in BOTH directions (pooled artifact at a per-security consumer and
    per-security artifact at a pooled/date-level consumer).
    """
    consumer = _validate_consumer_token(consumer_scope)
    artifact = validate_scope_token(artifact_scope)
    if consumer != ARTIFACT_SCOPE_TO_CONSUMER[artifact]:
        raise _mismatch_error(artifact, consumer)
    return artifact


def allowed_consumer_for(artifact_scope: object) -> str:
    """Return the single consumer token the given artifact scope admits."""
    artifact = validate_scope_token(artifact_scope)
    return ARTIFACT_SCOPE_TO_CONSUMER[artifact]


def artifact_scope_of(artifact: object) -> str:
    """Read and validate ``artifact.series_scope`` (missing attribute rejects)."""
    return validate_scope_token(getattr(artifact, "series_scope", None))


def require_per_security(artifact: object) -> str:
    """Admit an artifact only at a per-security consumer boundary."""
    return assert_scope_compatible(artifact_scope_of(artifact), PER_SECURITY_CONSUMER)


def require_pooled(artifact: object) -> str:
    """Admit an artifact only at a pooled date-level consumer boundary."""
    return assert_scope_compatible(artifact_scope_of(artifact), DATE_LEVEL_PORTFOLIO_CONSUMER)


__all__ = [
    "ARTIFACT_SCOPE_TO_CONSUMER",
    "CorruptedScopeError",
    "DATE_LEVEL_PORTFOLIO_CONSUMER",
    "KNOWN_ARTIFACT_SCOPES",
    "KNOWN_CONSUMER_SCOPES",
    "MissingScopeError",
    "PER_SECURITY_CONSUMER",
    "PER_SECURITY_SCOPE",
    "POOLED_DATE_LEVEL_SCOPE",
    "SECURITY_LEVEL_RET_1_SCOPE",
    "ScopeMismatchError",
    "UNIVARIATE_RETURN_SERIES_SCOPE",
    "UnknownScopeError",
    "VolScopeError",
    "allowed_consumer_for",
    "artifact_scope_of",
    "assert_scope_compatible",
    "require_per_security",
    "require_pooled",
    "validate_scope_token",
]
