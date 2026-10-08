# Volatility scope contract (v1) — `per_security` vs pooled date-level

This document is the COMPLETE integration surface for wiring volatility into
`src/quant_fund/pipeline/train.py` and `src/quant_fund/pipeline/forecast.py`
(task **T6**). It names every function, argument and return type, every scope
token, every error type, and the units/PIT contracts. It is intended to need
**no redesign** — consume it verbatim.

Honesty frame: everything here is a variance-forecasting and proper-scoring
contract. Nothing here computes or claims Sharpe/Sortino/Calmar/P&L/NAV, live
trading, broker connectivity or profitability. Tests use seeded SYNTHETIC
panels which are correctness fixtures only — never market evidence.

Modules:

| Module | Role |
|---|---|
| `quant_fund.models.vol_scope` | symmetric, fail-closed scope token contract |
| `quant_fund.models.vol_per_security` | keyed per-security variance forecasts (`per_security`) |
| `quant_fund.metrics.vol_eval` | units contract + keyed QLIKE/pinball proper scores |
| `quant_fund.models.volatility` | pooled/univariate legacy models (`GARCHVol` delegates its scope check to `vol_scope`) |

---

## 1. Scope tokens

### Artifact tokens (`series_scope` on the artifact / model)

| Token | Meaning | Key axis |
|---|---|---|
| `date_level_equal_weight_cross_section` (`POOLED_DATE_LEVEL_SCOPE`) | date-level equal-weight cross-sectional pooled volatility (market overlay). **NOT an asset-specific forecast** | none (one series per date) |
| `per_security` (`PER_SECURITY_SCOPE`) | keyed per-instrument variance forecasts from `PerSecurityVol` | `symbol` (explicit keys) |
| `security_level_ret_1` (`SECURITY_LEVEL_RET_1_SCOPE`) | legacy per-name namespace cloned from a pooled spec (`garch_name_forecasts_asof`, `garch_name_walk_forward`) | `security_id` |
| `univariate_return_series` (`UNIVARIATE_RETURN_SERIES_SCOPE`) | generic single-series legacy namespace | none |

### Consumer tokens

| Token | Consumers |
|---|---|
| `date_level_portfolio` (`DATE_LEVEL_PORTFOLIO_CONSUMER`) | pooled date-level risk: market-overlay covariance scaling, market-level ES/VaR, date-level vol targeting |
| `per_security` (`PER_SECURITY_CONSUMER`) | per-instrument risk: per-asset risk scaling, per-name vol targeting, per-name ES/VaR |
| `security_level_ret_1` | legacy per-name consumers only |
| `univariate_return_series` | legacy single-series consumers only |

### Compatibility matrix (each artifact admits EXACTLY ONE consumer)

| artifact `series_scope` | admitted consumer | every other consumer |
|---|---|---|
| `date_level_equal_weight_cross_section` | `date_level_portfolio` | `ScopeMismatchError` |
| `per_security` | `per_security` | `ScopeMismatchError` |
| `security_level_ret_1` | `security_level_ret_1` | `ScopeMismatchError` |
| `univariate_return_series` | `univariate_return_series` | `ScopeMismatchError` |

Fail-closed token rules (applied to BOTH sides, in BOTH directions):

| condition | outcome |
|---|---|
| missing token (`None`, `""`, whitespace) | `MissingScopeError` |
| non-string token (`5`, `b"per_security"`, list) | `CorruptedScopeError` |
| corrupted rendering of a known token (`"PER_SECURITY"`, `"per-security"`, `"per_securty"`, `"per__security"`) | `CorruptedScopeError` |
| well-formed unregistered token (`"quux_scope"`) | `UnknownScopeError` |
| valid pair not on the matrix | `ScopeMismatchError` |

Whitespace around a token is normalized (same rule `GARCHVol` uses); case,
separators and spelling must be exact. No token is ever guessed or defaulted.

## 2. `quant_fund.models.vol_scope` — exact API

```python
PER_SECURITY_SCOPE: Final[str] = "per_security"
POOLED_DATE_LEVEL_SCOPE: Final[str] = "date_level_equal_weight_cross_section"
SECURITY_LEVEL_RET_1_SCOPE: Final[str] = "security_level_ret_1"
UNIVARIATE_RETURN_SERIES_SCOPE: Final[str] = "univariate_return_series"
DATE_LEVEL_PORTFOLIO_CONSUMER: Final[str] = "date_level_portfolio"
PER_SECURITY_CONSUMER: Final[str] = "per_security"
ARTIFACT_SCOPE_TO_CONSUMER: Final[dict[str, str]]   # the matrix above
KNOWN_ARTIFACT_SCOPES: Final[frozenset[str]]
KNOWN_CONSUMER_SCOPES: Final[frozenset[str]]

class VolScopeError(ValueError): ...          # base; ValueError keeps legacy call sites
class MissingScopeError(VolScopeError): ...
class CorruptedScopeError(VolScopeError): ...
class UnknownScopeError(VolScopeError): ...
class ScopeMismatchError(VolScopeError): ...

def validate_scope_token(scope: object) -> str: ...
    # validates ONE artifact token; returns it exactly as registered
def assert_scope_compatible(artifact_scope: object, consumer_scope: object) -> str: ...
    # validates both tokens and admits/rejects the pair; returns the artifact token
def allowed_consumer_for(artifact_scope: object) -> str: ...
    # the single consumer token the artifact admits
def artifact_scope_of(artifact: object) -> str: ...
    # validates artifact.series_scope (missing attribute -> MissingScopeError)
def require_per_security(artifact: object) -> str: ...
    # == assert_scope_compatible(artifact_scope_of(artifact), PER_SECURITY_CONSUMER)
def require_pooled(artifact: object) -> str: ...
    # == assert_scope_compatible(artifact_scope_of(artifact), DATE_LEVEL_PORTFOLIO_CONSUMER)
```

`GARCHVol.assert_consumer_scope(consumer_scope: str) -> None` delegates to
`assert_scope_compatible(self.series_scope, consumer_scope)` and is kept for
legacy call sites. `PerSecurityVol.assert_consumer_scope` does the same with
its own `series_scope="per_security"`.

## 3. `quant_fund.models.vol_per_security` — keyed forecasts

```python
KEYED_VOL_SPECS: Final[tuple[str, ...]] = ("ewma", "garch", "har", "aparch", "figarch")

class PerKeyVolError(ValueError): ...   # one-key failure -> honest NaN channel

@dataclass(frozen=True)
class KeyedReturnPanel:
    keys: NDArray[Any]                 # (n,) non-empty str — the explicit key axis
    times: NDArray[Any]                # (n,) strictly increasing integer event time PER KEY
    returns: Array                     # (n,) decimal log returns
    labels: Array | None = None        # forward realized-var labels, VARIANCE_UNITS,
                                       # indexed at their forecast origin — EVALUATION ONLY
    source_label: str = "UNSPECIFIED"  # tests MUST pass "SYNTHETIC"
    def key_names(self) -> tuple[str, ...]: ...
    def key_returns(self, key: str) -> Array: ...              # time-ascending
    def trailing(self, key: str, as_of: int) -> Array: ...     # returns with time < as_of
    def label_at(self, key: str, time: int) -> float: ...      # NaN when absent
    def realized_return_at(self, key: str, time: int) -> float: ...
    def with_labels(self, labels: Array | None) -> KeyedReturnPanel: ...
    def with_returns(self, returns: Array) -> KeyedReturnPanel: ...

class PerSecurityVol:
    series_scope: str = "per_security"
    def __init__(self, spec: str = "garch", *, lam: float = 0.94, min_obs: int = 50,
                 p: int = 1, q: int = 1, dist: str = "normal", seed: int = 0) -> None: ...
    def clone(self) -> PerSecurityVol: ...
    def fit(self, panel: KeyedReturnPanel, *, as_of: int | None = None) -> PerSecurityVol: ...
    def forecast(self, key: str, horizon: int = 1, *,
                 quantiles: Sequence[float] | None = None,
                 seed: int | None = None) -> dict[str, Any]: ...
    def forecast_keys(self, horizon: int = 1, *, keys: Sequence[str] | None = None,
                      quantiles: Sequence[float] | None = None,
                      seed: int | None = None) -> KeyedVarianceForecast: ...
    def assert_consumer_scope(self, consumer_scope: str) -> None: ...
    def diagnostics(self) -> dict[str, Any]: ...
    def metadata(self) -> ModelMeta: ...

@dataclass(frozen=True)
class KeyedVarianceForecast:
    keys: tuple[str, ...]
    variance: Array                    # (n_keys, horizon) VARIANCE_UNITS; NaN for failed keys
    status: tuple[str, ...]            # "ok" | "failed:<reason>" per key
    horizon: int
    means: Array                       # (n_keys,)
    quantiles: Array | None = None     # (n_keys, horizon, n_levels) VOLATILITY_UNITS
    units: str = VARIANCE_UNITS
    scope: str = PER_SECURITY_SCOPE
    source_label: str = "UNSPECIFIED"
    def forecast(self, key: str) -> dict[str, Any]: ...   # same dict shape as below

@dataclass(frozen=True)
class KeyedWalkForwardResult:
    keys: tuple[str, ...]
    origins: Array                     # (n_origins,)
    one_step_variance: Array           # (n_keys, n_origins) VARIANCE_UNITS
    cumulative_variance: Array         # (n_keys, n_origins) h-bar target, VARIANCE_UNITS
    realized_variance: Array           # (n_keys, n_origins) forward labels — EVALUATION ONLY
    realized_return: Array             # (n_keys, n_origins) — EVALUATION ONLY
    status: dict[str, tuple[str, ...]]
    horizon: int
    quantiles: Array | None = None     # (n_keys, n_origins, n_levels)
    units: str = VARIANCE_UNITS
    scope: str = PER_SECURITY_SCOPE
    source_label: str = "UNSPECIFIED"

def walk_forward_per_security(panel: KeyedReturnPanel, model: PerSecurityVol, *,
                              origins: Sequence[int], horizon: int = 1,
                              quantiles: Sequence[float] | None = None,
                              seed: int | None = None) -> KeyedWalkForwardResult: ...
```

### Per-key `forecast(...)` dict shape (the variance entry point)

| key | type | units | notes |
|---|---|---|---|
| `"variance"` | `Array (horizon,)` | `decimal_squared` | **THE variance entry point** — a variance (sigma²), never a volatility |
| `"variance_units"` | `str` | — | always `"decimal_squared"` |
| `"sigma"` | `Array (horizon,)` | `decimal` | documented √ convenience (`sqrt(variance)`) |
| `"sigma_units"` | `str` | — | always `"decimal"` |
| `"cumulative_variance"` | `Array (horizon,)` | `decimal_squared` | `cumsum(variance)`; the h-bar realized-variance target (zero-autocovariance approximation, same convention as `GARCHVol`) |
| `"mean"` | `float` | `decimal` | conditional mean |
| `"quantiles"` | `Array (horizon, n_levels)` | `decimal` | optional predictive RETURN quantiles |
| `"horizon"`, `"key"`, `"scope"`, `"series_scope"`, `"fit_status"` | — | — | `"fit_status"` is `"ok"` or `"failed:<reason>"` |

Rules for T6:
- **Use `forecast(...)["variance"]` (or `cumulative_variance`) only** as a
  variance input. The legacy `predict()` sigma adapter is never valid input to
  variance QLIKE.
- A failed key yields a dict with NaN `variance` and `"fit_status" =
  "failed:<reason>"` instead of raising — one bad key never poisons the rest.
  Unknown keys (never fitted) raise `ValueError` (caller error).

## 4. Units contract (`quant_fund.metrics.vol_eval`)

```python
VARIANCE_UNITS: Final[str] = "decimal_squared"   # a VARIANCE (sigma^2) of decimal log returns
VOLATILITY_UNITS: Final[str] = "decimal"         # a VOLATILITY (sigma); same numeric scale as returns
KNOWN_VOL_UNITS: Final[frozenset[str]]

class VolUnitsError(ValueError): ...

def assert_units(actual: object, expected: object) -> str: ...
def coerce_units(values: Array, *, from_units: object, to_units: object) -> NDArray[np.float64]: ...
def rescale_units(values: Array, *, from_units: object, to_units: object) -> NDArray[np.float64]: ...
```

- `coerce_units` returns the values **bit-identical** when units match
  (variance in == variance out, volatility in == volatility out) and raises
  `VolUnitsError` across units — there is NO silent rescale anywhere.
- `rescale_units` is the ONLY sanctioned conversion (explicit √ / square;
  same-units calls pass through unchanged).
- Both `GARCHVol.forecast` and `PerSecurityVol.forecast` dicts carry
  `"variance_units"` / `"sigma_units"` tags; `diagnostics()["variance_units"]`
  is `"decimal_squared"`.

## 5. Keyed proper scores (`quant_fund.metrics.vol_eval`)

```python
def qlike_keyed(realized_var: Array, forecast_var: Array, *, keys: Sequence[str],
                units: str = VARIANCE_UNITS) -> dict[str, Any]: ...
    # matrices (n_keys, n_obs); Patton QLIKE; REQUIRES VARIANCE_UNITS
    # (volatility-unit input raises VolUnitsError — never silently squared)
    # returns {"score": "qlike", "units", "keys", "qlike_per_key": {key: float},
    #          "qlike_mean": float, "n_scored_per_key": {key: int}, "n_scored": int}

def pinball_keyed(realized: Array, quantile_forecast: Array, taus: Sequence[float], *,
                  keys: Sequence[str], units: str = VOLATILITY_UNITS) -> dict[str, Any]: ...
    # realized (n_keys, n_obs); quantile_forecast (n_keys, n_obs, n_levels), n_levels == len(taus)
    # returns {"score": "pinball", "units", "keys", "taus",
    #          "pinball_per_key": {key: {tau: float}}, "pinball_mean_per_tau": {tau: float},
    #          "n_scored_per_key": {key: int}}
```

Honest-NaN rule (both): a key with fewer than 10 finite scored pairs returns
NaN for that key WITHOUT poisoning the other keys or the pooled mean. Both are
PROPER SCORES ONLY (Patton 2011 QLIKE; pinball). They never emit
Sharpe/Sortino/Calmar/P&L/NAV keys.

Scoring a walk-forward result:

```python
result = walk_forward_per_security(panel, model, origins=origins, horizon=h)
qlike = qlike_keyed(result.realized_variance, result.cumulative_variance, keys=result.keys)
pinball = pinball_keyed(result.realized_return, result.quantiles, taus, keys=result.keys)
```

## 6. PIT semantics (point-in-time discipline)

- Every fit at origin `t` uses ONLY returns with `time < t`
  (`KeyedReturnPanel.trailing`). Trailing/as-of windows only.
- `KeyedReturnPanel.labels` are forward realized-variance labels indexed at
  their forecast origin; they are **evaluation-only metadata** and no
  fitting/forecasting path reads them. Alignment: the forecast issued at
  origin `t` predicts the realized variance of the window STARTING at `t`
  (one-step: `r[t] ** 2`), i.e. `label_at(key, t)`.
- `cumulative_variance[-1]` (h-bar) pairs with an h-bar forward label.
- Proofs (checked in `tests/unit/models/test_vol_per_security.py`, asserted,
  not commented):
  1. shifting the forward labels in time leaves every forecast bit-identical;
  2. perturbing returns at `t >= t0` leaves every forecast issued before `t0`
     bit-identical.
  Each proof ships with a permanent MUTANT-harness test: a deliberately
  label-leaking (resp. as-of-ignoring) implementation is fed to the SAME proof
  helper, which must raise — proving the proof can fail on the bug it targets.

## 7. T6 wiring recipes (no redesign needed)

### `pipeline/train.py`

1. Build a `KeyedReturnPanel` per training window (keys = `security_id`,
   times = the integer/date-derived event-time axis, returns = `ret_1`
   decimals, labels = `future_realized_var_h` — labels used for SCORING only).
2. `model = PerSecurityVol(spec, min_obs=...)`; fit via walk-forward
   `walk_forward_per_security(panel, model, origins=..., horizon=h)`.
3. Select models ONLY by proper scores from §5 (`qlike_mean`, pinball) — never
   by P&L/Sharpe keys.
4. Persist artifacts with `model.metadata()`; stamp
   `series_scope="per_security"` in every report/manifest diagnostics blob
   (`model.diagnostics()` already carries it).

### `pipeline/forecast.py`

1. Load the artifact; at a per-security consumer boundary call
   `require_per_security(artifact)` (or
   `assert_scope_compatible(artifact.series_scope, PER_SECURITY_CONSUMER)`).
   At a pooled/date-level boundary call `require_pooled(artifact)`.
2. Read variance ONLY from `forecast(...)["variance"]` /
   `["cumulative_variance"]`; record `["variance_units"]` alongside. Never
   feed `predict()` sigma into variance QLIKE; never square or sqrt values in
   place — use `rescale_units` explicitly at the boundary if a consumer truly
   needs the other unit.
3. A rejected pair raises `VolScopeError` (a `ValueError`): let it fail the
   request closed — do not catch-and-default.

## 8. Test map (evidence that the contract holds)

| claim | test |
|---|---|
| full rejection matrix, both directions | `tests/unit/models/test_vol_scope.py` |
| missing / unknown / corrupted tokens | `tests/unit/models/test_vol_scope.py` |
| `GARCHVol` inherits the contract | `tests/unit/models/test_vol_scope.py` |
| QLIKE + pinball on seeded SYNTHETIC keyed panels | `tests/unit/metrics/test_vol_eval_keyed.py`, `tests/unit/models/test_vol_per_security.py` |
| units preservation (variance in == variance out, no silent rescale) | `tests/unit/models/test_garch_units.py`, `tests/unit/metrics/test_vol_eval_keyed.py`, `tests/unit/models/test_vol_per_security.py` |
| honest-NaN per key, no poisoning | `tests/unit/models/test_vol_per_security.py`, `tests/unit/metrics/test_vol_eval_keyed.py` |
| no-forward-label proof + mutant harness | `tests/unit/models/test_vol_per_security.py` |
| PIT trailing-window proof + mutant harness | `tests/unit/models/test_vol_per_security.py` |
| deterministic-seed reproducibility | `tests/unit/models/test_vol_per_security.py`, `tests/unit/metrics/test_vol_eval_keyed.py` |
