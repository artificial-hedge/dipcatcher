"""Static, audited metadata for features exposed through fx-1 extensions.

The underlying implementations remain in the dipcatcher harness. Keeping this
small catalog inside fx-1 lets the model discover named feature modules while
preserving the repository's one-way fx1-to-harness architecture boundary.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureDefinition:
    """One named output of the registered ``build-features`` harness surface."""

    name: str
    family: str
    lookback: int
    point_in_time_safe: bool = True
    synthetic_only: bool = False


def _feature(name: str, lookback: int, family: str) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        family=family,
        lookback=lookback,
        synthetic_only=family == "synthetic_oracle",
    )


FEATURE_DEFINITIONS: tuple[FeatureDefinition, ...] = (
    _feature("ret_1", 1, "returns"),
    _feature("ret_5", 5, "returns"),
    _feature("ret_20", 20, "returns"),
    _feature("log_ret_1", 1, "returns"),
    _feature("ret_overnight", 1, "returns"),
    _feature("ret_open_close", 1, "returns"),
    _feature("ret_overnight_20", 20, "momentum"),
    _feature("ret_intraday_20", 20, "momentum"),
    _feature("mom_5", 5, "momentum"),
    _feature("mom_20", 20, "momentum"),
    _feature("mom_skip_5_20", 20, "momentum"),
    _feature("mom_60", 60, "momentum"),
    _feature("mom_126", 126, "momentum"),
    _feature("mom_252", 252, "momentum"),
    _feature("mom_12_1", 252, "momentum"),
    _feature("high_52w_prox", 252, "momentum"),
    _feature("reversal_1", 1, "reversal"),
    _feature("z_vs_ma20", 20, "reversal"),
    _feature("max_ret_20", 20, "lottery"),
    _feature("min_ret_20", 20, "lottery"),
    _feature("skew_20", 20, "moments"),
    _feature("kurt_20", 20, "moments"),
    _feature("vol_20", 20, "volatility"),
    _feature("vol_60", 60, "volatility"),
    _feature("vol_ewma", 20, "volatility"),
    _feature("vol_parkinson", 20, "volatility"),
    _feature("vol_garman_klass", 20, "volatility"),
    _feature("vol_of_vol", 40, "volatility"),
    _feature("downside_vol_20", 20, "volatility"),
    _feature("vol_ratio_20_60", 60, "volatility"),
    _feature("beta_60", 60, "market"),
    _feature("idio_vol_60", 60, "market"),
    _feature("idio_mom_20", 20, "momentum"),
    _feature("amihud", 20, "liquidity"),
    _feature("amihud_60", 60, "liquidity"),
    _feature("adv", 20, "liquidity"),
    _feature("dollar_volume", 1, "liquidity"),
    _feature("rel_volume", 20, "liquidity"),
    _feature("turnover_proxy", 20, "liquidity"),
    _feature("volume_vol", 20, "liquidity"),
    _feature("adv_ratio_20_60", 60, "liquidity"),
    _feature("log_price", 1, "price"),
    _feature("cs_z_ret_1", 1, "cross_sectional"),
    _feature("planted_signal", 0, "synthetic_oracle"),
    _feature("cs_z_planted_signal", 0, "synthetic_oracle"),
)

_BY_NAME = {feature.name: feature for feature in FEATURE_DEFINITIONS}


def list_feature_definitions() -> tuple[FeatureDefinition, ...]:
    """List every reviewed feature exposed as an fx-1 extension."""
    return FEATURE_DEFINITIONS


def get_feature_definition(name: str) -> FeatureDefinition:
    """Return one reviewed feature definition, failing closed for unknown names."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown feature extension {name!r}; known: {sorted(_BY_NAME)}") from None
