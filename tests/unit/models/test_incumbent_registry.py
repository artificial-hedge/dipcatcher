"""Per-incumbent output-contract tests for ``models.incumbent_registry``.

SYNTHETIC: every predictor here is a stub whose output *encodes the quantile
level it is meant to carry*. These prove adapter correctness (levels are
served by the right channel, malformed output fails closed, windows are
causal) and registry integrity — they are never market evidence and say
nothing about any pretrained model's forecasting skill.

Why level-encoding stubs: the TimesFM channel-5 incident (see
``scripts/splice_timesfm_fix.py``) was an adapter taking quantile columns by
*position* from a labelled channel layout, silently scoring the point forecast
as a quantile and dropping q90. A stub that returns a flat or window-mean fan
cannot catch that; a stub whose column ``j`` literally equals the level
``j`` is supposed to carry can.

The series fed to every head is constant zero. Every standardizing head
falls back to ``(mu, sigma) = (0, 1)`` on a zero-variance series, so the
scaler is the identity and a head's output must equal the encoded levels
exactly.
"""

from __future__ import annotations

import re
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quant_fund.models.incumbent_registry import (
    INCUMBENTS,
    TIMESFM_2P5_CHANNELS,
    IncumbentSpec,
    get_incumbent,
    incumbent_keys,
    labelled_channel_indices,
    load_head_class,
)
from quant_fund.research.fleet_eval import FLEET_HEAD_REGISTRY

_REPO = Path(__file__).resolve().parents[3]
_LOCK_NAMES = frozenset(re.findall(r'^name = "([^"]+)"$', (_REPO / "uv.lock").read_text(), re.M))

_LOOKBACK = 16
_N = _LOOKBACK + 8
_SAMPLES = np.linspace(0.0, 1.0, 101)  # np.quantile(_SAMPLES, t) == t exactly
_NATIVE = np.linspace(0.1, 0.9, 9)  # tirex2 native grid
_TAUS_CONTIGUOUS = (0.1, 0.5, 0.9)
_TAUS_SPARSE = (0.2, 0.5, 0.8)  # non-contiguous subset of the native grid

Builder = Callable[[tuple[float, ...], pytest.MonkeyPatch], Any]


def _zeros() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((_N, 2)), np.zeros(_N)


# --------------------------------------------------------------------------
# Level-encoding stubs, one builder per wired incumbent.
# --------------------------------------------------------------------------


def _build_sundial(taus: tuple[float, ...], mp: pytest.MonkeyPatch) -> Any:
    from quant_fund.models.sundial import SundialDistribution

    class _Pred:
        def generate(self, seqs: Any, *, max_new_tokens: int, num_samples: int) -> np.ndarray:
            return _SAMPLES.reshape(1, -1, 1)

    x, y = _zeros()
    return SundialDistribution(taus, lookback=_LOOKBACK, predictor=_Pred()).fit(x, y)


def _build_toto2(taus: tuple[float, ...], mp: pytest.MonkeyPatch) -> Any:
    from quant_fund.models.toto2 import Toto2Distribution

    class _Out:
        samples = _SAMPLES

    class _Pred:
        def predict(self, window: Any) -> Any:
            return _Out()

    x, y = _zeros()
    return Toto2Distribution(taus, lookback=_LOOKBACK, predictor=_Pred()).fit(x, y)


def _build_moirai2(taus: tuple[float, ...], mp: pytest.MonkeyPatch) -> Any:
    from quant_fund.models.moirai2 import Moirai2Distribution

    class _Forecast:
        samples = _SAMPLES

    class _Predictor:
        def predict(self, dataset: Any) -> Any:
            yield _Forecast()

    class _Module:
        @classmethod
        def from_pretrained(cls, weights_id: str) -> _Module:
            return cls()

    class _MoiraiForecast:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def create_predictor(self, batch_size: int = 32) -> _Predictor:
            return _Predictor()

    pkg, sub, mod = (types.ModuleType(n) for n in ("uni2ts", "uni2ts.model", "uni2ts.model.moirai"))
    mod.MoiraiForecast = _MoiraiForecast  # type: ignore[attr-defined]
    mod.MoiraiModule = _Module  # type: ignore[attr-defined]
    for name, m in (("uni2ts", pkg), ("uni2ts.model", sub), ("uni2ts.model.moirai", mod)):
        mp.setitem(sys.modules, name, m)
    x, y = _zeros()
    return Moirai2Distribution(taus, lookback=_LOOKBACK).fit(x, y)


def _build_tabpfn_ts(taus: tuple[float, ...], mp: pytest.MonkeyPatch) -> Any:
    from quant_fund.models.tabpfn_ts import TabpfnTsDistribution

    levels = np.asarray(taus, dtype=np.float64)

    class _Predictor:
        def __init__(self, tabpfn_config: Any = None) -> None:
            pass

        def predict(self, window: np.ndarray) -> np.ndarray:
            return levels.copy()

    mod = types.ModuleType("tabpfn_time_series")
    mod.TabPFNTimeSeriesPredictor = _Predictor  # type: ignore[attr-defined]
    mp.setitem(sys.modules, "tabpfn_time_series", mod)
    x, y = _zeros()
    return TabpfnTsDistribution(taus, lookback=_LOOKBACK).fit(x, y)


def _build_tirex2(taus: tuple[float, ...], mp: pytest.MonkeyPatch) -> Any:
    pytest.importorskip("torch")
    from quant_fund.models.tirex2 import Tirex2Distribution

    class _Ts:
        def __init__(self, target: Any, past_covariates: Any, future_covariates: Any) -> None:
            self.target = target

    class _Model:
        quantiles = _NATIVE.copy()

        def forecast(self, timeseries: list[Any], prediction_length: int, **kw: Any) -> list[Any]:
            # Column j carries native level j: value == the level it labels.
            return [_NATIVE.reshape(1, -1, 1).copy() for _ in timeseries]

    mod = types.ModuleType("tirex2")
    mod.load_model = lambda *a, **k: _Model()  # type: ignore[attr-defined]
    mod.TimeseriesType = _Ts  # type: ignore[attr-defined]
    mp.setitem(sys.modules, "tirex2", mod)
    x, y = _zeros()
    return Tirex2Distribution(taus, lookback=_LOOKBACK).fit(x, y)


_BUILDERS: dict[str, Builder] = {
    "sundial": _build_sundial,
    "toto2": _build_toto2,
    "moirai2": _build_moirai2,
    "tabpfn_ts": _build_tabpfn_ts,
    "tirex2": _build_tirex2,
}

_WIRED = [s.key for s in INCUMBENTS if s.wired]


# --------------------------------------------------------------------------
# Registry integrity — the table cannot drift from code or the lockfile.
# --------------------------------------------------------------------------


def test_keys_unique() -> None:
    keys = [s.key for s in INCUMBENTS]
    assert len(keys) == len(set(keys))


def test_every_wired_incumbent_has_a_contract_builder() -> None:
    assert sorted(_WIRED) == sorted(_BUILDERS)


@pytest.mark.parametrize("spec", INCUMBENTS, ids=lambda s: s.key)
def test_spec_internal_consistency(spec: IncumbentSpec) -> None:
    assert spec.reason.strip()
    if spec.wired:
        assert spec.module and spec.cls and spec.output_contract
    else:
        assert spec.module is None and spec.cls is None
    if spec.status == "not_implemented":
        assert spec.output_contract is None


@pytest.mark.parametrize("key", _WIRED)
def test_wired_class_name_and_fleet_registration(key: str) -> None:
    cls = load_head_class(key)
    assert cls.name == key
    assert key in FLEET_HEAD_REGISTRY


def test_fleet_foundation_heads_are_all_registered() -> None:
    fm_heads = {"sundial", "toto2", "tirex2", "tabpfn_ts", "moirai2"}
    assert fm_heads <= set(FLEET_HEAD_REGISTRY)
    assert fm_heads <= set(incumbent_keys(wired_only=True))


@pytest.mark.parametrize("spec", [s for s in INCUMBENTS if not s.wired], ids=lambda s: s.key)
def test_unwired_incumbents_are_not_fleet_heads(spec: IncumbentSpec) -> None:
    # A not_implemented / script_only entry must never be scored as a fleet head.
    assert spec.key not in FLEET_HEAD_REGISTRY
    with pytest.raises(RuntimeError, match="no adapter"):
        load_head_class(spec.key)


def test_unknown_incumbent_raises() -> None:
    with pytest.raises(KeyError, match="unknown incumbent"):
        get_incumbent("chronos_nope")


@pytest.mark.parametrize("spec", INCUMBENTS, ids=lambda s: s.key)
def test_lock_status_matches_uv_lock(spec: IncumbentSpec) -> None:
    assert spec.lock_dist
    in_lock = spec.lock_dist in _LOCK_NAMES
    assert in_lock == (spec.lock_status == "in_uv_lock"), (
        f"{spec.key}: registry says {spec.lock_status} but {spec.lock_dist!r} "
        f"{'is' if in_lock else 'is not'} in uv.lock"
    )


# --------------------------------------------------------------------------
# TimesFM labelled-channel contract (the incident itself).
# --------------------------------------------------------------------------


def test_timesfm_layout_matches_documented_channels() -> None:
    assert len(TIMESFM_2P5_CHANNELS) == 10
    assert TIMESFM_2P5_CHANNELS[5] is None  # channel 5 is the POINT forecast
    quantile_levels = [lv for lv in TIMESFM_2P5_CHANNELS if lv is not None]
    assert sorted(quantile_levels) == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def test_timesfm_levels_taken_by_label_not_position() -> None:
    idx = labelled_channel_indices(_TAUS_CONTIGUOUS, TIMESFM_2P5_CHANNELS)
    assert idx == (1, 0, 9)  # q10 -> ch1, q50 -> ch0 (NOT 5), q90 -> ch9
    # Encode each channel with the level it carries; the point channel gets a
    # sentinel no quantile level can equal.
    encoded = np.array([99.0 if lv is None else lv for lv in TIMESFM_2P5_CHANNELS])
    assert encoded[list(idx)].tolist() == pytest.approx(list(_TAUS_CONTIGUOUS))


def test_timesfm_legacy_positional_slice_is_wrong() -> None:
    # Guards against a vacuous test: the legacy ``q[:9]`` read demonstrably
    # does NOT deliver the nine quantile levels (it ships the point forecast
    # as a pseudo-quantile and drops q90).
    encoded = np.array([99.0 if lv is None else lv for lv in TIMESFM_2P5_CHANNELS])
    legacy = encoded[:9]
    assert 99.0 in legacy
    assert 0.9 not in legacy
    by_label = encoded[list(labelled_channel_indices(np.arange(1, 10) / 10, TIMESFM_2P5_CHANNELS))]
    assert by_label.tolist() == pytest.approx((np.arange(1, 10) / 10).tolist())


def test_timesfm_level_not_carried_fails_closed() -> None:
    for bad in (0.05, 0.95, 0.55):
        with pytest.raises(ValueError, match="not carried by exactly one"):
            labelled_channel_indices((bad,), TIMESFM_2P5_CHANNELS)


def test_labelled_channels_duplicate_label_fails_closed() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        labelled_channel_indices((0.5,), (0.5, 0.5, 0.9))


# --------------------------------------------------------------------------
# Per-incumbent output-contract tests (stubbed, SYNTHETIC).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("taus", [_TAUS_CONTIGUOUS, _TAUS_SPARSE], ids=["contiguous", "sparse"])
@pytest.mark.parametrize("key", _WIRED)
def test_head_serves_each_requested_level(
    key: str, taus: tuple[float, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    head = _BUILDERS[key](taus, monkeypatch)
    hist = np.zeros(_N)
    out = head.predict_from_history(hist)
    assert out.shape == (_N - _LOOKBACK + 1, len(taus))
    np.testing.assert_allclose(out, np.tile(taus, (out.shape[0], 1)), atol=1e-12)
    # predict() tiles the same level-faithful row.
    tiled = head.predict(np.zeros((3, 2)))
    np.testing.assert_allclose(tiled, np.tile(taus, (3, 1)), atol=1e-12)


@pytest.mark.parametrize("key", _WIRED)
def test_head_rows_are_monotone_in_level(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    out = _BUILDERS[key](_TAUS_CONTIGUOUS, monkeypatch).predict_from_history(np.zeros(_N))
    assert np.all(np.diff(out, axis=1) >= 0.0)


def test_tirex2_non_native_level_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    # 0.15 sits between native columns 0.1 and 0.2: never snapped to either.
    with pytest.raises(ValueError, match="cannot honor tau"):
        _build_tirex2((0.15, 0.5, 0.9), monkeypatch)


def test_tabpfn_ts_wrong_width_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    head = _build_tabpfn_ts(_TAUS_CONTIGUOUS, monkeypatch)

    class _Narrow:
        def predict(self, window: np.ndarray) -> np.ndarray:
            return np.array([0.0, 0.0])

    head._predictor = _Narrow()
    with pytest.raises(ValueError, match="quantiles for a"):
        head.predict(np.zeros((1, 2)))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_toto2_non_finite_output_fails_closed(bad: float) -> None:
    from quant_fund.models.toto2 import Toto2Distribution

    class _Out:
        samples = np.array([0.0, bad, 1.0])

    class _Pred:
        def predict(self, window: Any) -> Any:
            return _Out()

    x, y = _zeros()
    head = Toto2Distribution(_TAUS_CONTIGUOUS, lookback=_LOOKBACK, predictor=_Pred()).fit(x, y)
    with pytest.raises(ValueError, match="non-finite"):
        head.predict(np.zeros((1, 2)))
