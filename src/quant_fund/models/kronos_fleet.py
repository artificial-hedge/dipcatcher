"""Kronos fleet-protocol head (P2.6). Research-only (SYNTHETIC).

Wraps the existing :class:`quant_fund.models.kronos.KronosAdapter` /
``KronosPredictor`` protocol behind the fleet's univariate contract.
Two resolved dependencies, both fail-closed:

- The predictor is injected (tests/research) or loaded via
  ``load_local_predictor`` — which requires local artifact dirs and
  refuses the network. The upstream code itself is GitHub-only
  (``github.com/NeoQuasarAI/Kronos``, no PyPI dist), so without a local
  clone + HF snapshot the head raises a named error at fit time.
- The fleet protocol carries a univariate return series, but Kronos was
  trained on OHLCV candles. This wrapper synthesizes *range-degenerate*
  candles from the series (``open = prev close``, ``high/low`` bound the
  endpoints, ``volume = amount = 0``) and stamps
  ``input_mode=returns_synthesized_degenerate_candles`` +
  ``degenerate_input=True`` in metadata — a reader can never mistake the
  construction for native OHLCV input. Timestamps are synthetic daily
  bars (the fleet protocol carries no clock); disclosed the same way.

Quantile rows are empirical cross-sample quantiles: ``predict`` is
called ``n_samples`` times per window and the tau grid is taken over the
sampled next-bar close-to-close returns — real model samples, not a
fitted parametric spread.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from quant_fund.models.base import JoblibMixin, ModelMeta
from quant_fund.models.kronos import OHLCV_COLUMNS, KronosPredictor, validate_ohlcv_frame

Array = NDArray[np.float64]

_MIN_LOOKBACK = 16
_BASE_STAMP = datetime(2000, 1, 1, tzinfo=UTC)


def _as_tau_grid(taus: Sequence[float]) -> Array:
    t = np.asarray(list(taus), dtype=np.float64)
    if (
        t.size == 0
        or not np.isfinite(t).all()
        or np.any((t <= 0.0) | (t >= 1.0))
        or np.any(np.diff(t) <= 0.0)
    ):
        raise ValueError("taus must be a nonempty strictly increasing grid inside (0, 1)")
    return t


def _returns_to_candle_frame(log_returns: Array) -> pd.DataFrame:
    """Range-degenerate OHLCV frame from a return series (disclosed mode)."""
    rets = np.asarray(log_returns, dtype=np.float64).reshape(-1)
    closes = np.exp(np.cumsum(rets))
    prev = np.concatenate(([1.0], closes[:-1]))
    frame = pd.DataFrame(
        {
            "open": prev,
            "high": np.maximum(prev, closes),
            "low": np.minimum(prev, closes),
            "close": closes,
            "volume": np.zeros(rets.size),
            "amount": np.zeros(rets.size),
        }
    )
    return validate_ohlcv_frame(frame)


def _synthetic_stamps(n_history: int, pred_len: int) -> tuple[pd.Series, pd.Series]:
    """Synthetic daily stamps — the fleet protocol carries no wall clock."""
    x_ts = pd.Series([_BASE_STAMP + timedelta(days=i) for i in range(n_history)])
    y_ts = pd.Series([_BASE_STAMP + timedelta(days=n_history + i) for i in range(pred_len)])
    return x_ts, y_ts


class KronosFleetDistribution(JoblibMixin):
    """Fleet head: Kronos candle model over synthesized return candles.

    ``predictor`` injects any object satisfying ``KronosPredictor``
    (deterministic stubs in tests; ``load_local_predictor`` output in
    research). With ``predictor=None`` the head resolves local artifacts
    lazily inside ``fit`` and fails closed with a named error.
    """

    name = "kronos_base"

    def __init__(
        self,
        taus: Sequence[float] = (0.1, 0.5, 0.9),
        *,
        seed: int = 0,
        lookback: int = 256,
        n_samples: int = 32,
        predictor: KronosPredictor | None = None,
        model_path: str | None = None,
        tokenizer_path: str | None = None,
    ) -> None:
        self.taus = _as_tau_grid(taus)
        self.seed = int(seed)
        if lookback < _MIN_LOOKBACK:
            raise ValueError(f"lookback must be >= {_MIN_LOOKBACK}; got {lookback}")
        if n_samples < 4:
            raise ValueError("n_samples must be >= 4")
        self.lookback = int(lookback)
        self.n_samples = int(n_samples)
        self._predictor = predictor
        self._model_path = model_path
        self._tokenizer_path = tokenizer_path
        self._window: Array | None = None

    def _ensure_predictor(self) -> KronosPredictor:
        if self._predictor is None:
            if self._model_path is None or self._tokenizer_path is None:
                raise RuntimeError(
                    "kronos_base requires an injected predictor or "
                    "model_path/tokenizer_path pointing at local artifacts; "
                    "upstream has no PyPI dist (NeoQuasarAI/Kronos on GitHub) "
                    "and load_local_predictor refuses the network"
                )
            from quant_fund.models.kronos import load_local_predictor

            self._predictor = load_local_predictor(
                model_path=Path(self._model_path),
                tokenizer_path=Path(self._tokenizer_path),
                max_context=self.lookback,
            )
        return self._predictor

    def _window_quantiles(self, window_returns: Array) -> Array:
        """Empirical quantiles over n_samples independent sampled paths."""
        predictor = self._ensure_predictor()
        candles = _returns_to_candle_frame(window_returns)
        x_ts, y_ts = _synthetic_stamps(candles.shape[0], 1)
        samples = np.empty(self.n_samples, dtype=np.float64)
        last_close = float(candles["close"].iloc[-1])
        for s in range(self.n_samples):
            out = predictor.predict(
                candles[list(OHLCV_COLUMNS)],
                x_ts,
                y_ts,
                1,
                T=1.0,
                top_k=0,
                top_p=0.9,
                sample_count=1,
                verbose=False,
            )
            if not isinstance(out, pd.DataFrame) or list(out.columns) != list(OHLCV_COLUMNS):
                raise ValueError("kronos predictor must return a six-column OHLCV frame")
            close = float(out["close"].iloc[0])
            if not np.isfinite(close) or close <= 0.0:
                raise ValueError("kronos emitted a non-finite or non-positive close")
            samples[s] = np.log(close / last_close)
        return np.quantile(samples, self.taus).astype(np.float64)

    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
    ) -> KronosFleetDistribution:
        yy = np.asarray(y, dtype=np.float64).reshape(-1)
        if not np.isfinite(yy).all():
            raise ValueError(f"{self.name} requires an all-finite series")
        if yy.size < self.lookback + 1:
            raise ValueError(
                f"{self.name} requires >= {self.lookback + 1} observations; got {yy.size}"
            )
        # Resolving the predictor is the dep check: fails closed so a
        # half-fit object can never predict.
        self._ensure_predictor()
        self._window = yy[-self.lookback :].astype(np.float64)
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._window is None:
            raise RuntimeError("distribution model has not been fitted")
        arr = np.asarray(x)
        if arr.ndim == 0 or arr.shape[0] < 1:
            raise ValueError("predict requires at least one row")
        q = np.sort(self._window_quantiles(self._window))
        return np.tile(q, (int(arr.shape[0]), 1))

    def predict_from_history(self, history: NDArray[np.float64]) -> NDArray[np.float64]:
        """Per-window quantiles over the univariate series (causal windows)."""
        if self._window is None:
            raise RuntimeError("distribution model has not been fitted")
        hh = np.asarray(history, dtype=np.float64).reshape(-1)
        if hh.size < self.lookback or not np.isfinite(hh).all():
            raise ValueError(
                f"{self.name} predict_from_history requires >= lookback finite observations"
            )
        out = np.empty((hh.size - self.lookback + 1, self.taus.size), dtype=np.float64)
        for i in range(0, hh.size - self.lookback + 1):
            out[i] = np.sort(self._window_quantiles(hh[i : i + self.lookback]))
        return out

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="distribution",
            name=self.name,
            version="v1",
            seed=self.seed,
            extra={
                "framework": "NeoQuasarAI/Kronos",
                "device": "cpu",
                "pretrained": True,
                "weights_note": (
                    "HF weights via load_local_predictor (local artifacts "
                    "only, sha256-pinnable); code is GitHub-only — no PyPI "
                    "dist, cannot enter uv.lock"
                ),
                "lookback": self.lookback,
                "n_samples": self.n_samples,
                "input_mode": "returns_synthesized_degenerate_candles",
                "timestamps": "synthetic_daily",
                "degenerate_input": True,
            },
        )
