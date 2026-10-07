"""Toto-2 foundation-model quantile head (P2.4). Research-only (SYNTHETIC).

Datadog ``Toto-2.0`` is a u-μP-scaled transformer family (4M–2.5B params)
with *native quantile-based* probabilistic forecasting — the cleanest fit
to the fleet contract of the zero-shot SOTA heads. The dependency is a
lazy import resolved inside ``fit``: ``toto-2`` requires
``gluonts[torch]>=0.16``, whose ``toolz~=0.10`` pin conflicts with this
repo's pinned ``exchange-calendars`` (``toolz>=1``); the v1 ``toto-ts``
package is worse (hard ``==`` pins incl. jupyter). Neither can enter
uv.lock, so the adapter fails closed with a named error until upstream
loosens the constraint.

Fleet contract (mirrors ``models/distribution.py``): ``fit`` validates
the series and resolves the predictor; ``predict`` tiles the last-window
quantile row; ``predict_from_history`` scores every consecutive
``lookback`` window causally — each window sees only its own history.
Quantile extraction accepts an injected predictor returning either a
``(n_taus,)`` vector already on the requested grid or an object with a
``quantiles``/``samples`` field; samples are reduced to empirical
quantiles. Anything else fails closed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.base import JoblibMixin, ModelMeta

Array = NDArray[np.float64]

_MIN_LOOKBACK = 8
_NN_MIN_WINDOWS = 4

_DEP_NAME = "toto-2"
_DEP_ERROR = (
    f"{_DEP_NAME} is not installed. It is an optional lane dep kept out of "
    "uv.lock: it requires gluonts[torch]>=0.16, whose toolz~=0.10 pin is "
    "unsatisfiable against pinned exchange-calendars (toolz>=1); toto-ts v1 "
    "hard-pins its whole stack. Install in a separate venv for manual runs; "
    "the head fails closed rather than degrade silently."
)

_MODEL_ID = "Datadog/Toto-2.0-22m"


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


def _load_model_cls() -> Any:
    """Fail-closed lazy import of the toto-2 model class."""
    try:
        from toto2 import Toto2Model
    except ImportError as exc:
        raise RuntimeError(_DEP_ERROR) from exc
    return Toto2Model


def _prediction_quantiles(prediction: Any, taus: Array) -> Array:
    """Reduce a predictor output to the requested quantile grid.

    Accepts: a ``len(taus)`` vector already on the grid; an object with a
    ``quantiles`` row keyed to the grid; or a ``samples``/plain ndarray
    with a sample axis — reduced to empirical quantiles. Anything else
    fails closed.
    """
    # A 1-D vector whose length matches the grid is taken as the grid row —
    # a wrong-length 1-D vector is NOT silently reinterpreted as samples
    # (a 2-element vector and a 2-sample axis are indistinguishable).
    if isinstance(prediction, np.ndarray) or not (
        hasattr(prediction, "quantiles") or hasattr(prediction, "samples")
    ):
        arr = np.asarray(prediction, dtype=np.float64)
        arr = np.squeeze(arr)
        if arr.ndim == 1 and arr.shape[0] == taus.shape[0]:
            if not np.isfinite(arr).all():
                raise ValueError("toto2 emitted non-finite quantiles")
            return arr
        raise ValueError(
            f"toto2 prediction shape {arr.shape} cannot be reduced to "
            f"{taus.shape[0]} quantiles — expected a grid row or a "
            ".samples/.quantiles-carrying object"
        )
    quantiles = (
        np.asarray(prediction.quantiles, dtype=np.float64).squeeze()
        if hasattr(prediction, "quantiles")
        else None
    )
    if quantiles is not None and quantiles.ndim == 1 and quantiles.shape[0] == taus.shape[0]:
        if not np.isfinite(quantiles).all():
            raise ValueError("toto2 emitted non-finite quantiles")
        return quantiles
    samples_attr = getattr(prediction, "samples", None)
    if samples_attr is not None:
        samples = np.asarray(samples_attr, dtype=np.float64).squeeze()
        if samples.ndim == 1 and samples.shape[0] > 1:
            if not np.isfinite(samples).all():
                raise ValueError("toto2 emitted non-finite samples")
            return np.quantile(samples, taus).astype(np.float64)
        if samples.ndim == 2:
            flat = samples.reshape(-1)
            if not np.isfinite(flat).all():
                raise ValueError("toto2 emitted non-finite samples")
            return np.quantile(flat, taus).astype(np.float64)
    raise ValueError(
        f"toto2 prediction (quantiles={quantiles if quantiles is not None else 'absent'}, "
        f"samples shape={getattr(samples_attr, 'shape', None)}) cannot be reduced to "
        f"{taus.shape[0]} quantiles"
    )


class Toto2Distribution(JoblibMixin):
    """Zero-shot Toto-2 quantile head on the trailing return series.

    The pretrained model is a fixed black box — ``fit`` only validates the
    series, records the trailing window, and resolves the model (the dep
    check). All model calls run per-window inside ``predict`` /
    ``predict_from_history``; no state crosses calls except the fitted
    window.
    """

    name = "toto2"

    def __init__(
        self,
        taus: Sequence[float] = (0.1, 0.5, 0.9),
        *,
        seed: int = 0,
        lookback: int = 256,
        model_id: str = _MODEL_ID,
        predictor: Any = None,
    ) -> None:
        self.taus = _as_tau_grid(taus)
        self.seed = int(seed)
        if lookback < _MIN_LOOKBACK:
            raise ValueError(f"lookback must be >= {_MIN_LOOKBACK}; got {lookback}")
        self.lookback = int(lookback)
        self.model_id = model_id
        self._predictor = predictor
        self._window: Array | None = None
        self._mu = 0.0
        self._sig = 1.0

    def _ensure_predictor(self) -> Any:
        if self._predictor is None:
            cls = _load_model_cls()
            self._predictor = cls.from_pretrained(self.model_id)
        return self._predictor

    def _window_quantiles(self, window_z: Array) -> Array:
        model = self._ensure_predictor()
        raw = model.predict(np.asarray(window_z, dtype=np.float64).reshape(-1))
        q = _prediction_quantiles(raw, self.taus)
        return np.sort(q * self._sig + self._mu)

    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
    ) -> Toto2Distribution:
        yy = np.asarray(y, dtype=np.float64).reshape(-1)
        if not np.isfinite(yy).all():
            raise ValueError(f"{self.name} requires an all-finite series")
        if yy.size < self.lookback + _NN_MIN_WINDOWS:
            raise ValueError(
                f"{self.name} requires >= {self.lookback + _NN_MIN_WINDOWS} observations "
                f"(lookback {self.lookback} + {_NN_MIN_WINDOWS} train windows); got {yy.size}"
            )
        # Resolving the model is the dep check: fails closed so a half-fit
        # object can never predict.
        self._ensure_predictor()
        self._mu = float(yy.mean())
        s = float(yy.std(ddof=1))
        self._sig = s if np.isfinite(s) and s > 0.0 else 1.0
        self._window = ((yy[-self.lookback :] - self._mu) / self._sig).astype(np.float64)
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Tile the last-window one-step quantiles across ``x`` rows."""
        if self._window is None:
            raise RuntimeError("distribution model has not been fitted")
        arr = np.asarray(x)
        if arr.ndim == 0 or arr.shape[0] < 1:
            raise ValueError("predict requires at least one row")
        q = self._window_quantiles(self._window)
        return np.tile(q, (int(arr.shape[0]), 1))

    def predict_from_history(self, history: NDArray[np.float64]) -> NDArray[np.float64]:
        """Quantile row for every consecutive ``lookback`` window in ``history``.

        Each window is standardized by the fitted scaler and scored
        independently — no leakage beyond the window itself. Returns
        ``(len(history) - lookback + 1, n_taus)``.
        """
        if self._window is None:
            raise RuntimeError("distribution model has not been fitted")
        hh = np.asarray(history, dtype=np.float64).reshape(-1)
        if hh.size < self.lookback or not np.isfinite(hh).all():
            raise ValueError(
                f"{self.name} predict_from_history requires >= lookback finite observations"
            )
        z = (hh - self._mu) / self._sig
        out = np.empty((hh.size - self.lookback + 1, self.taus.size), dtype=np.float64)
        for i in range(0, hh.size - self.lookback + 1):
            out[i] = np.sort(self._window_quantiles(z[i : i + self.lookback]))
        return out

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="distribution",
            name=self.name,
            version="v1",
            seed=self.seed,
            extra={
                "framework": "toto-2",
                "model_id": self.model_id,
                "device": "cpu",
                "pretrained": True,
                "weights_note": (
                    "HF weights Datadog/Toto-2.0-*; dep is uv.lock-incompatible "
                    "(gluonts toolz~=0.10 vs exchange-calendars toolz>=1) — "
                    "fails closed until upstream loosens"
                ),
                "lookback": self.lookback,
                "standardize": True,
            },
        )
