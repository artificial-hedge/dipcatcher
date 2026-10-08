"""Sundial flow-matching foundation head (P2.3). Research-only (SYNTHETIC).

THU-MT ``Sundial`` (``thuml/sundial-base-128m`` on HF) is a generative
foundation model — ``generate(seqs, max_new_tokens=pred_len,
num_samples=k)`` emits ``(B, k, pred_len)`` sampled paths, so the quantile
grid comes from the empirical distribution over samples: a real
probabilistic forecast, no parametric spread assumption.

Two reasons this stays a lazy, fail-closed dep rather than entering
uv.lock:

- Upstream requires ``transformers==4.40.1`` exactly (April 2024) —
  pinning it would drag the whole environment backward for one lane.
- ``AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)``
  *executes model-repo code* on import; per supply-chain policy that
  stays an explicit manual step, never an implicit install.

Fleet contract (mirrors ``models/distribution.py``): ``fit`` validates
the series and resolves the predictor (the dep check); ``predict`` tiles
the last-window quantile row; ``predict_from_history`` scores every
consecutive ``lookback`` window causally. Sample-to-quantile reduction
is empirical over ``num_samples`` paths per window.
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

_DEP_NAME = "thuml/Sundial"
_DEP_ERROR = (
    f"{_DEP_NAME} is not installed. It is an optional lane dep kept out of "
    "uv.lock: upstream pins transformers==4.40.1 exactly and requires "
    "trust_remote_code=True (model-repo code executes on import). Install "
    "manually in a separate venv for runs; the head fails closed rather "
    "than degrade silently."
)

_MODEL_ID = "thuml/sundial-base-128m"


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
    """Fail-closed lazy import of the transformers auto-loader."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(_DEP_ERROR) from exc
    return AutoModelForCausalLM


def _samples_to_quantiles(samples: Any, taus: Array) -> Array:
    """Empirical tau quantiles over Sundial's (B, num_samples, pred_len) draws."""
    arr = np.asarray(samples, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        # (num_samples, pred_len) — take step-0 closes.
        flat = arr[:, 0]
    elif arr.ndim == 1:
        flat = arr
    else:
        raise ValueError(
            f"sundial sample tensor has unexpected shape {arr.shape}; "
            "expected (samples, pred_len) or (samples,)"
        )
    if flat.size < 2 or not np.isfinite(flat).all():
        raise ValueError("sundial emitted non-finite or singleton samples")
    result: Array = np.quantile(flat, taus).astype(np.float64)
    return result


class SundialDistribution(JoblibMixin):
    """Zero-shot Sundial quantile head on the trailing return series.

    Pretrained weights are a fixed black box — ``fit`` validates the
    series, records the trailing standardized window, and resolves the
    model (the dep check). Sampling happens per-window inside
    ``predict``/``predict_from_history``; nothing crosses calls.
    """

    name = "sundial"

    def __init__(
        self,
        taus: Sequence[float] = (0.1, 0.5, 0.9),
        *,
        seed: int = 0,
        lookback: int = 288,
        num_samples: int = 32,
        model_id: str = _MODEL_ID,
        predictor: Any = None,
    ) -> None:
        self.taus = _as_tau_grid(taus)
        self.seed = int(seed)
        if lookback < _MIN_LOOKBACK:
            raise ValueError(f"lookback must be >= {_MIN_LOOKBACK}; got {lookback}")
        if num_samples < 4:
            raise ValueError("num_samples must be >= 4")
        self.lookback = int(lookback)
        self.num_samples = int(num_samples)
        self.model_id = model_id
        self._predictor = predictor
        self._window: Array | None = None
        self._mu = 0.0
        self._sig = 1.0

    def _ensure_predictor(self) -> Any:
        if self._predictor is None:
            cls = _load_model_cls()
            self._predictor = cls.from_pretrained(self.model_id, trust_remote_code=True)
        return self._predictor

    def _window_quantiles(self, window_z: Array) -> Array:
        model = self._ensure_predictor()
        raw = model.generate(
            np.asarray(window_z, dtype=np.float64).reshape(1, -1),
            max_new_tokens=1,
            num_samples=self.num_samples,
        )
        q = _samples_to_quantiles(raw, self.taus)
        return np.sort(q * self._sig + self._mu)

    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
    ) -> SundialDistribution:
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

        Each window is standardized by the fitted scaler and sampled
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
                "framework": "thuml/Sundial",
                "model_id": self.model_id,
                "device": "cpu",
                "pretrained": True,
                "weights_note": (
                    "HF weights thuml/sundial-base-128m; dep is "
                    "uv.lock-incompatible (transformers==4.40.1 pin + "
                    "trust_remote_code executes repo code) — fails closed "
                    "until upstream loosens"
                ),
                "lookback": self.lookback,
                "num_samples": self.num_samples,
                "standardize": True,
            },
        )
