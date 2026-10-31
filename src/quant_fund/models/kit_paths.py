"""KiT: K-line Diffusion Transformer for conditional OHLCV path generation.

Zhang, B. & Li, H. (2026), "KiT: A Foundation Model for Financial
Time-Series Forecasting using DiffusionTransformers", arXiv:2609.34507
(cs.LG), submitted 28 Sep 2026 (citation verified against the arXiv abstract
page; the lane spec's id/title are correct). KiT reformulates candlestick
forecasting as conditional path generation via flow matching (Liu et al.
2023; Lipman et al. 2023; Esser et al. 2024) on a Diffusion Transformer
backbone (Peebles & Xie 2023), generating the whole forecast horizon in one
shot -- no autoregressive rollout, hence no error accumulation.

Paper math implemented here (verified against arXiv:2609.34507v1):

- Candle anatomy (Eq. 3): each OHLCV bar is encoded as the scale-free,
  exactly invertible five-dimensional log-ratio state

      x_t = (r_gap, r_body, r_up, r_dn, v)_t
          = ( ln(O_t / C_{t-1}),
              ln(C_t / O_t),
              ln(H_t / max(O_t, C_t)),
              ln(min(O_t, C_t) / L_t),
              ln((V_t + 1) / (E_t + 1)) ),

  where E_t is a strictly causal exponential moving average of volume
  (E_t uses V_{<t} only). Invertibility: given C_{t-1} and the causal EMA
  chain, a closed-form chain of exponentials recovers (O, H, L, C, V)_t.
- Structural legality (Sec. 3.2): r_up >= 0 and r_dn >= 0 by construction;
  the paper applies "a simple clamp" at generation time. Here the
  rectifier is folded INTO the decoder map -- phi(z) = max(z, 0) on the
  shadow coordinates before exponentiation, and a nonnegative range
  restriction on the volume coordinate -- so the decoder's codomain is the
  legal candle space: EVERY latent vector decodes to a candle with
  H >= max(O, C), L <= min(O, C), V >= 0 and strictly positive prices.
  The constraint is structural (a property of the decode function's range),
  not a post-hoc repair pass; a ``rectifier="softplus"`` variant gives the
  smooth interior map phi(z) = softplus(z) > 0.
- Robust normalization (Sec. 3.2): per-feature median / MAD scaling
  followed by a tanh soft-clip, ``w = tanh((x - med) / s)`` with
  ``s = 1.4826 * MAD`` (the Gaussian-consistency constant); the inverse is
  ``x = med + s * atanh(w)``, defined on |w| < 1.
- Sequence layout (Eq. 4): [register tokens | clean context bars | noised
  target bars] in ONE token stream; history is never noised and the loss is
  masked to the L_h target positions (Eq. 8).
- KiT block (Eqs. 5-7): DiT-style bidirectional self-attention with RoPE
  (Su et al. 2024) and QK-Norm (Henry et al. 2020), SwiGLU feed-forward
  (Shazeer 2020), and role-aware dual modulation -- a shared PixArt-alpha
  AdaLN trunk (Chen et al. 2024) maps ``c_base + e_t`` to modulation
  vectors evaluated twice: at flow time t for target tokens and at t = 0
  for context/register tokens ("clean-data end of the flow"). All gate
  scalars and per-layer biases are zero-initialised, so v_theta = 0 at
  initialisation.
- Training (Eqs. 1-2, 8): z_t = (1 - t) x_0 + t eps, v* = eps - x_0, the
  masked MSE over target positions, t ~ logit-normal(0, 1)
  (Esser et al. 2024). Classifier-free guidance uses the paper's three
  condition-dropout schemes: eligible identity signals (instrument /
  sector) replaced by a shared [UNK] slot, ALL identity signals by a
  [NULL] slot, and the entire history context zeroed.
- Inference (Eq. 9): Euler integration of dz = v_theta dt from z_1 = eps
  back to z_0 with DUAL classifier-free guidance,

      v_hat = v_theta + (w_id - 1)(v_theta - v_{id-drop})
                        + (w_hist - 1)(v_theta - v_{hist-drop}),

  yielding an ensemble of M plausible OHLCV trajectories decoded through
  the invertible map.

Composition (import, do not reimplement): ``_torch`` lazy-imports the
optional ``nn`` extra exactly as ``models/deep_hedging.py`` /
``models/diffusion_forecaster.py`` (the wave-17 DiffPTS lane -- the spec's
"models/diffpts.py" pointer resolved to that landed filename); the Adam +
cosine-LR seeded training loop mirrors ``_train_diffpts``. Path scores are
``metrics/energy_score.energy_score`` (Gneiting & Raftery 2007 multivariate
proper score) and ``metrics/scoring.crps_empirical`` /
``pinball_loss`` -- proper scores only. Candle naming follows the
``microstructure/candle_book_features.py`` OHLCV column conventions
(open/high/low/close/volume); that module works on polars frames so the
primitives stay local to this numpy lane.

Spec-vs-paper deviations (documented, harness-driven):

1. Volume non-negativity is enforced inside the decoder map
   (V = max((E+1) e^v - 1, 0)); the paper clamps only the two shadow
   coordinates. The lane spec asks for volume non-negativity as a
   structural constraint -- folding it into decode keeps the codomain
   legal for ANY latent, matching the shadow treatment.
2. Calendar conditioning is exposed as a generic per-bar feature channel
   (``calendar_dim`` + linear projection added to token embeddings) rather
   than the paper's fixed list (Fourier intraday clock, session position,
   day-of-week, event flags, month, year-of-day); callers supply whichever
   deterministic calendar columns their feed provides.
3. Identity conditioning is a generic tuple of cardinalities
   ``cond_cardinalities`` (paper: market / sector / instrument /
   timescale); each signal gets two reserved embedding slots -- index
   ``card`` = [NULL] (all-identity dropout / CFG axis) and ``card + 1`` =
   [UNK] (shared unknown slot). The paper's "instrument and sector" UNK
   pair becomes the configurable ``unk_signals`` index set.
4. Optimizer: Adam + cosine annealing on a single CPU thread (repo
   convention, cf. deep_regime_mixture / deep_bsde / diffusion_forecaster);
   the paper's AdamW/BF16/H200 pre-training regime is out of scope -- this
   module is the research-harness implementation, not the 284M-param
   foundation model.
5. Inference and training run in torch float32 on the extracted states;
   the numpy encode/decode/score core is float64 end-to-end.

Honesty (AGENTS.md contract): every generated path and every diagnostic
here is SYNTHETIC or model-produced -- correctness evidence, never market
evidence. Path realism is reported through PROPER scores (energy score,
CRPS, pinball, coverage/width) plus candle-consistency violation rates;
there is no Sharpe/Sortino/Calmar/P&L/NAV headline anywhere in this module,
and ``bench_kit_paths`` keys are prefixed ``SYNTHETIC_`` accordingly. No
live-trading claims, no broker connectivity. Fail-closed: invalid candles
(non-positive prices, H < max(O, C) beyond float tolerance, L > min(O, C),
negative volume), non-finite or shape-mismatched inputs, degenerate
statistics, unfitted use, non-finite training loss and unknown options all
raise. Deterministic: all randomness flows through seeded numpy Generators
plus a seeded torch init; repeated calls on CPU are bit-identical (GPU
determinism is not claimed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.energy_score import energy_score
from quant_fund.metrics.scoring import crps_empirical, pinball_loss

Array = NDArray[np.float64]

__all__ = [
    "KIT_STATE_DIM",
    "KIT_STATE_FEATURES",
    "OHLCV_COLUMNS",
    "KitConfig",
    "KitDecoded",
    "KitEncoding",
    "KitFitInfo",
    "KitPathGenerator",
    "KitSample",
    "KitStateScaler",
    "bench_kit_paths",
    "bootstrap_horizon",
    "candle_consistency_report",
    "decode_encoding",
    "decode_kit_states",
    "encode_candles",
    "encode_ohlcv",
    "kit_windows",
    "logit_normal_times",
    "path_score_report",
    "synthetic_ohlcv",
]

#: Five-dimensional state of Eq. 3, in column order.
KIT_STATE_FEATURES: tuple[str, ...] = ("r_gap", "r_body", "r_up", "r_dn", "v")
KIT_STATE_DIM = 5

#: Column order of the decoded OHLCV payload (microstructure convention).
OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

# Relative tolerance for the legal-candle input check: real feeds round
# H/L to a tick, so a sub-1e-9 relative breach is float noise, not a
# violation; anything larger raises (fail-closed).
_CANDLE_RTOL = 1e-9

# atanh clip for the tanh soft-clip inverse: |w| <= 1 - eps keeps the
# inverse finite (|atanh| <= ~16.1) while leaving generated values in
# (-1, 1) exact.
_ATANH_EPS = 1e-7

_RECTIFIERS = ("relu", "softplus")


# ---------------------------------------------------------------------------
# validation helpers (house style: fail-closed, mirror diffusion_forecaster)
# ---------------------------------------------------------------------------


def _torch() -> Any:
    """Lazily import torch (optional ``nn`` extra); fail closed with guidance."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "KiT path generation needs the optional 'nn' extra (torch): uv sync --extra nn"
        ) from exc
    return torch


def _check_count(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be an int >= 1; got {value!r}")
    return int(value)


def _check_positive(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite; got {value!r}")
    return v


def _check_choice(value: str, allowed: tuple[str, ...], name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}; got {value!r}")
    return value


def _check_ohlcv(
    open_: Array, high: Array, low: Array, close: Array, volume: Array
) -> tuple[Array, ...]:
    """Validate raw candle arrays: 1-D, equal length, finite, legal geometry."""
    cols = tuple(np.asarray(x, dtype=float).reshape(-1) for x in (open_, high, low, close, volume))
    names = ("open", "high", "low", "close", "volume")
    n = cols[0].shape[0]
    if n < 1:
        raise ValueError("candle arrays must be non-empty")
    for name, col in zip(names, cols, strict=True):
        if col.shape[0] != n:
            raise ValueError("open/high/low/close/volume must share one length")
        if not bool(np.all(np.isfinite(col))):
            raise ValueError(f"{name} must be finite (NaN/inf rejected)")
    oo, hh, ll, cc, vv = cols
    if not bool(np.all(oo > 0.0) and np.all(hh > 0.0) and np.all(ll > 0.0) and np.all(cc > 0.0)):
        raise ValueError("open/high/low/close must be strictly positive")
    if bool(np.any(vv < 0.0)):
        raise ValueError("volume must be non-negative")
    oc_max = np.maximum(oo, cc)
    oc_min = np.minimum(oo, cc)
    if bool(np.any(hh < oc_max * (1.0 - _CANDLE_RTOL))):
        raise ValueError("inconsistent candle: high < max(open, close)")
    if bool(np.any(ll > oc_min * (1.0 + _CANDLE_RTOL))):
        raise ValueError("inconsistent candle: low > min(open, close)")
    return cols


def _check_states(states: Array, name: str = "states") -> Array:
    """Validate a (n, 5) or (..., n, 5) state block of Eq. 3."""
    arr = np.asarray(states, dtype=float)
    if arr.ndim < 2 or arr.shape[-1] != KIT_STATE_DIM:
        raise ValueError(f"{name} must have trailing dim {KIT_STATE_DIM}; got {arr.shape}")
    if arr.shape[-2] < 1:
        raise ValueError(f"{name} must be non-empty")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must be finite (NaN/inf rejected)")
    return arr


def _check_candles(candles: Array, name: str = "candles") -> Array:
    """Validate an (n, 5) or (..., n, 5) OHLCV block (columns of OHLCV_COLUMNS)."""
    arr = np.asarray(candles, dtype=float)
    if arr.ndim < 2 or arr.shape[-1] != KIT_STATE_DIM:
        raise ValueError(f"{name} must have trailing dim {KIT_STATE_DIM}; got {arr.shape}")
    if arr.shape[-2] < 1:
        raise ValueError(f"{name} must be non-empty")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must be finite (NaN/inf rejected)")
    return arr


def _ema_alpha(ema_halflife: float) -> float:
    """EMA smoothing constant from a halflife in bars: alpha = 1 - 2^{-1/hl}."""
    hl = _check_positive(ema_halflife, "ema_halflife")
    return float(1.0 - math.exp(-math.log(2.0) / hl))


def _rectify(z: Array, kind: str) -> Array:
    """Nonnegativity map folded into the decoder (phi: R -> R_{>=0})."""
    _check_choice(kind, _RECTIFIERS, "rectifier")
    if kind == "relu":
        return np.maximum(z, 0.0)
    return np.asarray(np.logaddexp(z, 0.0), dtype=float)  # softplus, strictly > 0


# ---------------------------------------------------------------------------
# KiT sequence encoding (Eq. 3) -- pure numpy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KitEncoding:
    """Encoded candle sequence of Eq. 3 with the decode anchors.

    ``states``: (n, 5) columns ``(r_gap, r_body, r_up, r_dn, v)``;
    ``vol_ema``: (n,) strictly-causal volume EMA E_t used by the ``v``
    channel (E_t uses V_{<t} only); ``prev_close0`` = C_{-1} used by the
    first bar's ``r_gap``; ``ema_halflife`` the EMA parameter in bars.
    Together these make the encoding self-describing: ``decode_kit_states``
    reproduces the source candles bit-for-bit from these fields alone.
    """

    states: Array
    vol_ema: Array
    prev_close0: float
    ema_halflife: float

    @property
    def n_bars(self) -> int:
        return int(self.states.shape[0])


def _causal_volume_ema(
    volume: Array,
    alpha: float,
    ema0: float | None,
    volume0: float | None,
) -> Array:
    """Strictly causal volume EMA: E_t = alpha V_{t-1} + (1-alpha) E_{t-1}.

    Self-seeded convention when no anchor is supplied: E_1 = V_1, so the
    first bar's volume ratio is v_1 = 0. With ``ema0``/``volume0`` (the
    previous bar's EMA and volume -- both or neither), bar 1 continues the
    chain as E_1 = alpha * volume0 + (1 - alpha) * ema0.
    """
    n = int(volume.shape[0])
    ema = np.empty(n, dtype=float)
    if ema0 is None and volume0 is None:
        ema[0] = float(volume[0])
    else:
        if ema0 is None or volume0 is None:
            raise ValueError("ema0 and volume0 must be supplied together")
        e0 = float(ema0)
        v0 = float(volume0)
        if not math.isfinite(e0) or e0 < 0.0 or not math.isfinite(v0) or v0 < 0.0:
            raise ValueError("ema0/volume0 must be finite and non-negative")
        ema[0] = alpha * v0 + (1.0 - alpha) * e0
    for t in range(1, n):
        ema[t] = alpha * float(volume[t - 1]) + (1.0 - alpha) * ema[t - 1]
    return ema


def encode_ohlcv(
    open_: Array,
    high: Array,
    low: Array,
    close: Array,
    volume: Array,
    *,
    ema_halflife: float = 32.0,
    prev_close0: float | None = None,
    ema0: float | None = None,
    volume0: float | None = None,
) -> KitEncoding:
    """Encode raw OHLCV bars into the five-dimensional KiT state of Eq. 3.

    ``prev_close0`` is C_{-1} for the first bar's gap; when omitted the
    first bar is gapless (C_{-1} := O_1, so r_gap = 0). ``ema0``/``volume0``
    continue the causal volume EMA from an earlier segment (both or
    neither); without them the chain self-seeds E_1 = V_1. Fail-closed on
    illegal candles, non-positive prices, negative volume, non-finite or
    length-mismatched inputs.
    """
    oo, hh, ll, cc, vv = _check_ohlcv(open_, high, low, close, volume)
    alpha = _ema_alpha(ema_halflife)
    if prev_close0 is None:
        c_prev = float(oo[0])
    else:
        c_prev = _check_positive(prev_close0, "prev_close0")
    ema = _causal_volume_ema(vv, alpha, ema0, volume0)
    r_gap = np.log(oo / np.concatenate([[c_prev], cc[:-1]]))
    r_body = np.log(cc / oo)
    r_up = np.log(hh / np.maximum(oo, cc))
    r_dn = np.log(np.minimum(oo, cc) / ll)
    v_ch = np.log((vv + 1.0) / (ema + 1.0))
    states = np.column_stack([r_gap, r_body, r_up, r_dn, v_ch])
    if not bool(np.all(np.isfinite(states))):
        raise ValueError("encoded states are not finite")
    return KitEncoding(
        states=np.asarray(states, dtype=float),
        vol_ema=ema,
        prev_close0=c_prev,
        ema_halflife=_check_positive(ema_halflife, "ema_halflife"),
    )


def encode_candles(candles: Array, **kwargs: Any) -> KitEncoding:
    """``encode_ohlcv`` on an (n, 5) OHLCV block (columns of OHLCV_COLUMNS)."""
    arr = _check_candles(candles)
    return encode_ohlcv(arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3], arr[..., 4], **kwargs)


# ---------------------------------------------------------------------------
# Structural decoder -- Eq. 3 inverted, codomain restricted to legal candles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KitDecoded:
    """Decoded candles plus the chain anchors needed to continue decoding.

    ``candles``: (n, 5) OHLCV (or batched (..., n, 5)); ``last_close`` /
    ``last_volume`` / ``last_ema`` (scalars for (n, 5) input, else (...,)
    arrays) are exactly the ``prev_close0`` / ``volume0`` / ``ema0`` a
    continuation :func:`encode_ohlcv` or a further decode needs.
    """

    candles: Array
    last_close: Array
    last_volume: Array
    last_ema: Array


def decode_kit_states(
    states: Array,
    prev_close0: float,
    *,
    ema_halflife: float = 32.0,
    vol_ema: Array | None = None,
    ema0: float | None = None,
    volume0: float | None = None,
    rectifier: str = "relu",
) -> KitDecoded:
    """Invert Eq. 3 with the legality constraints inside the decode map.

    Per bar: O_t = C_{t-1} e^{r_gap}; C_t = O_t e^{r_body};
    H_t = max(O_t, C_t) e^{phi(r_up)}; L_t = min(O_t, C_t) e^{-phi(r_dn)};
    V_t = max((E_t + 1) e^v - 1, 0), where phi is the nonnegative
    ``rectifier`` ("relu" is the paper's generation-time clamp folded into
    the map, so the constraint is structural -- the decoder CANNOT emit an
    illegal candle -- not a post-hoc repair; "softplus" is the strictly
    interior variant) and the volume coordinate passes through a
    nonnegative range restriction.

    ``states`` is (n, 5) or batched (..., n, 5); the time loop runs over
    the second-to-last axis and stays vectorized over every leading axis.
    Volume EMA modes (mutually exclusive): ``vol_ema`` (n,) reproduces a
    recorded causal EMA exactly (round-trip of :func:`encode_ohlcv`);
    ``ema0`` + ``volume0`` continue the chain E_1 = alpha*volume0 +
    (1-alpha)*ema0 from a context tail -- the generation path. Fail-closed
    on conflicting or missing anchors and on non-finite decoded candles.
    """
    st = _check_states(states)
    _check_choice(rectifier, _RECTIFIERS, "rectifier")
    alpha = _ema_alpha(ema_halflife)
    c0 = _check_positive(prev_close0, "prev_close0")
    n = int(st.shape[-2])
    lead = st.shape[:-2]

    if vol_ema is not None:
        if ema0 is not None or volume0 is not None:
            raise ValueError("conflicting volume anchors: pass vol_ema OR ema0+volume0, not both")
        ema = np.asarray(vol_ema, dtype=float).reshape(-1)
        if ema.shape[0] != n:
            raise ValueError(f"vol_ema must have length {n}; got {ema.shape[0]}")
        if not bool(np.all(np.isfinite(ema))) or bool(np.any(ema < 0.0)):
            raise ValueError("vol_ema must be finite and non-negative")
        ema_b = np.broadcast_to(ema, lead + (n,))
        rolling = False
        e_prev: Array = np.zeros(lead, dtype=float)
        v_prev: Array = np.zeros(lead, dtype=float)
    else:
        if ema0 is None or volume0 is None:
            raise ValueError(
                "decode needs a volume anchor: pass vol_ema (exact) or "
                "ema0+volume0 (rolling continuation)"
            )
        e0 = float(ema0)
        v0 = float(volume0)
        if not math.isfinite(e0) or e0 < 0.0 or not math.isfinite(v0) or v0 < 0.0:
            raise ValueError("ema0/volume0 must be finite and non-negative")
        ema_b = np.zeros(lead + (n,), dtype=float)
        rolling = True
        e_prev = np.broadcast_to(np.asarray(e0), lead).copy()
        v_prev = np.broadcast_to(np.asarray(v0), lead).copy()

    r_gap = st[..., 0]
    r_body = st[..., 1]
    r_up = _rectify(st[..., 2], rectifier)
    r_dn = _rectify(st[..., 3], rectifier)
    v_ch = st[..., 4]

    out = np.empty(lead + (n, KIT_STATE_DIM), dtype=float)
    c_prev = np.broadcast_to(np.asarray(c0), lead).copy()
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(n):
            o_t = c_prev * np.exp(r_gap[..., t])
            c_t = o_t * np.exp(r_body[..., t])
            oc_max = np.maximum(o_t, c_t)
            oc_min = np.minimum(o_t, c_t)
            h_t = oc_max * np.exp(r_up[..., t])
            l_t = oc_min * np.exp(-r_dn[..., t])
            if rolling:
                e_t = alpha * v_prev + (1.0 - alpha) * e_prev
                ema_b[..., t] = e_t
            else:
                e_t = ema_b[..., t]
            v_t = np.maximum((e_t + 1.0) * np.exp(v_ch[..., t]) - 1.0, 0.0)
            out[..., t, 0] = o_t
            out[..., t, 1] = h_t
            out[..., t, 2] = l_t
            out[..., t, 3] = c_t
            out[..., t, 4] = v_t
            c_prev = c_t
            v_prev = v_t
            e_prev = e_t
    if not bool(np.all(np.isfinite(out))):
        raise ValueError(
            "decoded candles are not finite -- latent coordinates too extreme "
            "(normalized-space inputs stay bounded via the scaler)"
        )
    if not lead:
        last_close = np.asarray(float(out[-1, 3]))
        last_volume = np.asarray(float(out[-1, 4]))
        last_ema = np.asarray(float(e_prev))
    else:
        last_close = np.asarray(out[..., -1, 3], dtype=float)
        last_volume = np.asarray(out[..., -1, 4], dtype=float)
        last_ema = np.asarray(e_prev, dtype=float)
    return KitDecoded(
        candles=np.asarray(out, dtype=float),
        last_close=last_close,
        last_volume=last_volume,
        last_ema=last_ema,
    )


def decode_encoding(enc: KitEncoding, *, rectifier: str = "relu") -> KitDecoded:
    """Exact inversion of :func:`encode_ohlcv` on its recorded anchors."""
    return decode_kit_states(
        enc.states,
        enc.prev_close0,
        ema_halflife=enc.ema_halflife,
        vol_ema=enc.vol_ema,
        rectifier=rectifier,
    )


# ---------------------------------------------------------------------------
# Robust normalization (Sec. 3.2): median/MAD + tanh soft-clip
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KitStateScaler:
    """Per-feature robust scaler of Sec. 3.2.

    ``w = tanh((x - median) / scale)`` with ``scale = 1.4826 * MAD`` (the
    Gaussian consistency constant); a degenerate (MAD = 0) feature falls
    back to ``scale = 1`` so the transform stays well-defined -- the column
    is constant and maps to 0. ``inverse`` applies ``median + scale *
    atanh(clip(w, -(1-eps), 1-eps))``: exact on the open interval (-1, 1),
    the clip only guarding the measure-zero tanh asymptote.
    """

    median: Array
    scale: Array
    clip_eps: float = _ATANH_EPS

    @staticmethod
    def fit(states: Array) -> KitStateScaler:
        x = _check_states(states)
        x2 = np.asarray(x, dtype=float).reshape(-1, KIT_STATE_DIM)
        if x2.shape[0] < 2:
            raise ValueError("scaler fit needs >= 2 rows")
        med = np.median(x2, axis=0)
        mad = np.median(np.abs(x2 - med), axis=0)
        scale = np.where(mad > 0.0, 1.4826 * mad, 1.0)
        if not bool(np.all(np.isfinite(med)) and np.all(np.isfinite(scale))):
            raise ValueError("scaler statistics are not finite")
        return KitStateScaler(
            median=np.asarray(med, dtype=float),
            scale=np.asarray(scale, dtype=float),
        )

    def transform(self, states: Array) -> Array:
        x = _check_states(states)
        return np.asarray(np.tanh((x - self.median) / self.scale), dtype=float)

    def inverse(self, normalized: Array) -> Array:
        w = _check_states(normalized, "normalized")
        w_clip = np.clip(w, -(1.0 - self.clip_eps), 1.0 - self.clip_eps)
        return np.asarray(self.median + self.scale * np.arctanh(w_clip), dtype=float)


def kit_windows(states: Array, context: int, horizon: int) -> tuple[Array, Array]:
    """Sliding (context, horizon) windows over an encoded state sequence.

    ``states`` (n, 5) -> (``ctx`` (B, Lc, 5), ``tgt`` (B, Lh, 5)) with
    B = n - Lc - Lh + 1 aligned windows (the paper's single-token-stream
    train pairs). Fail-closed when no complete window exists.
    """
    st = _check_states(states)
    if st.ndim != 2:
        raise ValueError("states must be a single (n, 5) sequence")
    lc = _check_count(context, "context")
    lh = _check_count(horizon, "horizon")
    n = int(st.shape[0])
    b = n - lc - lh + 1
    if b < 1:
        raise ValueError(f"need at least context+horizon = {lc + lh} bars; got {n}")
    ctx = np.stack([st[i : i + lc] for i in range(b)], axis=0)
    tgt = np.stack([st[i + lc : i + lc + lh] for i in range(b)], axis=0)
    return np.asarray(ctx, dtype=float), np.asarray(tgt, dtype=float)


# ---------------------------------------------------------------------------
# Diagnostics: candle consistency + proper path scores
# ---------------------------------------------------------------------------


def candle_consistency_report(candles: Array) -> dict[str, float]:
    """Violation rates of the legal-candle axioms over an (..., n, 5) block.

    Counts, per bar: ``high < max(O, C)`` (upper shadow), ``low >
    min(O, C)`` (lower shadow), ``volume < 0``, non-positive prices --
    each beyond the same relative float tolerance the encoder admits.
    ``violation_rate`` is the fraction of bars failing ANY axiom. For
    :func:`decode_kit_states` output every rate is exactly 0 by
    construction; for raw/generated third-party candles the report is the
    path-realism diagnostic the paper's design is built to zero.
    """
    arr = _check_candles(candles)
    oo, hh, ll, cc, vv = (arr[..., i] for i in range(KIT_STATE_DIM))
    n_bars = int(oo.size)
    oc_max = np.maximum(oo, cc)
    oc_min = np.minimum(oo, cc)
    high_viol = hh < oc_max * (1.0 - _CANDLE_RTOL)
    low_viol = ll > oc_min * (1.0 + _CANDLE_RTOL)
    vol_viol = vv < 0.0
    price_viol = (oo <= 0.0) | (hh <= 0.0) | (ll <= 0.0) | (cc <= 0.0)
    any_viol = high_viol | low_viol | vol_viol | price_viol
    return {
        "n_bars": float(n_bars),
        "high_violation_rate": float(np.mean(high_viol)),
        "low_violation_rate": float(np.mean(low_viol)),
        "volume_violation_rate": float(np.mean(vol_viol)),
        "price_violation_rate": float(np.mean(price_viol)),
        "violation_rate": float(np.mean(any_viol)),
        "n_violations": float(np.count_nonzero(any_viol)),
    }


def path_score_report(
    ensemble: Array,
    realized: Array,
    *,
    interval: float = 0.8,
) -> dict[str, float]:
    """Proper scores of a generated state-path ensemble vs the realized path.

    Inputs live in the Eq.-3 STATE space (``ensemble`` (M, L, 5),
    ``realized`` (L, 5)): all five channels are scale-free log ratios, so
    scores are homogeneous across coordinates -- unlike raw OHLCV, where
    the volume unit (~1e4) would swamp the price channels (~1e2). Scores
    (all proper): the multivariate ``energy_score`` on the flattened path
    (Gneiting & Raftery 2007); mean marginal ``crps_empirical`` over all
    (step, channel) pairs; CRPS and pinball losses at tau in {0.1, 0.5,
    0.9} of the horizon TERMINAL log-return ``R_L = sum_t (r_gap +
    r_body)`` (exactly the close-to-close log return of the decoded path,
    no decode needed); and the empirical coverage/mean width of the
    central ``interval`` band on the per-step cumulative log-return path.
    No Sharpe-family quantities -- proper scores and coverage/width
    diagnostics only.
    """
    ens = _check_states(ensemble, "ensemble")
    if ens.ndim != 3:
        raise ValueError(f"ensemble must be (M, L, 5); got ndim={ens.ndim}")
    obs = _check_states(realized, "realized")
    if obs.ndim != 2:
        raise ValueError(f"realized must be (L, 5); got ndim={obs.ndim}")
    m, lh, _ = ens.shape
    if obs.shape[0] != lh:
        raise ValueError(f"realized length {obs.shape[0]} != ensemble horizon {lh}")
    if not 0.0 < float(interval) < 1.0:
        raise ValueError("interval must be in (0, 1)")

    es = energy_score(ens.reshape(m, -1), obs.reshape(-1))
    crps_cells = [
        crps_empirical(float(obs[t, j]), ens[:, t, j])
        for t in range(lh)
        for j in range(KIT_STATE_DIM)
    ]
    crps_marginal = float(np.nanmean(np.asarray(crps_cells, dtype=float)))

    # cumulative log-close path and terminal return (sum of gap + body).
    cum_ens = np.cumsum(ens[..., 0] + ens[..., 1], axis=1)  # (M, L)
    cum_obs = np.cumsum(obs[:, 0] + obs[:, 1])  # (L,)
    term_ens = cum_ens[:, -1]
    term_obs = float(cum_obs[-1])
    crps_terminal = crps_empirical(term_obs, np.asarray(term_ens, dtype=float))
    q_ens = np.quantile(np.asarray(term_ens, dtype=float), [0.1, 0.5, 0.9])
    pin = [
        float(np.mean(pinball_loss(np.asarray([term_obs]), np.asarray([q_ens[k]]), tau)))
        for k, tau in enumerate((0.1, 0.5, 0.9))
    ]
    alpha2 = (1.0 - float(interval)) / 2.0
    band_lo = np.quantile(cum_ens, alpha2, axis=0)
    band_hi = np.quantile(cum_ens, 1.0 - alpha2, axis=0)
    cover = np.asarray((cum_obs >= band_lo) & (cum_obs <= band_hi), dtype=float)
    return {
        "n_samples": float(m),
        "horizon": float(lh),
        "energy_score": float(es),
        "crps_marginal_mean": float(crps_marginal),
        "crps_terminal_ret": float(crps_terminal),
        "pinball_10_terminal_ret": pin[0],
        "pinball_50_terminal_ret": pin[1],
        "pinball_90_terminal_ret": pin[2],
        "coverage_ret": float(np.mean(cover)),
        "width_ret": float(np.mean(band_hi - band_lo)),
        "interval": float(interval),
    }


# ---------------------------------------------------------------------------
# SYNTHETIC data + non-parametric reference sampler
# ---------------------------------------------------------------------------


def synthetic_ohlcv(
    n_bars: int,
    seed: int,
    *,
    s0: float = 100.0,
    ema_halflife: float = 32.0,
    base_volume: float = 1e4,
    vol_persist: float = 0.97,
) -> Array:
    """Seeded SYNTHETIC candle stream -- correctness fixture, never market data.

    A two-regime Markov chain over latent state scales: calm
    (sigma_body = 0.004) vs storm (0.02) with persistence
    ``vol_persist``; r_up/r_dn half-normal, r_gap small, the volume channel
    v ~ N(0, 0.5). Latents decode through :func:`decode_kit_states`
    (rolling EMA seeded at ``base_volume``), so every emitted candle is
    legal BY CONSTRUCTION -- the fixture cannot produce a consistency
    violation. Deterministic given ``seed``.
    """
    n = _check_count(n_bars, "n_bars")
    _check_positive(s0, "s0")
    _check_positive(base_volume, "base_volume")
    if not 0.0 <= float(vol_persist) < 1.0:
        raise ValueError("vol_persist must be in [0, 1)")
    rng = np.random.default_rng(int(seed))
    regime = np.empty(n, dtype=int)
    regime[0] = 0
    for t in range(1, n):
        if regime[t - 1] == 0:
            regime[t] = int(rng.random() > vol_persist)
        else:
            regime[t] = int(rng.random() <= vol_persist)
    sig_body = np.where(regime == 1, 0.02, 0.004)
    sig_sh = np.where(regime == 1, 0.008, 0.0015)
    states = np.column_stack(
        [
            rng.normal(0.0, 0.002, n),  # r_gap
            rng.normal(0.0, 1.0, n) * sig_body,  # r_body
            np.abs(rng.normal(0.0, 1.0, n)) * sig_sh + 1e-5,  # r_up >= 0
            np.abs(rng.normal(0.0, 1.0, n)) * sig_sh + 1e-5,  # r_dn >= 0
            rng.normal(0.0, 0.5, n),  # v (log ratio to causal volume EMA)
        ]
    )
    dec = decode_kit_states(
        states,
        s0,
        ema_halflife=ema_halflife,
        ema0=base_volume,
        volume0=base_volume,
    )
    return dec.candles


def bootstrap_horizon(
    target_pool: Array,
    n_samples: int,
    rng: np.random.Generator | int,
) -> Array:
    """Resample horizon blocks from an encoded target pool (non-parametric).

    ``target_pool`` (B, Lh, 5) encoded state windows (e.g. from
    :func:`kit_windows`); draws ``n_samples`` rows with replacement from a
    seeded Generator. A torch-free reference sampler for the synthetic
    battery: it is exactly the empirical distribution of the encoded
    horizons, so its proper scores calibrate the fixture's sampling noise.
    """
    pool = _check_states(target_pool, "target_pool")
    if pool.ndim != 3:
        raise ValueError(f"target_pool must be (B, Lh, 5); got ndim={pool.ndim}")
    m = _check_count(n_samples, "n_samples")
    g = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    idx = g.integers(0, pool.shape[0], size=m)
    return np.asarray(pool[idx], dtype=float)


def logit_normal_times(n: int, rng: np.random.Generator | int) -> Array:
    """t = sigmoid(N(0, 1)) draws in (0, 1) -- the Eq. 2 flow-time schedule."""
    nn = _check_count(n, "n")
    g = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    u = g.standard_normal(nn)
    return np.asarray(1.0 / (1.0 + np.exp(-u)), dtype=float)


# ---------------------------------------------------------------------------
# torch lane: KiT backbone (DiT + role-aware dual AdaLN) -- lazily gated
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KitConfig:
    """Backbone/training/sampling configuration for :class:`KitPathGenerator`.

    ``cond_cardinalities``: per-signal identity vocab size (market /
    sector / instrument / timescale in the paper); each signal reserves
    slots ``card`` = [NULL] and ``card + 1`` = [UNK]. ``unk_signals``:
    signal indices eligible for the shared-UNK drop (paper: instrument and
    sector). ``calendar_dim``: width of the optional per-bar calendar
    channel added to token embeddings. ``ffn_hidden=None`` -> the SwiGLU
    width ``round(8 d / 3)`` (DiT-ratio equivalent). Drop probabilities are
    the three CFG training schemes of Sec. 3.5; ``w_id``/``w_hist`` the
    dual-guidance weights of Eq. 9 (>= 1). ``n_euler`` ODE steps for
    inference.
    """

    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    n_register: int = 4
    ffn_hidden: int | None = None
    cond_cardinalities: tuple[int, ...] = ()
    unk_signals: tuple[int, ...] = ()
    calendar_dim: int = 0
    p_unk: float = 0.1
    p_null: float = 0.1
    p_hist: float = 0.1
    w_id: float = 1.0
    w_hist: float = 1.0
    n_euler: int = 16
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int | None = 64
    seed: int = 0

    def __post_init__(self) -> None:
        if self.d_model < 1 or self.n_heads < 1 or self.n_layers < 1:
            raise ValueError("d_model/n_heads/n_layers must be >= 1")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("head dim must be even (RoPE)")
        if self.n_register < 0:
            raise ValueError("n_register must be >= 0")
        if self.ffn_hidden is not None and self.ffn_hidden < 1:
            raise ValueError("ffn_hidden must be >= 1 or None")
        cards = tuple(int(c) for c in self.cond_cardinalities)
        if any(c < 1 for c in cards):
            raise ValueError("cond_cardinalities must be >= 1")
        object.__setattr__(self, "cond_cardinalities", cards)
        unk = tuple(int(i) for i in self.unk_signals)
        if any(i < 0 or i >= len(cards) for i in unk):
            raise ValueError("unk_signals must index cond_cardinalities")
        object.__setattr__(self, "unk_signals", unk)
        if self.calendar_dim < 0:
            raise ValueError("calendar_dim must be >= 0")
        for name in ("p_unk", "p_null", "p_hist"):
            p = float(getattr(self, name))
            if not math.isfinite(p) or not 0.0 <= p <= 1.0:
                raise ValueError(f"{name} must be a probability in [0, 1]")
        for name in ("w_id", "w_hist"):
            w = float(getattr(self, name))
            if not math.isfinite(w) or w < 1.0:
                raise ValueError(f"{name} must be >= 1 (Eq. 9 guidance weight)")
        _check_count(self.n_euler, "n_euler")
        _check_count(self.epochs, "epochs")
        _check_positive(self.lr, "lr")
        if self.batch_size is not None:
            _check_count(self.batch_size, "batch_size")


def _build_kit_backbone(torch: Any, cfg: KitConfig, context_len: int, horizon_len: int) -> Any:
    """Construct the KiT DiT (Eqs. 4-7); the class lives here so the module
    imports without torch (the ``nn`` extra)."""
    d = cfg.d_model
    n_heads = cfg.n_heads
    head_dim = d // n_heads
    ffn_hidden = cfg.ffn_hidden if cfg.ffn_hidden is not None else max(1, int(round(8 * d / 3)))
    n_blocks_out = 8  # 6 block modulations (b1,g1,a1,b2,g2,a2) + final (b_f, g_f)

    class _RMSNorm(torch.nn.Module):
        """Parameter-free RMSNorm: the affine terms come from AdaLN."""

        def __init__(self, dim: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.eps = eps

        def forward(self, x: Any) -> Any:
            return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def _rope_tables(seq_len: int) -> tuple[Any, Any]:
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        ang = torch.arange(seq_len).float()[:, None] * inv_freq[None, :]
        return torch.cos(ang), torch.sin(ang)  # (S, hd/2)

    class _Attention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv = torch.nn.Linear(d, 3 * d, bias=False)
            self.proj = torch.nn.Linear(d, d)

        def forward(self, x: Any, cos: Any, sin: Any) -> Any:
            b, s, _ = x.shape
            qkv = self.qkv(x).reshape(b, s, 3, n_heads, head_dim)
            q, k, v = qkv.unbind(dim=2)  # (B, S, H, hd)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)  # (B, H, S, hd)
            # QK-Norm (Henry et al. 2020): RMS-normalize q, k per head dim.
            q = q * torch.rsqrt(q.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            k = k * torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            # RoPE (Su et al. 2024): rotate-half pairs.
            half = head_dim // 2
            qc = torch.cat(
                [
                    q[..., :half] * cos - q[..., half:] * sin,
                    q[..., half:] * cos + q[..., :half] * sin,
                ],
                dim=-1,
            )
            kc = torch.cat(
                [
                    k[..., :half] * cos - k[..., half:] * sin,
                    k[..., half:] * cos + k[..., :half] * sin,
                ],
                dim=-1,
            )
            # Bidirectional (non-causal) attention: every token attends to
            # the full register|context|target stream.
            o = torch.nn.functional.scaled_dot_product_attention(qc, kc, v)
            o = o.transpose(1, 2).reshape(b, s, d)
            return self.proj(o)

    class _SwiGLU(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w1 = torch.nn.Linear(d, ffn_hidden)
            self.w3 = torch.nn.Linear(d, ffn_hidden)
            self.w2 = torch.nn.Linear(ffn_hidden, d)

        def forward(self, x: Any) -> Any:
            return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

    class _KitBlock(torch.nn.Module):
        """Eqs. 5-7: adaLN-modulated attention + SwiGLU, bidirectional."""

        def __init__(self) -> None:
            super().__init__()
            self.norm1 = _RMSNorm(d)
            self.attn = _Attention()
            self.norm2 = _RMSNorm(d)
            self.ffn = _SwiGLU()

        def forward(self, x: Any, mod6: Any, cos: Any, sin: Any) -> Any:
            b1, g1, a1, b2, g2, a2 = mod6.chunk(6, dim=-1)
            h = self.norm1(x) * (1.0 + g1) + b1
            x = x + a1 * self.attn(h, cos, sin)
            h2 = self.norm2(x) * (1.0 + g2) + b2
            x = x + a2 * self.ffn(h2)
            return x

    class _KitDiT(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.d = d
            self.n_reg = cfg.n_register
            self.lc = context_len
            self.lh = horizon_len
            self.in_proj = torch.nn.Linear(KIT_STATE_DIM, d)
            self.role_ctx = torch.nn.Parameter(torch.zeros(d))
            self.role_tgt = torch.nn.Parameter(torch.zeros(d))
            self.registers = torch.nn.Parameter(torch.zeros(max(cfg.n_register, 0), d))
            self.cond_embs = torch.nn.ModuleList(
                [torch.nn.Embedding(card + 2, d) for card in cfg.cond_cardinalities]
            )
            self.cal_proj = torch.nn.Linear(cfg.calendar_dim, d) if cfg.calendar_dim > 0 else None
            # Shared AdaLN trunk (PixArt-alpha style): c_base + e_t -> 8d.
            self.t_embed_dim = d
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(d, d),
                torch.nn.SiLU(),
                torch.nn.Linear(d, n_blocks_out * d),
            )
            self.blocks = torch.nn.ModuleList([_KitBlock() for _ in range(cfg.n_layers)])
            # Per-layer learnable biases b_l (Eq. 5), zero-initialised.
            self.block_bias = torch.nn.Parameter(torch.zeros(cfg.n_layers, 6 * d))
            self.final_norm = _RMSNorm(d)
            self.head = torch.nn.Linear(d, KIT_STATE_DIM)
            total_len = cfg.n_register + context_len + horizon_len
            cos, sin = _rope_tables(total_len)
            self.register_buffer("rope_cos", cos)
            self.register_buffer("rope_sin", sin)
            # Zero-init: trunk output (gates alpha + final modulation) and
            # the head -> every block is an identity and v_theta = 0 at init.
            with torch.no_grad():
                torch.nn.init.zeros_(self.trunk[-1].weight)
                torch.nn.init.zeros_(self.trunk[-1].bias)
                torch.nn.init.zeros_(self.head.weight)
                torch.nn.init.zeros_(self.head.bias)

        def _time_embed(self, t: Any) -> Any:
            # Sinusoidal flow-time embedding e_t (Sec. 3.3).
            half = self.t_embed_dim // 2
            freqs = torch.exp(
                -math.log(10000.0)
                * torch.arange(half, device=t.device, dtype=t.dtype)
                / max(half, 1)
            )
            arg = (t[:, None] * 1000.0) * freqs[None, :]
            emb = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)
            if emb.shape[-1] < self.t_embed_dim:
                emb = torch.nn.functional.pad(emb, (0, self.t_embed_dim - emb.shape[-1]))
            return emb

        def forward(
            self,
            ctx_x: Any,
            tgt_z: Any,
            t: Any,
            cond_ids: Any | None = None,
            cal_ctx: Any | None = None,
            cal_tgt: Any | None = None,
            null_ids: Any | None = None,
            hist_mask: Any | None = None,
        ) -> Any:
            """Velocity prediction on the target span, (B, Lh, 5).

            ``null_ids`` (B,) bool -> all identity slots become [NULL]
            (the v_{id-drop} axis of Eq. 9); ``hist_mask`` (B,) bool -> the
            context tokens are zeroed entirely (v_{hist-drop}).
            """
            b = tgt_z.shape[0]
            emb = torch.zeros(b, self.d, device=tgt_z.device, dtype=tgt_z.dtype)
            for i, table in enumerate(self.cond_embs):
                if cond_ids is None:
                    ids = torch.full(
                        (b,), table.num_embeddings - 2, dtype=torch.long, device=tgt_z.device
                    )
                else:
                    ids = cond_ids[:, i]
                if null_ids is not None:
                    ids = torch.where(null_ids, torch.full_like(ids, table.num_embeddings - 2), ids)
                emb = emb + table(ids)
            e_t = self._time_embed(t)
            c_tgt = self.trunk(emb + e_t)  # (B, 8d)
            e_0 = self._time_embed(torch.zeros_like(t))
            c_ctx = self.trunk(emb + e_0)  # t = 0: the clean-data end

            x_ctx = self.in_proj(ctx_x) + self.role_ctx
            x_tgt = self.in_proj(tgt_z) + self.role_tgt
            if self.cal_proj is not None:
                if cal_ctx is None or cal_tgt is None:
                    raise ValueError("calendar features required (calendar_dim > 0)")
                x_ctx = x_ctx + self.cal_proj(cal_ctx)
                x_tgt = x_tgt + self.cal_proj(cal_tgt)
            if hist_mask is not None:
                x_ctx = torch.where(hist_mask[:, None, None], torch.zeros_like(x_ctx), x_ctx)
            regs = self.registers.unsqueeze(0).expand(b, -1, -1)
            x = torch.cat([regs, x_ctx, x_tgt], dim=1)

            n_ctx_role = self.n_reg + self.lc
            for li, blk in enumerate(self.blocks):
                mod_ctx = c_ctx[:, : 6 * self.d][:, None, :].expand(b, n_ctx_role, 6 * self.d)
                mod_tgt = c_tgt[:, : 6 * self.d][:, None, :].expand(b, self.lh, 6 * self.d)
                mod = torch.cat([mod_ctx, mod_tgt], dim=1) + self.block_bias[li][None, None, :]
                x = blk(x, mod, self.rope_cos, self.rope_sin)
            out_tokens = x[:, n_ctx_role:, :]
            b_f = c_tgt[:, 6 * self.d : 7 * self.d]
            g_f = c_tgt[:, 7 * self.d : 8 * self.d]
            h = self.final_norm(out_tokens) * (1.0 + g_f[:, None, :]) + b_f[:, None, :]
            return self.head(h)

    return _KitDiT()


def _apply_condition_dropout(
    cond_ids: NDArray[np.int64] | None,
    cfg: KitConfig,
    n_batch: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int64] | None, NDArray[np.bool_]]:
    """Sec. 3.5 three stochastic condition-dropout schemes (per-sample).

    Returns (cond_ids', hist_mask): scheme (i) the ``unk_signals`` slots of
    a drawn subset of samples become [UNK] (index card+1); scheme (ii) all
    identity slots of a drawn subset become [NULL] (index card); scheme
    (iii) a drawn subset gets ``hist_mask`` = True (context zeroed).
    """
    hist_mask = rng.random(n_batch) < cfg.p_hist
    if cond_ids is None or len(cfg.cond_cardinalities) == 0:
        return None, hist_mask
    ids = np.asarray(cond_ids, dtype=np.int64).copy()
    cards = np.asarray(cfg.cond_cardinalities, dtype=np.int64)
    if bool(np.any(ids < 0)) or bool(np.any(ids >= cards[None, :])):
        raise ValueError("cond_ids out of range for cond_cardinalities")
    if cfg.unk_signals and cfg.p_unk > 0.0:
        m_unk = rng.random(n_batch) < cfg.p_unk
        for sig in cfg.unk_signals:
            ids[m_unk, sig] = cards[sig] + 1  # shared [UNK] slot
    if cfg.p_null > 0.0:
        m_null = rng.random(n_batch) < cfg.p_null
        ids[m_null, :] = cards[None, :]  # all-identity [NULL]
    return ids, hist_mask


def _kit_flow_matching_loss(
    torch: Any,
    model: Any,
    ctx: Any,
    tgt: Any,
    cond_ids: Any | None,
    cal_ctx: Any | None,
    cal_tgt: Any | None,
    t: Any,
    eps: Any,
    hist_mask: Any | None,
) -> Any:
    """Eqs. 1-2, 8: masked MSE of the velocity on the target span only."""
    t_col = t.reshape(-1, 1, 1).to(tgt.dtype)
    z_t = (1.0 - t_col) * tgt + t_col * eps
    v_star = eps - tgt
    v_hat = model(
        ctx,
        z_t,
        t,
        cond_ids=cond_ids,
        cal_ctx=cal_ctx,
        cal_tgt=cal_tgt,
        hist_mask=hist_mask,
    )
    resid = v_hat - v_star
    return torch.mean(resid * resid)


def _train_kit(
    torch: Any,
    model: Any,
    ctx_np: Array,
    tgt_np: Array,
    cond_ids_np: NDArray[np.int64] | None,
    cal_ctx_np: Array | None,
    cal_tgt_np: Array | None,
    cfg: KitConfig,
) -> list[float]:
    """Adam + cosine LR on the Eq. 8 loss; deterministic given ``cfg.seed``.

    numpy Generator drives t ~ logit-normal, eps ~ N(0, I), the three CFG
    dropout masks and the batch permutation -- the torch stream only seeds
    init, so the two RNG families never interleave (repo convention).
    """
    torch.manual_seed(int(cfg.seed))
    torch.set_num_threads(1)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))
    n = int(tgt_np.shape[0])
    bsz = n if cfg.batch_size is None else min(int(cfg.batch_size), n)
    steps_per_epoch = max(1, -(-n // bsz))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(cfg.epochs) * steps_per_epoch, eta_min=float(cfg.lr) / 10.0
    )
    ctx_all = torch.as_tensor(ctx_np, dtype=torch.float32)
    tgt_all = torch.as_tensor(tgt_np, dtype=torch.float32)
    cal_ctx_all = None if cal_ctx_np is None else torch.as_tensor(cal_ctx_np, dtype=torch.float32)
    cal_tgt_all = None if cal_tgt_np is None else torch.as_tensor(cal_tgt_np, dtype=torch.float32)
    rng = np.random.default_rng(int(cfg.seed))
    curve: list[float] = []
    for _epoch in range(int(cfg.epochs)):
        order = np.arange(n) if cfg.batch_size is None else rng.permutation(n)
        for start in range(0, n, bsz):
            idx = order[start : start + bsz]
            idx_t = torch.as_tensor(idx, dtype=torch.long)
            nb = int(idx.size)
            t_np = logit_normal_times(nb, rng)
            eps_np = rng.standard_normal((nb,) + tgt_np.shape[1:])
            ids_np, hist_np = _apply_condition_dropout(
                None if cond_ids_np is None else cond_ids_np[idx],
                cfg,
                nb,
                rng,
            )
            opt.zero_grad(set_to_none=True)
            loss = _kit_flow_matching_loss(
                torch,
                model,
                ctx_all[idx_t],
                tgt_all[idx_t],
                None if ids_np is None else torch.as_tensor(ids_np, dtype=torch.long),
                None if cal_ctx_all is None else cal_ctx_all[idx_t],
                None if cal_tgt_all is None else cal_tgt_all[idx_t],
                torch.as_tensor(t_np, dtype=torch.float32),
                torch.as_tensor(eps_np, dtype=torch.float32),
                torch.as_tensor(hist_np),
            )
            lv = float(loss.detach().numpy())
            if not math.isfinite(lv):
                raise ValueError(
                    f"KiT training loss is not finite (loss={lv!r}); reduce lr or check inputs"
                )
            curve.append(lv)
            loss.backward()
            opt.step()
            sched.step()
    return curve


def _kit_euler_sample(
    torch: Any,
    model: Any,
    ctx_np: Array,
    n_samples: int,
    cfg: KitConfig,
    *,
    cond_ids_np: NDArray[np.int64] | None = None,
    cal_ctx_np: Array | None = None,
    cal_tgt_np: Array | None = None,
    n_euler: int | None = None,
    w_id: float | None = None,
    w_hist: float | None = None,
    seed: int = 0,
) -> Array:
    """Eq. 9 inference: Euler ODE z_1 -> z_0 with dual classifier-free guidance.

    Extra velocity evaluations are skipped when their weight is 1 (the
    paper's default). z_1 ~ N(0, I) comes from a seeded numpy Generator;
    the ODE is deterministic, so the sample is bit-identical for a fixed
    (model, seed, shapes).
    """
    torch.set_num_threads(1)
    steps = _check_count(n_euler if n_euler is not None else cfg.n_euler, "n_euler")
    wi = float(cfg.w_id if w_id is None else w_id)
    wh = float(cfg.w_hist if w_hist is None else w_hist)
    if wi < 1.0 or wh < 1.0:
        raise ValueError("guidance weights must be >= 1")
    ctx = np.asarray(ctx_np, dtype=float)
    n_ctx = int(ctx.shape[0])
    lh = int(model.lh)
    rng = np.random.default_rng(int(seed))
    ctx_rep = np.repeat(ctx, int(n_samples), axis=0)
    b = int(ctx_rep.shape[0])
    z = rng.standard_normal((b, lh, KIT_STATE_DIM))
    cond_rep = (
        None
        if cond_ids_np is None
        else np.repeat(
            np.atleast_2d(np.asarray(cond_ids_np, dtype=np.int64)), int(n_samples), axis=0
        )
    )
    cal_ctx_rep = (
        None
        if cal_ctx_np is None
        else np.repeat(np.asarray(cal_ctx_np, dtype=float), int(n_samples), axis=0)
    )
    cal_tgt_rep = (
        None
        if cal_tgt_np is None
        else np.repeat(np.asarray(cal_tgt_np, dtype=float), int(n_samples), axis=0)
    )
    ctx_t = torch.as_tensor(ctx_rep, dtype=torch.float32)
    ids_t = None if cond_rep is None else torch.as_tensor(cond_rep, dtype=torch.long)
    cal_ctx_t = None if cal_ctx_rep is None else torch.as_tensor(cal_ctx_rep, dtype=torch.float32)
    cal_tgt_t = None if cal_tgt_rep is None else torch.as_tensor(cal_tgt_rep, dtype=torch.float32)
    all_true = torch.ones(b, dtype=torch.bool)
    z_t = torch.as_tensor(z, dtype=torch.float32)
    dt = 1.0 / float(steps)
    with torch.no_grad():
        for k in range(steps):
            t_now = 1.0 - k * dt
            t_vec = torch.full((b,), t_now, dtype=torch.float32)
            v_c = model(
                ctx_t,
                z_t,
                t_vec,
                cond_ids=ids_t,
                cal_ctx=cal_ctx_t,
                cal_tgt=cal_tgt_t,
            )
            v = v_c
            if wi > 1.0:
                v_id = model(
                    ctx_t,
                    z_t,
                    t_vec,
                    cond_ids=ids_t,
                    cal_ctx=cal_ctx_t,
                    cal_tgt=cal_tgt_t,
                    null_ids=all_true,
                )
                v = v + (wi - 1.0) * (v_c - v_id)
            if wh > 1.0:
                v_hist = model(
                    ctx_t,
                    z_t,
                    t_vec,
                    cond_ids=ids_t,
                    cal_ctx=cal_ctx_t,
                    cal_tgt=cal_tgt_t,
                    hist_mask=all_true,
                )
                v = v + (wh - 1.0) * (v_c - v_hist)
            z_t = z_t - dt * v
    out = np.asarray(z_t.numpy(), dtype=float)
    return out.reshape(n_ctx, int(n_samples), lh, KIT_STATE_DIM)


# ---------------------------------------------------------------------------
# the generator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KitFitInfo:
    """Training trace: per-update masked-MSE curve (Eq. 8) and metadata."""

    epochs: int
    n_windows: int
    context: int
    horizon: int
    seed: int
    loss_curve: list[float]
    final_loss: float


@dataclass(frozen=True)
class KitSample:
    """Generated OHLCV path ensemble plus decode anchors.

    ``paths`` (M, Lh, 5) decoded candles -- legal by construction;
    ``states`` (M, Lh, 5) the Eq.-3 latents the candles were decoded from
    (pre-rectifier: the nonnegativity map phi is applied inside decode, so
    shadow coordinates here may read negative while the decoded candle is
    still legal); ``normalized`` (M, Lh, 5) the raw ODE output, tanh space.
    """

    paths: Array
    states: Array
    normalized: Array
    consistency: dict[str, float]


class KitPathGenerator:
    """KiT conditional OHLCV path generator (arXiv:2609.34507).

    ``fit_ohlcv`` encodes a training candle stream (Eq. 3), fits the
    robust scaler on the training states ONLY (leakage-free, paper Sec.
    4.1), builds sliding (context, horizon) windows, and trains the DiT
    velocity field on the masked flow-matching MSE with the three CFG
    dropout schemes. ``sample_ohlcv`` encodes a fresh context, integrates
    the learned ODE with dual guidance (Eq. 9), inverse-transforms and
    structurally decodes an ensemble of ``n_samples`` legal candle paths.
    Requires the torch ``nn`` extra for fit/sample; encode/decode/score
    stay pure numpy.
    """

    def __init__(self, config: KitConfig | None = None, *, ema_halflife: float = 32.0) -> None:
        self.config = config if config is not None else KitConfig()
        self.ema_halflife = _check_positive(ema_halflife, "ema_halflife")
        self._model: Any | None = None
        self._scaler: KitStateScaler | None = None
        self._context_len = 0
        self._horizon_len = 0
        self.fit_info: KitFitInfo | None = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None and self._scaler is not None

    def _require_fitted(self) -> tuple[Any, KitStateScaler]:
        if self._model is None or self._scaler is None:
            raise RuntimeError("KitPathGenerator is not fitted")
        return self._model, self._scaler

    def fit_ohlcv(
        self,
        candles: Array,
        *,
        context: int,
        horizon: int,
        cond_ids: Array | None = None,
        calendar: Array | None = None,
    ) -> KitPathGenerator:
        """Train on an (n, 5) candle stream (SYNTHETIC or research data).

        ``cond_ids``: optional (B, K) per-window integer identity codes
        matching ``config.cond_cardinalities`` (B = number of windows).
        ``calendar``: optional (n, calendar_dim) per-bar features, windowed
        alongside the states. Validation precedes torch, so contract errors
        raise ``ValueError`` even without the ``nn`` extra.
        """
        arr = _check_candles(candles)
        if arr.ndim != 2:
            raise ValueError(f"candles must be (n, 5); got ndim={arr.ndim}")
        lc = _check_count(context, "context")
        lh = _check_count(horizon, "horizon")
        enc = encode_candles(arr, ema_halflife=self.ema_halflife)
        scaler = KitStateScaler.fit(enc.states)
        normed = scaler.transform(enc.states)
        ctx, tgt = kit_windows(normed, lc, lh)
        n_win = int(ctx.shape[0])
        cfg = self.config

        ids_np: NDArray[np.int64] | None = None
        if len(cfg.cond_cardinalities) > 0:
            if cond_ids is None:
                raise ValueError("cond_ids required when cond_cardinalities is non-empty")
            ids_np = np.asarray(cond_ids, dtype=np.int64)
            if ids_np.shape != (n_win, len(cfg.cond_cardinalities)):
                raise ValueError(f"cond_ids must be (n_windows, {len(cfg.cond_cardinalities)})")
            cards = np.asarray(cfg.cond_cardinalities, dtype=np.int64)
            if bool(np.any(ids_np < 0)) or bool(np.any(ids_np >= cards[None, :])):
                raise ValueError("cond_ids out of range for cond_cardinalities")
        elif cond_ids is not None:
            raise ValueError("cond_ids given but cond_cardinalities is empty")

        cal_ctx_np: Array | None = None
        cal_tgt_np: Array | None = None
        if cfg.calendar_dim > 0:
            if calendar is None:
                raise ValueError("calendar required when calendar_dim > 0")
            cal = np.asarray(calendar, dtype=float)
            if cal.shape != (arr.shape[0], cfg.calendar_dim):
                raise ValueError(f"calendar must be (n_bars, {cfg.calendar_dim})")
            if not bool(np.all(np.isfinite(cal))):
                raise ValueError("calendar must be finite")
            cal_ctx_np, cal_tgt_np = kit_windows(cal, lc, lh)
        elif calendar is not None:
            raise ValueError("calendar given but calendar_dim = 0")

        torch = _torch()
        model = _build_kit_backbone(torch, cfg, lc, lh)
        curve = _train_kit(torch, model, ctx, tgt, ids_np, cal_ctx_np, cal_tgt_np, cfg)
        self._model = model
        self._scaler = scaler
        self._context_len = lc
        self._horizon_len = lh
        self.fit_info = KitFitInfo(
            epochs=int(cfg.epochs),
            n_windows=n_win,
            context=lc,
            horizon=lh,
            seed=int(cfg.seed),
            loss_curve=curve,
            final_loss=float(curve[-1]),
        )
        return self

    def sample_states(
        self,
        context_candles: Array,
        *,
        n_samples: int = 16,
        n_euler: int | None = None,
        w_id: float | None = None,
        w_hist: float | None = None,
        cond_ids: Array | None = None,
        calendar_context: Array | None = None,
        calendar_target: Array | None = None,
        seed: int = 0,
    ) -> Array:
        """Sample normalized-state paths: returns (M, Lh, 5) in tanh space."""
        model, scaler = self._require_fitted()
        arr = _check_candles(context_candles, "context_candles")
        if arr.ndim != 2 or arr.shape[0] != self._context_len:
            raise ValueError(f"context_candles must be ({self._context_len}, 5); got {arr.shape}")
        m = _check_count(n_samples, "n_samples")
        enc = encode_candles(arr, ema_halflife=self.ema_halflife)
        ctx_norm = scaler.transform(enc.states)[None, :, :]
        ids_np: NDArray[np.int64] | None = None
        if cond_ids is not None:
            ids_np = np.asarray(cond_ids, dtype=np.int64).reshape(1, -1)
            if ids_np.shape[1] != len(self.config.cond_cardinalities):
                raise ValueError("cond_ids width != len(cond_cardinalities)")
        elif len(self.config.cond_cardinalities) > 0:
            raise ValueError("cond_ids required when cond_cardinalities is non-empty")
        cal_ctx_np: Array | None = None
        cal_tgt_np: Array | None = None
        if self.config.calendar_dim > 0:
            if calendar_context is None or calendar_target is None:
                raise ValueError("calendar_context and calendar_target required")
            cal_ctx_np = np.asarray(calendar_context, dtype=float)[None, :, :]
            cal_tgt_np = np.asarray(calendar_target, dtype=float)[None, :, :]
            if cal_ctx_np.shape != (1, self._context_len, self.config.calendar_dim):
                raise ValueError("calendar_context shape mismatch")
            if cal_tgt_np.shape != (1, self._horizon_len, self.config.calendar_dim):
                raise ValueError("calendar_target shape mismatch")
        torch = _torch()
        out = _kit_euler_sample(
            torch,
            model,
            ctx_norm,
            m,
            self.config,
            cond_ids_np=ids_np,
            cal_ctx_np=cal_ctx_np,
            cal_tgt_np=cal_tgt_np,
            n_euler=n_euler,
            w_id=w_id,
            w_hist=w_hist,
            seed=int(seed),
        )
        return np.asarray(out[0], dtype=float)  # (M, Lh, 5)

    def sample_ohlcv(
        self,
        context_candles: Array,
        *,
        n_samples: int = 16,
        n_euler: int | None = None,
        w_id: float | None = None,
        w_hist: float | None = None,
        cond_ids: Array | None = None,
        calendar_context: Array | None = None,
        calendar_target: Array | None = None,
        seed: int = 0,
    ) -> KitSample:
        """Conditional path generation: (M, Lh, 5) legal OHLCV ensemble.

        The context tail (last close, last volume, last causal EMA) seeds
        the decoder chains, so generated bars continue the context's own
        volume regime; the rectified decode guarantees zero consistency
        violations.
        """
        _model, scaler = self._require_fitted()
        arr = _check_candles(context_candles, "context_candles")
        enc_ctx = encode_candles(arr, ema_halflife=self.ema_halflife)
        normed = self.sample_states(
            arr,
            n_samples=n_samples,
            n_euler=n_euler,
            w_id=w_id,
            w_hist=w_hist,
            cond_ids=cond_ids,
            calendar_context=calendar_context,
            calendar_target=calendar_target,
            seed=seed,
        )
        states = scaler.inverse(normed)
        dec = decode_kit_states(
            states,
            float(arr[-1, 3]),
            ema_halflife=self.ema_halflife,
            ema0=float(enc_ctx.vol_ema[-1]),
            volume0=float(arr[-1, 4]),
        )
        report = candle_consistency_report(dec.candles)
        return KitSample(
            paths=dec.candles,
            states=np.asarray(states, dtype=float),
            normalized=np.asarray(normed, dtype=float),
            consistency=report,
        )


# ---------------------------------------------------------------------------
# SYNTHETIC validation battery
# ---------------------------------------------------------------------------


def bench_kit_paths(
    n_train_bars: int = 512,
    context: int = 32,
    horizon: int = 8,
    n_samples: int = 64,
    seed: int = 0,
    ema_halflife: int | float = 32,
) -> dict[str, float | str]:
    """Seeded SYNTHETIC battery for the KiT path pipeline -- correctness only.

    All keys carry the ``SYNTHETIC_`` prefix (AGENTS.md honesty contract):
    encode/decode round-trip fidelity on a seeded stream (the map is
    exactly invertible), the scaler tanh round-trip, bootstrap-resampled
    horizon ensembles decoded through the structural map (violation rates
    must be identically 0), and proper scores of the ensemble vs the held
    realized horizon (energy score, marginal + terminal CRPS, pinball,
    coverage/width). No torch needed: this bench exercises the numpy core;
    the torch lane is covered by the gated unit tests.
    """
    n_tr = _check_count(n_train_bars, "n_train_bars")
    lc = _check_count(context, "context")
    lh = _check_count(horizon, "horizon")
    m = _check_count(n_samples, "n_samples")
    hl = _check_positive(ema_halflife, "ema_halflife")

    stream = synthetic_ohlcv(n_tr + lc + lh, int(seed), ema_halflife=float(hl))
    enc = encode_candles(stream, ema_halflife=float(hl))
    dec = decode_encoding(enc)
    roundtrip_err = float(np.max(np.abs(dec.candles - stream)))

    scaler = KitStateScaler.fit(enc.states)
    normed = scaler.transform(enc.states)
    scaler_rt = float(
        np.max(
            np.abs(
                scaler.inverse(normed[: min(normed.shape[0], 64)])
                - enc.states[: min(normed.shape[0], 64)]
            )
        )
    )

    _, pool_tgt = kit_windows(normed[:n_tr], lc, lh)
    bench_rng = np.random.default_rng(int(seed) + 1)
    ens_norm = bootstrap_horizon(pool_tgt, m, bench_rng)
    ens_raw = scaler.inverse(ens_norm)
    anchor_close = float(stream[n_tr + lc - 1, 3])
    anchor_vol = float(stream[n_tr + lc - 1, 4])
    anchor_ema = float(enc.vol_ema[n_tr + lc - 1])
    ens_dec = decode_kit_states(
        ens_raw,
        anchor_close,
        ema_halflife=float(hl),
        ema0=anchor_ema,
        volume0=anchor_vol,
    )
    realized_states = enc.states[n_tr + lc : n_tr + lc + lh]
    scores = path_score_report(ens_raw, realized_states)
    consistency = candle_consistency_report(ens_dec.candles)
    ctx_consistency = candle_consistency_report(stream)

    return {
        "synthetic_roundtrip_max_abs_err": roundtrip_err,
        "synthetic_scaler_roundtrip_max_abs_err": scaler_rt,
        "synthetic_violation_rate": float(consistency["violation_rate"]),
        "synthetic_high_violation_rate": float(consistency["high_violation_rate"]),
        "synthetic_low_violation_rate": float(consistency["low_violation_rate"]),
        "synthetic_volume_violation_rate": float(consistency["volume_violation_rate"]),
        "synthetic_source_violation_rate": float(ctx_consistency["violation_rate"]),
        "synthetic_energy_score": float(scores["energy_score"]),
        "synthetic_crps_marginal_mean": float(scores["crps_marginal_mean"]),
        "synthetic_crps_terminal_ret": float(scores["crps_terminal_ret"]),
        "synthetic_pinball_50_terminal_ret": float(scores["pinball_50_terminal_ret"]),
        "synthetic_coverage_ret": float(scores["coverage_ret"]),
        "synthetic_width_ret": float(scores["width_ret"]),
        "synthetic_n_samples": float(m),
        "synthetic_n_windows": float(pool_tgt.shape[0]),
        "synthetic_context": float(lc),
        "synthetic_horizon": float(lh),
        "synthetic_dgp": "fixture",
        "synthetic_claim": "research_metric_only",
        "synthetic_synthetic": "two_regime_candle_stream_seeded",
    }
