"""Tick-rule trade signing and imbalance bars.

Trade signs come from the tick rule (sign of the price change, zeros carrying
the previous non-zero sign). Imbalance bars close when the *cumulative signed
flow* since the bar open reaches ±threshold — i.e. when the bar has absorbed
a fixed amount of (directional) information rather than a fixed amount of
activity. Two flavours are provided:

- static threshold: ``*_imbalance_bar_ids(signs, values, threshold)``;
- adaptive threshold (López de Prado): expected bar length × expected
  per-trade signed imbalance, estimated with EMAs, so thresholds track
  changing order-flow persistence.

The scanning loop is O(trades) and deliberately plain: correctness and
readability beat vectorisation for stateful resets.

Honesty: signing and bar ids are deterministic functions of the input tape;
synthetic fixtures only.

References:
- López de Prado, M. (2018). *Advances in Financial Machine Learning*,
  ch. 2 — tick/volume/dollar imbalance bars and the EMA-adaptive threshold.
- Cont, R., Cucuringu, M., Zhang, C. (2023). Cross-impact of order flow
  imbalance — signed-flow bar conventions.
- Easley, D., López de Prado, M., O'Hara, M. (2012). The volume clock:
  informed trading in the presence of toxic flow.

Composition: pure numpy; deterministic; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def tick_rule_signs(prices: FloatArray) -> IntArray:
    """Trade signs from price changes; zeros inherit the previous sign."""
    prices = np.asarray(prices, dtype=np.float64)
    if prices.ndim != 1:
        raise ValueError("prices must be one-dimensional")
    delta = np.diff(prices, prepend=prices[0])
    signs = np.sign(delta).astype(np.int64)
    out = np.empty_like(signs)
    last = 1
    for i in range(len(signs)):
        s = int(signs[i])
        if s == 0:
            s = last
        else:
            last = s
        out[i] = s
    return out


def imbalance_bar_ids(signs: IntArray, values: FloatArray, threshold: float) -> IntArray:
    """Generic imbalance bars: close when |Σ sign·value| since open ≥ threshold.

    ``values`` is typically sizes (volume imbalance) or dollar amounts
    (dollar imbalance). The final open bar is closed at the last trade.
    """
    signs = np.asarray(signs, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if signs.shape != values.shape:
        raise ValueError("signs and values must have the same shape")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    n = len(signs)
    ids = np.zeros(n, dtype=np.int64)
    if n == 0:
        return ids
    bar = 0
    run = 0.0
    for i in range(n):
        run += float(signs[i]) * float(values[i])
        ids[i] = bar
        if abs(run) >= threshold:
            bar += 1
            run = 0.0
    return ids


def tick_imbalance_bar_ids(signs: IntArray, threshold: float) -> IntArray:
    """Tick imbalance bars: each trade counts ±1."""
    signs = np.asarray(signs, dtype=np.int64)
    return imbalance_bar_ids(signs, np.ones(len(signs), dtype=np.float64), threshold)


def volume_imbalance_bar_ids(signs: IntArray, sizes: FloatArray, threshold: float) -> IntArray:
    """Volume imbalance bars: each trade counts ±size."""
    return imbalance_bar_ids(signs, sizes, threshold)


def dollar_imbalance_bar_ids(signs: IntArray, dollars: FloatArray, threshold: float) -> IntArray:
    """Dollar imbalance bars: each trade counts ±dollar amount."""
    return imbalance_bar_ids(signs, dollars, threshold)


def adaptive_imbalance_bar_ids(
    signs: IntArray,
    values: FloatArray,
    *,
    warm_up: int = 20,
    min_threshold: float = 1e-9,
    ema_halflife: float = 20.0,
) -> IntArray:
    """EMA-adaptive imbalance bars (López de Prado, AFML ch. 2).

    Threshold_t = max(min_threshold, EMA_T(t) × |2·p̂_t − 1| × v̄_t) where p̂
    is the EMA up-probability of signed flow and v̄ the EMA mean |value| per
    trade. Expectations update only at bar closes, as in the reference.
    """
    signs = np.asarray(signs, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if signs.shape != values.shape:
        raise ValueError("signs and values must have the same shape")
    n = len(signs)
    ids = np.zeros(n, dtype=np.int64)
    if n == 0:
        return ids
    lam = float(np.log(2.0) / ema_halflife)
    p_ema = 0.5
    abs_imb_ema = 0.0
    t_ema = float(warm_up)
    bar = 0
    run = 0.0
    bar_start = 0
    for i in range(n):
        run += float(signs[i]) * float(values[i])
        ids[i] = bar
        threshold = max(min_threshold, t_ema * abs(2.0 * p_ema - 1.0) * max(abs_imb_ema, 1e-12))
        if abs(run) >= threshold and i - bar_start + 1 >= warm_up:
            bar_len = float(i - bar_start + 1)
            up_frac = float(np.mean(signs[bar_start : i + 1] > 0))
            mean_abs_val = float(np.mean(np.abs(values[bar_start : i + 1])))
            p_ema = (1.0 - lam) * p_ema + lam * up_frac
            abs_imb_ema = (1.0 - lam) * abs_imb_ema + lam * mean_abs_val
            t_ema = (1.0 - lam) * t_ema + lam * bar_len
            bar += 1
            run = 0.0
            bar_start = i + 1
    return ids


def signed_flow_imbalance(signs: IntArray, sizes: FloatArray) -> FloatArray:
    """Cumulative signed volume from the tape start (stateless diagnostic)."""
    signs = np.asarray(signs, dtype=np.int64)
    sizes = np.asarray(sizes, dtype=np.float64)
    if signs.shape != sizes.shape:
        raise ValueError("signs and sizes must have the same shape")
    return np.cumsum(signs.astype(np.float64) * sizes)
