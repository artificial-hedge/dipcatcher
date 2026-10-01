"""markout — post-execution adverse-selection curve (markout analysis).

For each visible EXECUTION on a LOBSTER tape, the markout at horizon ``h``
is the signed mid move over the following ``h`` seconds, expressed in ticks:

- buy-aggressor fill:  ``mid(t+h) - mid(t)``
- sell-aggressor fill: ``-(mid(t+h) - mid(t))``

so a positive markout means the mid kept moving in the aggressor's direction
after the fill — the textbook signature of informed (toxic) flow and the
maker's adverse-selection cost. Horizons are sampled on the book-event time
grid via ``searchsorted`` (first book row at or after ``t + h``).

LOBSTER conventions (same as the replay lane): message direction records the
*resting* side, so the aggressor sign is ``-direction``; raw prices are
``price * 10_000``, hence ``raw / 100`` gives cent (tick) units. The
orderbook row paired with a message row is the book state *after* that event,
which is the ``mid(t)`` anchor.

Sim arms repeat the identical measurement inside the zero-intelligence LOB
(``zi_lob_simulator``): the append-only ``sim.trades`` list supplies
executions and ``sim.mid`` sampled after every event supplies the mid series
(converted to ticks as ``mid / cfg.tick``). Arms:

- ``iid`` — flat ZI flow (uninformed Poisson MOs; mechanical-impact-only
  baseline),
- ``markov_regime`` — two-state Markov flow on the MO clock
  (``MarkovRegimeFlow``, calm ↔ bursty buy-tilted regimes),
- ``split`` — metaorder-splitting flow (``SplitFlow``): episodes of
  ``k`` same-direction MOs with power-law episode lengths — the
  Lillo-Mike-Farmer mechanism for long memory in trade sign.

``toxicity_half_life_s`` is the first sampled horizon whose *pooled signed*
mean markout exceeds its own noise floor, where the floor is the standard
error of the mean ``std / sqrt(n)`` at that horizon (``None`` when no sampled
horizon clears it — e.g. genuinely uninformed flow).

Honesty: the real tape is genuine LOBSTER sample data (``REAL``); every sim
arm is labeled ``SYNTHETIC`` correctness/comparison evidence, never market
evidence. The receipt carries no forbidden headline-metric keys, and
``receipt_sha256`` seals the payload under ``canonical_json_bytes``.

References:
- Cont, Cucuringu, Zhang (2014). Impact of order flow on prices. — signed
  markout / adverse-selection measurement on event data.
- Lillo, Mike, Farmer (2005). *PRE* 71:066122 and Tóth et al. (2015) — order
  splitting as the origin of long memory in the sign of market orders.
- LOBSTER output structure: https://data.lobsterdata.com/info/DataStructure.php
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

Array = NDArray[np.float64]

MARKOUT_SCHEMA = "markout.v1"
MARKOUT_KIND = "markout"
MARKOUT_CLAIM = "markout_curve_measured"

#: Horizon grid (seconds) for the markout curve.
HORIZONS_S: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)

#: Arm-vs-real divergence gate: |arm - real| at this horizon, as a fraction of
#: |real|, above which the arm is flagged divergent.
REFERENCE_HORIZON_S = 5.0
DIVERGENCE_FRACTION = 0.5

# LOBSTER message event types (subset used here).
EXECUTION = 4  # execution of a visible limit order


# ---------------------------------------------------------------------------
# LOBSTER tape parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LobsterEvent:
    """One LOBSTER message row."""

    time_s: float
    event_type: int
    order_id: int
    size: int
    price: int  # price * 10_000
    direction: int  # +1 buy-side order, -1 sell-side order (resting side)


def parse_messages(path: Path) -> Iterator[LobsterEvent]:
    """Yield LOBSTER message-file events (time, type, id, size, price, dir)."""
    with path.open() as f:
        for row in csv.reader(f):
            if not row:
                continue
            yield LobsterEvent(
                time_s=float(row[0]),
                event_type=int(row[1]),
                order_id=int(row[2]),
                size=int(row[3]),
                price=int(row[4]),
                direction=int(row[5]),
            )


def parse_orderbook_row(row: list[str]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """LOBSTER orderbook CSV row → (asks, bids) top-N (price, size).

    Layout is ``[ask_p, ask_sz, bid_p, bid_sz]`` repeated per level; unoccupied
    levels carry a dummy price with size 0 and are dropped.
    """
    vals = [float(x) for x in row]
    asks: list[tuple[int, int]] = []
    bids: list[tuple[int, int]] = []
    for lvl in range(len(vals) // 4):
        ap, asz, bp, bsz = vals[4 * lvl : 4 * lvl + 4]
        if asz > 0:
            asks.append((int(ap), int(asz)))
        if bsz > 0:
            bids.append((int(bp), int(bsz)))
    return asks, bids


def _top_mid_ticks(asks: list[tuple[int, int]], bids: list[tuple[int, int]]) -> float | None:
    """Top-of-book mid in ticks (raw LOBSTER price units / 100), or None."""
    if not asks or not bids:
        return None
    return (asks[0][0] + bids[0][0]) / 200.0


# ---------------------------------------------------------------------------
# SplitFlow — metaorder-splitting MO flow (LMF mechanism)
# ---------------------------------------------------------------------------


class SplitFlow:
    """Episodic same-direction MO flow on the market-order clock.

    With probability ``p_start`` per MO event (while idle) a metaorder
    episode begins: a direction (50/50) and a length ``k`` drawn from a
    truncated power law ``P(k) ∝ k**-size_tail`` over ``[k_min, k_max]``.
    For the next ``k`` MO events the flow returns the episode's
    ``RegimeState`` — ``intensity_mult`` on the MO rate and ``p_buy`` pinned
    to the episode direction — then goes idle again. This is the
    order-splitting mechanism that produces long memory in trade sign
    (Lillo-Mike-Farmer 2005; Tóth et al. 2015), i.e. flow that *looks*
    informed on a markout curve because one parent's children keep walking
    the mid the same way. Seeded and deterministic; the
    ``ZILobSimulator`` flow interface is ``current()`` / ``advance()``.
    """

    def __init__(
        self,
        *,
        p_start: float,
        size_tail: float,
        k_min: int,
        k_max: int,
        intensity_mult: float,
        seed: int = 0,
    ) -> None:
        p = float(p_start)
        if not math.isfinite(p) or p < 0.0 or p > 1.0:
            raise ValueError(f"p_start must be a probability in [0, 1], got {p_start!r}")
        tail = float(size_tail)
        if not math.isfinite(tail) or tail <= 0.0:
            raise ValueError(f"size_tail must be positive and finite, got {size_tail!r}")
        if isinstance(k_min, bool) or isinstance(k_max, bool) or int(k_min) < 1:
            raise ValueError(f"k_min must be an int >= 1, got {k_min!r}")
        if int(k_max) < int(k_min):
            raise ValueError(f"k_max must be >= k_min, got {k_max!r}")
        mult = float(intensity_mult)
        if not math.isfinite(mult) or mult <= 0.0:
            raise ValueError(f"intensity_mult must be positive and finite, got {intensity_mult!r}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an int, got {seed!r}")
        self._p_start = p
        self._size_tail = tail
        self._k_min = int(k_min)
        self._k_max = int(k_max)
        self._intensity_mult = mult
        self._rng = np.random.default_rng(seed)
        ks = np.arange(self._k_min, self._k_max + 1, dtype=np.float64)
        weights = ks ** (-tail)
        self._len_cdf = np.cumsum(weights / weights.sum())
        self._len_cdf[-1] = 1.0
        self._remaining = 0
        self._side = 0  # +1 buy metaorder episode, -1 sell, 0 idle
        self.n_mo = 0
        self.n_episodes = 0
        self.episode_lengths: list[int] = []

    def current(self) -> RegimeState:
        """Flow parameters for the next MO event (sim reads before draw)."""
        if self._remaining > 0:
            if self._side > 0:
                return RegimeState("split_buy", self._intensity_mult, 1.0)
            return RegimeState("split_sell", self._intensity_mult, 0.0)
        return RegimeState("idle", 1.0, 0.5)

    def advance(self) -> None:
        """One MO-clock tick: consume episode length, maybe start a new one."""
        self.n_mo += 1
        if self._remaining > 0:
            self._remaining -= 1
            if self._remaining > 0:
                return
            self._side = 0
        if float(self._rng.random()) < self._p_start:
            k = self._k_min + int(
                np.searchsorted(self._len_cdf, float(self._rng.random()), side="left")
            )
            self._remaining = min(k, self._k_max)
            self._side = 1 if float(self._rng.random()) < 0.5 else -1
            self.n_episodes += 1
            self.episode_lengths.append(self._remaining)

    @property
    def in_episode(self) -> bool:
        return self._remaining > 0


# ---------------------------------------------------------------------------
# Markout measurement
# ---------------------------------------------------------------------------


def _per_horizon_stats(samples: list[Array]) -> dict[str, Any]:
    """mean / median / n over each horizon's finite markout values."""
    means: list[float | None] = []
    medians: list[float | None] = []
    ns: list[int] = []
    for vals in samples:
        finite = vals[np.isfinite(vals)]
        n = int(finite.size)
        ns.append(n)
        if n == 0:
            means.append(None)
            medians.append(None)
        else:
            means.append(float(finite.mean()))
            medians.append(float(np.median(finite)))
    return {"mean": means, "median": medians, "n": ns}


def _sem_floors(samples: list[Array]) -> list[float | None]:
    """Per-horizon noise floor: std / sqrt(n) of that horizon's markouts."""
    floors: list[float | None] = []
    for vals in samples:
        finite = vals[np.isfinite(vals)]
        n = int(finite.size)
        if n < 2:
            floors.append(None)
            continue
        floors.append(float(finite.std(ddof=1) / math.sqrt(n)))
    return floors


def _toxicity_half_life(
    horizons: Sequence[float], means: list[float | None], floors: list[float | None]
) -> float | None:
    """First horizon where the pooled signed mean clears its own SEM floor."""
    for h, mean, floor in zip(horizons, means, floors, strict=True):
        if mean is None or floor is None:
            continue
        if mean > floor:
            return float(h)
    return None


def markout_curve(
    sample_times: Array,
    sample_mid_ticks: Array,
    exec_sample_idx: Array,
    exec_signs: Array,
    horizons: Sequence[float] = HORIZONS_S,
) -> dict[str, Any]:
    """Signed markout curve over a (time, mid) sample series.

    ``exec_sample_idx[k]`` is the sample index at which execution ``k``
    occurred (its ``mid(t)`` anchor) and ``exec_signs[k]`` is +1 for a
    buy-aggressor fill, -1 for a sell-aggressor fill. For each horizon the
    markout of execution ``k`` is
    ``exec_signs[k] * (mid[idx(t_k + h)] - mid[exec_sample_idx[k]])`` where
    ``idx(t)`` is the first sample at or after ``t``. Executions whose anchor
    or horizon mid is undefined, or whose horizon overruns the series,
    contribute no value at that horizon (``n`` counts finite values only).
    """
    times = np.asarray(sample_times, dtype=np.float64)
    mids = np.asarray(sample_mid_ticks, dtype=np.float64)
    idx = np.asarray(exec_sample_idx, dtype=np.int64)
    signs = np.asarray(exec_signs, dtype=np.float64)
    if times.ndim != 1 or mids.shape != times.shape:
        raise ValueError("sample_times and sample_mid_ticks must be equal-length 1-D arrays")
    if idx.shape != signs.shape:
        raise ValueError("exec_sample_idx and exec_signs must be equal-length")
    if times.size == 0:
        raise ValueError("empty sample series")
    if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= times.size):
        raise ValueError("exec_sample_idx out of range")
    hs = [float(h) for h in horizons]
    for h in hs:
        if not math.isfinite(h) or h <= 0.0:
            raise ValueError(f"horizons must be positive and finite, got {h!r}")
    n_samples = times.size
    exec_t = times[idx] if idx.size else np.zeros(0)

    per_side: dict[str, list[Array]] = {
        "buy": [np.empty(0) for _ in hs],
        "sell": [np.empty(0) for _ in hs],
    }
    pooled: list[Array] = [np.empty(0) for _ in hs]
    for j, h in enumerate(hs):
        if idx.size == 0:
            per_side["buy"][j] = np.empty(0)
            per_side["sell"][j] = np.empty(0)
            pooled[j] = np.empty(0)
            continue
        j_h = np.searchsorted(times, exec_t + h, side="left")
        ok = j_h < n_samples
        anchor = np.isfinite(mids[idx])
        target = np.zeros(idx.size, dtype=bool)
        target[ok] = np.isfinite(mids[j_h[ok]])
        valid = ok & anchor & target
        marks = np.full(idx.size, np.nan)
        marks[valid] = signs[valid] * (mids[j_h[valid]] - mids[idx[valid]])
        pooled[j] = marks
        per_side["buy"][j] = marks[signs > 0]
        per_side["sell"][j] = marks[signs < 0]

    sides = {side: _per_horizon_stats(per_side[side]) for side in ("buy", "sell")}
    signed = _per_horizon_stats(pooled)
    floors = _sem_floors(pooled)
    signed["noise_floor_sem"] = floors
    return {
        "horizons_s": list(hs),
        "n_executions": int(idx.size),
        "sides": sides,
        "signed": signed,
        "toxicity_half_life_s": _toxicity_half_life(hs, signed["mean"], floors),
    }


def tape_markout_curve(
    message_path: Path,
    orderbook_path: Path,
    horizons: Sequence[float] = HORIZONS_S,
) -> dict[str, Any]:
    """Markout curve over a real LOBSTER (message, orderbook) CSV pair.

    Rows are zipped in lockstep: every message row carries the book state
    after its event, so an EXECUTION row's own orderbook row is its mid
    anchor.
    """
    times: list[float] = []
    mids: list[float] = []
    exec_idx: list[int] = []
    exec_signs: list[float] = []
    with orderbook_path.open() as f_ob:
        for i, (ev, ob_row) in enumerate(
            zip(parse_messages(message_path), csv.reader(f_ob), strict=True)
        ):
            asks, bids = parse_orderbook_row(ob_row)
            times.append(ev.time_s)
            mid = _top_mid_ticks(asks, bids)
            mids.append(float("nan") if mid is None else mid)
            if ev.event_type == EXECUTION:
                exec_idx.append(i)
                exec_signs.append(float(-ev.direction))  # direction = resting side
    curve = markout_curve(
        np.asarray(times),
        np.asarray(mids),
        np.asarray(exec_idx, dtype=np.int64),
        np.asarray(exec_signs),
        horizons,
    )
    curve["n_book_events"] = len(times)
    return curve


def sim_markout_curve(
    sim: ZILobSimulator,
    *,
    sim_seconds: float,
    horizons: Sequence[float] = HORIZONS_S,
) -> dict[str, Any]:
    """Identical markout measurement on a ZI-LOB event path.

    The mid series is sampled after every event (``sim.mid / cfg.tick``, NaN
    while a side is empty); executions are the append-only ``sim.trades``
    entries, attributed to the sample index of the step that emitted them.
    """
    horizon = float(sim_seconds)
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError(f"sim_seconds must be positive and finite, got {sim_seconds!r}")
    tick = sim.cfg.tick
    times: list[float] = []
    mids: list[float] = []
    exec_idx: list[int] = []
    exec_signs: list[float] = []
    n_before = 0
    while sim.t < horizon:
        sim.step()
        mid = sim.mid
        times.append(sim.t)
        mids.append(float("nan") if mid is None else mid / tick)
        if len(sim.trades) > n_before:
            row = len(times) - 1
            for tr in sim.trades[n_before:]:
                exec_idx.append(row)
                exec_signs.append(1.0 if tr.aggressor == "buy" else -1.0)
            n_before = len(sim.trades)
    curve = markout_curve(
        np.asarray(times),
        np.asarray(mids),
        np.asarray(exec_idx, dtype=np.int64),
        np.asarray(exec_signs),
        horizons,
    )
    curve["n_events"] = sim.n_events
    curve["sim_seconds"] = horizon
    return curve


# ---------------------------------------------------------------------------
# Bench + sealed receipt
# ---------------------------------------------------------------------------


def _divergences(
    real: dict[str, Any],
    arms: dict[str, dict[str, Any]],
    horizons: Sequence[float],
    *,
    ref_horizon: float = REFERENCE_HORIZON_S,
    fraction: float = DIVERGENCE_FRACTION,
) -> list[dict[str, Any]]:
    """Per-arm |arm - real| at the reference horizon vs ``fraction * |real|``."""
    try:
        j = [float(h) for h in horizons].index(ref_horizon)
    except ValueError:
        j = -1
    real_ref = real["signed"]["mean"][j] if j >= 0 else None
    threshold = fraction * abs(real_ref) if isinstance(real_ref, float) else None
    out: list[dict[str, Any]] = []
    for name, arm in arms.items():
        arm_ref = arm["signed"]["mean"][j] if j >= 0 else None
        gap = (
            abs(arm_ref - real_ref)
            if isinstance(arm_ref, float) and isinstance(real_ref, float)
            else None
        )
        out.append(
            {
                "arm": name,
                "horizon_s": ref_horizon,
                "real_mean_ticks": real_ref,
                "arm_mean_ticks": arm_ref,
                "abs_gap_ticks": gap,
                "threshold_ticks": threshold,
                "diverges": bool(gap is not None and threshold is not None and gap > threshold),
            }
        )
    return out


def _tape_paths(tape_dir: Path, ticker: str) -> tuple[Path, Path]:
    msg = sorted(tape_dir.glob(f"{ticker}_*_message_*.csv"))
    ob = sorted(tape_dir.glob(f"{ticker}_*_orderbook_*.csv"))
    if not msg or not ob:
        raise FileNotFoundError(
            f"no LOBSTER {ticker}_*_message_*_10.csv / {ticker}_*_orderbook_*_10.csv "
            f"pair under {tape_dir}"
        )
    return msg[0], ob[0]


def markout_bench(
    tape_dir: Path,
    ticker: str = "AMZN",
    seed: int = 7,
    *,
    sim_seconds: float = 45_000.0,
    horizons: Sequence[float] = HORIZONS_S,
) -> dict[str, Any]:
    """Markout bench: real LOBSTER tape vs the three sim arms, sealed receipt.

    ``sim_seconds`` sizes each synthetic arm's event path (real tape covers
    the full session). Returns the sealed ``markout.v1`` receipt dict —
    ``data_label='MIXED'``: real tape (LOBSTER AMZN 2012-06-21 sample) plus
    SYNTHETIC ZI arms; research-only, no live-trading claim.
    """
    tape_dir = Path(tape_dir)
    msg, ob = _tape_paths(tape_dir, ticker)
    hs = [float(h) for h in horizons]

    real = tape_markout_curve(msg, ob, hs)
    real["tape_files"] = [msg.name, ob.name]
    real["tape_sha256"] = hash_bytes(msg.read_bytes())
    real["data_label"] = "REAL_LOBSTER_SAMPLE"

    arms: dict[str, dict[str, Any]] = {}

    iid_sim = ZILobSimulator(ZILobConfig(seed=seed))
    arms["iid"] = sim_markout_curve(iid_sim, sim_seconds=sim_seconds, horizons=hs)
    arms["iid"]["flow"] = "iid_zi"
    arms["iid"]["seed"] = seed
    arms["iid"]["data_label"] = "SYNTHETIC"

    regime_flow = MarkovRegimeFlow(
        states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
        stay_probs=(0.995, 0.985),
        seed=seed + 1,
    )
    regime_sim = ZILobSimulator(ZILobConfig(seed=seed + 1), flow=regime_flow)
    arms["markov_regime"] = sim_markout_curve(regime_sim, sim_seconds=sim_seconds, horizons=hs)
    arms["markov_regime"]["flow"] = "markov_regime_calm_bursty"
    arms["markov_regime"]["seed"] = seed + 1
    arms["markov_regime"]["data_label"] = "SYNTHETIC"

    split_flow = SplitFlow(
        p_start=0.10, size_tail=1.2, k_min=10, k_max=600, intensity_mult=3.0, seed=seed + 2
    )
    split_sim = ZILobSimulator(ZILobConfig(seed=seed + 2), flow=split_flow)
    arms["split"] = sim_markout_curve(split_sim, sim_seconds=sim_seconds, horizons=hs)
    arms["split"]["flow"] = "split_metaorder_powerlaw"
    arms["split"]["seed"] = seed + 2
    arms["split"]["n_episodes"] = split_flow.n_episodes
    arms["split"]["data_label"] = "SYNTHETIC"

    payload: dict[str, Any] = {
        "kind": MARKOUT_KIND,
        "schema": MARKOUT_SCHEMA,
        "ticker": ticker,
        "real": real,
        "sim_arms": arms,
        "divergences": _divergences(real, arms, hs),
        "claim": MARKOUT_CLAIM,
        "interpretation": (
            "Post-execution adverse-selection curve: signed mid move in ticks "
            "at h in {0.05..30}s after each visible EXECUTION, sign-adjusted so "
            "positive = drift in the aggressor's direction (toxic flow). Real "
            "arm is the official LOBSTER AMZN 2012-06-21 level-10 sample; sim "
            "arms are labeled SYNTHETIC ZI-LOB variants (iid baseline, Markov "
            "calm/bursty regime flow, power-law metaorder splitting flow) under "
            "the identical measurement. 'divergences' flags arms whose 5s "
            "markout differs from the tape's by more than half the tape value. "
            "toxicity_half_life_s is the first horizon whose pooled signed mean "
            "exceeds its own standard-error floor (std/sqrt(n)); None when no "
            "horizon clears it. Research-only; no live-trading claim."
        ),
        "git_revision": git_revision(),
        "data_label": "MIXED",
        "research_only": True,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


__all__ = [
    "DIVERGENCE_FRACTION",
    "EXECUTION",
    "HORIZONS_S",
    "MARKOUT_CLAIM",
    "MARKOUT_KIND",
    "MARKOUT_SCHEMA",
    "REFERENCE_HORIZON_S",
    "LobsterEvent",
    "SplitFlow",
    "markout_bench",
    "markout_curve",
    "parse_messages",
    "parse_orderbook_row",
    "sim_markout_curve",
    "tape_markout_curve",
]
