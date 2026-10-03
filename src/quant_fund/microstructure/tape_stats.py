"""tape_stats — empirical joint-distribution tables from committed receipts.

Reads the sealed real-tape receipts (``round_lot`` size histogram,
``event_matrix`` arrival-rate card, ``spread_dynamics`` occupancy) and
returns a :class:`TapeStats` record the empirical-flow lane consumes.
Every loader is fail-closed: a missing receipt, missing key, or
malformed bucket raises ``ValueError`` — no silent defaults.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SIZE_BUCKET = re.compile(r"^(\d+)-(\d+|inf)$")


def _load_receipt(root: Path, name: str) -> dict[str, Any]:
    path = root / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"missing tape receipt {path}")
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable tape receipt {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"tape receipt {path} is not a JSON object")
    return data


def _size_pmf(hist: dict[str, int]) -> tuple[tuple[int, float], ...]:
    """Bucket counts → size pmf over bucket midpoints (inf → 2× lower edge)."""
    out: list[tuple[int, float]] = []
    for bucket, count in hist.items():
        m = _SIZE_BUCKET.match(str(bucket))
        if m is None:
            raise ValueError(f"unparseable size bucket {bucket!r}")
        lo_s, hi_s = m.groups()
        lo, hi = int(lo_s), (int(hi_s) if hi_s != "inf" else 2 * int(lo_s))
        if not isinstance(count, (int, float)) or count < 0:
            raise ValueError(f"size bucket {bucket!r} has bad count {count!r}")
        mid = max(1, int(round(0.5 * (lo + hi))))
        if count > 0:
            out.append((mid, float(count)))
    if not out:
        raise ValueError("size histogram is empty")
    return tuple(out)


@dataclass(frozen=True)
class TapeStats:
    """Empirical tables resampled by the empirical-flow config builder.

    ``size_pmf`` comes from the round-lot execution-size histogram
    (bucket midpoints, counts as weights). ``events_per_s`` is the
    event_matrix rate card. ``spread_occupancy`` is the tick-bucket
    occupancy of the inside spread.
    """

    ticker: str
    size_pmf: tuple[tuple[int, float], ...]
    events_per_s: dict[str, float]
    spread_occupancy: dict[str, float]
    mean_spread_ticks: float

    def rate(self, key: str) -> float:
        v = self.events_per_s.get(key)
        if v is None:
            raise ValueError(f"event_matrix receipt lacks rate {key!r}")
        v = float(v)
        if not math.isfinite(v) or v < 0.0:
            raise ValueError(f"event_matrix rate {key!r} not a non-negative finite")
        return v


def load_tape_stats(receipts_root: str | Path, ticker: str = "amzn") -> TapeStats:
    """Load the empirical tables for ``ticker`` from committed receipts."""
    root = Path(receipts_root)
    if not root.is_dir():
        raise ValueError(f"receipts root {root} is not a directory")
    tag = str(ticker).lower()
    round_lot = _load_receipt(root, f"round_lot_{tag}")
    event_matrix = _load_receipt(root, f"event_matrix_{tag}")
    spread = _load_receipt(root, f"spread_dynamics_{tag}")
    real_rl = round_lot.get("real")
    real_em = event_matrix.get("real")
    real_sd = spread.get("real")
    for name, sec in (
        ("round_lot", real_rl),
        ("event_matrix", real_em),
        ("spread_dynamics", real_sd),
    ):
        if not isinstance(sec, dict):
            raise ValueError(f"{name} receipt lacks a 'real' section")
    assert isinstance(real_rl, dict) and isinstance(real_em, dict) and isinstance(real_sd, dict)
    hist = real_rl.get("size_hist")
    if not isinstance(hist, dict):
        raise ValueError("round_lot receipt lacks real.size_hist")
    eps = real_em.get("events_per_s")
    if not isinstance(eps, dict):
        raise ValueError("event_matrix receipt lacks real.events_per_s")
    occ = real_sd.get("spread_occupancy")
    if not isinstance(occ, dict):
        raise ValueError("spread_dynamics receipt lacks real.spread_occupancy")
    mean_spread = float(_nonneg(real_sd.get("mean_spread_ticks"), "mean_spread_ticks"))
    return TapeStats(
        ticker=tag,
        size_pmf=_size_pmf(hist),
        events_per_s={str(k): float(v) for k, v in eps.items()},
        spread_occupancy={str(k): float(v) for k, v in occ.items()},
        mean_spread_ticks=mean_spread,
    )


def _nonneg(x: object, name: str) -> float:
    if not isinstance(x, (int, float)) or not math.isfinite(float(x)) or float(x) < 0.0:
        raise ValueError(f"{name} must be a non-negative finite, got {x!r}")
    return float(x)


def scaled_size_pmf(
    pmf: tuple[tuple[int, float], ...], scale: float
) -> tuple[tuple[int, float], ...]:
    """Rescale share sizes to sim units (min size 1, weights preserved)."""
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be positive and finite, got {scale!r}")
    return tuple((max(1, int(round(s * scale))), w) for s, w in pmf)
