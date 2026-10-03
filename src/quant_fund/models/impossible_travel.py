"""Impossible-travel detection (defensive) — wave 286.

Login at (lat,lon,t) flagged when required speed between consecutive
events exceeds max_travel_kmh.
"""

import numpy as np

_SEED = 20261231 + 803


def _hav(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def flag(events: list[tuple[float, float, float]], vmax: float = 900.0) -> list[int]:
    # events: (lat, lon, unix_hour); returns indices of impossible hops
    out = []
    for i in range(1, len(events)):
        lat1, lon1, t1 = events[i - 1]
        lat2, lon2, t2 = events[i]
        dt = max(t2 - t1, 1e-6)
        if _hav(lat1, lon1, lat2, lon2) / dt > vmax:
            out.append(i)
    return out


def bench_impossible_travel(seed: int = _SEED) -> dict[str, float]:
    # london -> tokyo in 1h is impossible; london -> paris in 2h fine
    legit = [(51.5, -0.12, 0.0), (48.85, 2.35, 2.0), (48.85, 2.35, 30.0)]
    fraud = [(51.5, -0.12, 0.0), (35.68, 139.69, 1.0)]
    return {
        "synthetic_travel_flag": float(flag(fraud) == [1]),
        "synthetic_travel_clear": float(flag(legit) == []),
    }
