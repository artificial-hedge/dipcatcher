"""Occupancy-grid lidar ray casting (wave 283) (SYNTHETIC).

DDA march from the sensor pose until a hit cell or max range; verified by
placing walls at known offsets and checking measured range.
"""

import numpy as np

_SEED = 20261231 + 784


def cast(grid: np.ndarray, x: float, y: float, ang: float, max_r: float = 20.0) -> float:
    dx, dy = np.cos(ang), np.sin(ang)
    for i in range(1, int(max_r * 10)):
        r = i / 10.0
        cx, cy = int(x + dx * r), int(y + dy * r)
        if cx < 0 or cy < 0 or cx >= grid.shape[0] or cy >= grid.shape[1] or grid[cx, cy]:
            return r
    return max_r


def bench_ray_lidar(seed: int = _SEED) -> dict[str, float]:
    grid = np.zeros((40, 40), dtype=int)
    grid[15, :] = 1  # wall plane x=15
    ok = 0
    for ang in (-0.3, 0.0, 0.3):
        r = cast(grid, 5.0, 20.0, ang)
        want = 10.0 / np.cos(ang)
        ok += int(abs(r - want) < 0.25)
    grid2 = np.zeros((40, 40), dtype=int)
    ok += int(cast(grid2, 5.0, 5.0, 0.0) == 20.0)  # no walls -> max range
    return {"synthetic_ray_lidar": float(ok == 4)}
