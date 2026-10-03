"""LOD selection (wave 293).

Screen-space-error metric: render the COARSEST level whose projected
geometric error rho = err_world * K / dist stays below threshold.
Monotone in distance, verified at boundary distances.
"""

_SEED = 20261231 + 845


def lod_level(errors: list[float], dist: float, k: float = 256.0, thresh: float = 1.0) -> int:
    best = 0
    for lv, e in enumerate(errors):
        if e * k / max(dist, 1e-6) <= thresh:
            best = lv
    return best


def bench_lod_select(seed: int = _SEED) -> dict[str, float]:
    errors = [0.0, 0.01, 0.1, 1.0]
    # near: only finest passes (0.01*256/1 = 2.56 > 1)
    ok = int(lod_level(errors, 1.0) == 0)
    # mid: level 1 acceptable
    ok += int(lod_level(errors, 10.0) == 1)
    # far: coarsest
    ok += int(lod_level(errors, 300.0) == 3)
    # monotone nondecreasing with distance
    ok += int(
        all(
            lod_level(errors, d1) <= lod_level(errors, d2)
            for d1, d2 in [(1, 5), (5, 50), (50, 500)]
        )
    )
    return {"synthetic_lod": float(ok == 4)}
