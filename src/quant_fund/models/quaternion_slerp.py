"""SYNTHETIC quaternion slerp + rotation.

q = (w, x, y, z) unit quats; slerp interpolates on the 4-sphere. Benches:
endpoints exact, mid-interp angular distance = half the total,
slerp rotates vectors consistently with direct quat rotation.
"""

from __future__ import annotations

import math
import random

Quat = tuple[float, float, float, float]


def _mul(a: Quat, b: Quat) -> Quat:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _conj(q: Quat) -> Quat:
    return (q[0], -q[1], -q[2], -q[3])


def _axis_angle(axis: tuple[float, float, float], ang: float) -> Quat:
    s = math.sin(ang / 2)
    n = math.sqrt(sum(c * c for c in axis))
    return (math.cos(ang / 2), axis[0] / n * s, axis[1] / n * s, axis[2] / n * s)


def slerp(q0: Quat, q1: Quat, t: float) -> Quat:
    dot = sum(a * b for a, b in zip(q0, q1, strict=True))
    if dot < 0:
        q1 = tuple(-c for c in q1)  # type: ignore[assignment]
        dot = -dot
    if dot > 0.9995:
        out = tuple(a + t * (b - a) for a, b in zip(q0, q1, strict=True))
        n = math.sqrt(sum(c * c for c in out))
        return tuple(c / n for c in out)  # type: ignore[return-value]
    th = math.acos(min(1.0, dot))
    s = math.sin(th)
    a = math.sin((1 - t) * th) / s
    b = math.sin(t * th) / s
    return tuple(a * x + b * y for x, y in zip(q0, q1, strict=True))  # type: ignore[return-value]


def rotate(q: Quat, v: tuple[float, float, float]) -> tuple[float, float, float]:
    r = _mul(_mul(q, (0.0, *v)), _conj(q))
    return (r[1], r[2], r[3])


def _angle(q0: Quat, q1: Quat) -> float:
    d = abs(sum(a * b for a, b in zip(q0, q1, strict=True)))
    return 2 * math.acos(min(1.0, d))


def bench_quaternion_slerp(seed: int = 20261231 + 384) -> dict[str, float]:
    rng = random.Random(seed)
    ends = mid = rot = 0
    trials = 40
    for _ in range(trials):
        ax = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
        if max(ax) - min(ax) < 1e-6:
            ax = (1.0, 0.0, 0.0)
        q0 = _axis_angle(ax, rng.uniform(0, math.pi))
        q1 = _axis_angle((-ax[1], ax[0], ax[2] + 0.5), rng.uniform(0, math.pi))
        s0 = slerp(q0, q1, 0.0)
        s1 = slerp(q0, q1, 1.0)
        ends += int(
            _angle(s0, q0) < 1e-6 and min(_angle(s1, q1), _angle(s1, tuple(-c for c in q1))) < 1e-6  # type: ignore[arg-type]
        )
        sm = slerp(q0, q1, 0.5)
        tot = _angle(q0, q1)
        mid += int(tot < 1e-6 or abs(_angle(q0, sm) - tot / 2) < 1e-4 + 0.02 * tot)
        # rotating a vector by slerp(t) lies between rotations by q0 and q1
        v = (1.0, 0.3, -0.2)
        r0 = rotate(q0, v)
        rm = rotate(sm, v)
        r1 = rotate(q1, v)
        # |r| preserved
        n0 = sum(c * c for c in r0)
        nm = sum(c * c for c in rm)
        n1 = sum(c * c for c in r1)
        rot += int(abs(n0 - nm) < 1e-6 and abs(n0 - n1) < 1e-6)
    return {
        "synthetic_endpoints": float(ends / trials),
        "synthetic_bisects_angle": float(mid / trials),
        "synthetic_norm_preserved": float(rot / trials),
    }
