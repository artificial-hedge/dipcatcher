"""Cubemap face selection (wave 293) (SYNTHETIC).

Direction → (face, u, v): dominant axis picks face; verified against
the six canonical directions and consistency: sampling each face center
returns that face's texel.
"""

import numpy as np

_SEED = 20261231 + 846

FACES = ["+x", "-x", "+y", "-y", "+z", "-z"]


def face_uv(d: np.ndarray) -> tuple[int, float, float]:
    ax = int(np.argmax(np.abs(d)))
    s = np.sign(d[ax]) or 1
    face = 2 * ax + int(s < 0)
    if ax == 0:
        u, v = -d[2] * s, -d[1]
    elif ax == 1:
        u, v = d[0], -d[2] * s
    else:
        u, v = d[0] * s, d[1]
    m = abs(d[ax])
    return face, (u / m + 1) / 2, (v / m + 1) / 2


def bench_env_map(seed: int = _SEED) -> dict[str, float]:
    dirs = [
        (np.array([1.0, 0, 0]), 0),
        (np.array([-1.0, 0, 0]), 1),
        (np.array([0, 1.0, 0]), 2),
        (np.array([0, -1.0, 0]), 3),
        (np.array([0, 0, 1.0]), 4),
        (np.array([0, 0, -1.0]), 5),
    ]
    ok = int(all(face_uv(d)[0] == f for d, f in dirs))
    # uv within [0,1]
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(200, 3))
    ok += int(all(0 <= face_uv(p)[1] <= 1 and 0 <= face_uv(p)[2] <= 1 for p in pts))
    # face center maps to uv=(0.5,0.5)
    ok += int(
        all(abs(face_uv(d)[1] - 0.5) < 1e-12 and abs(face_uv(d)[2] - 0.5) < 1e-12 for d, _f in dirs)
    )
    return {"synthetic_env_map": float(ok == 3)}
