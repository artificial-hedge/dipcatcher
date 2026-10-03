"""R(3,3) = 6 verification (wave 282).

Any 2-coloring of K6 contains a monochromatic triangle: verified by
exhaustive check that every one of the 2^15 colorings has one, plus a
witness coloring of K5 that avoids it (pale construction / 5-cycle).
"""

import itertools

_SEED = 20261231 + 780


def _has_mono(n: int, color: list[list[int]]) -> bool:
    for a, b, c in itertools.combinations(range(n), 3):
        if color[a][b] == color[b][c] == color[a][c]:
            return True
    return False


def bench_ramsey_bound(seed: int = _SEED) -> dict[str, float]:
    # K5 witness: color cycle edges red, chords blue -> no mono triangle
    c5 = [[0] * 5 for _ in range(5)]
    for i in range(5):
        c5[i][(i + 1) % 5] = c5[(i + 1) % 5][i] = 1
    witness_ok = not _has_mono(5, c5)
    # every K6 coloring has a mono triangle
    edges = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    all_mono = True
    for bits in range(1 << len(edges)):
        c6 = [[0] * 6 for _ in range(6)]
        for e_i, (i, j) in enumerate(edges):
            if bits >> e_i & 1:
                c6[i][j] = c6[j][i] = 1
        if not _has_mono(6, c6):
            all_mono = False
            break
    return {"synthetic_ramsey": float(witness_ok and all_mono)}
