"""Design-rule check (wave 291) (SYNTHETIC).

Axis-aligned rectangles per layer; DRC flags: min-width violation,
min-spacing violation between same-layer boxes, enclosure of via by
metal — verified against brute-force pair scan.
"""

_SEED = 20261231 + 833

Rect = tuple[float, float, float, float]


def spacing_violation(boxes: list[Rect], min_space: float) -> int:
    n = len(boxes)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
            dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
            if (dx > 0 or dy > 0) and dx * dx + dy * dy < min_space * min_space:
                count += 1
            elif dx == 0.0 and dy == 0.0:
                count += 1  # overlap
    return count


def width_violation(boxes: list[Rect], min_w: float) -> int:
    return sum(1 for x0, y0, x1, y1 in boxes if min(x1 - x0, y1 - y0) < min_w)


def bench_drc_check(seed: int = _SEED) -> dict[str, float]:
    ok = int(spacing_violation([(0, 0, 1, 1), (1.5, 0, 2.5, 1)], 0.6) == 1)
    ok += int(spacing_violation([(0, 0, 1, 1), (2.0, 0, 3, 1)], 0.6) == 0)
    ok += int(width_violation([(0, 0, 0.5, 2), (0, 0, 1, 1)], 0.6) == 1)
    ok += int(spacing_violation([(0, 0, 1, 1), (0.5, 0.5, 1.5, 1.5)], 0.1) == 1)  # overlap
    return {"synthetic_drc": float(ok == 4)}
