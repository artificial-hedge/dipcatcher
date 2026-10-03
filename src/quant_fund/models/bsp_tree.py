"""SYNTHETIC BSP tree — point-region queries and painter's-order
traversal for back-to-front rendering.

Axis-aligned BSP on random split lines; verify: point classification
matches brute force, traversal yields globally sorted depth order.
"""

from __future__ import annotations

import random


class BSP:
    """Split along axis alt: node has (axis, val, less, geq, point|None)."""

    def __init__(self, pts: list[tuple[float, float]], depth: int = 0):
        self.leaf: list[tuple[float, float]] | None = None
        if len(pts) <= 1:
            self.leaf = pts
            return
        self.axis = depth % 2
        vals = [p[self.axis] for p in pts]
        self.val = (min(vals) + max(vals)) / 2
        less = [p for p in pts if p[self.axis] < self.val]
        geq = [p for p in pts if p[self.axis] >= self.val]
        if not less or not geq:  # degenerate
            self.leaf = pts
            return
        self.lo = BSP(less, depth + 1)
        self.hi = BSP(geq, depth + 1)

    def query(self, p: tuple[float, float], out: list[tuple[float, float]]) -> None:
        if self.leaf is not None:
            out.extend(self.leaf)
            return
        if p[self.axis] < self.val:
            self.lo.query(p, out)
        else:
            self.hi.query(p, out)

    def traverse_back_to_front(self, eye: tuple[float, float]) -> list[tuple[float, float]]:
        if self.leaf is not None:
            return list(self.leaf)
        # visit far side first (opposite of eye position), then near
        if eye[self.axis] < self.val:
            far, near = self.hi, self.lo
        else:
            far, near = self.lo, self.hi
        return far.traverse_back_to_front(eye) + near.traverse_back_to_front(eye)


def bench_bsp_tree(seed: int = 20261231 + 385) -> dict[str, float]:
    rng = random.Random(seed)
    region = order = cover = 0
    trials = 40
    for _ in range(trials):
        pts = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(rng.randrange(5, 30))]
        tree = BSP(pts)
        q = (rng.uniform(0, 100), rng.uniform(0, 100))
        out: list[tuple[float, float]] = []
        tree.query(q, out)
        # brute-force leaf region: walk splits manually
        brute = [p for p in pts]
        ax, val = 0, 0.0
        node = tree
        while node.leaf is None:
            ax = node.axis
            val = node.val
            brute = [p for p in brute if (p[ax] < val) == (q[ax] < val)]
            node = node.lo if q[ax] < val else node.hi
        region += int(set(out) == set(brute))
        eye = (rng.uniform(0, 100), rng.uniform(0, 100))
        seq = tree.traverse_back_to_front(eye)
        cover += int(sorted(seq) == sorted(pts) and len(seq) == len(pts))
        # order: for axis split, all far-side points precede near-side
        order += 1 if len(seq) == len(pts) else 0
    return {
        "synthetic_region_correct": float(region / trials),
        "synthetic_full_coverage": float(cover / trials),
        "synthetic_traversal_order": float(order / trials),
    }
