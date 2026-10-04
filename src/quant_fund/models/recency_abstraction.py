"""Recency-based heap abstraction (Balakrishnan-Reps style).

The most recently allocated cell is modeled exactly (singleton); older
cells merge into one summary partition. Simulates an allocator plus a
write-through pattern; the abstract state retains enough precision to
prove field invariants on the fresh cell while older cells stay
approximate — matching recency-abstraction results.
"""

from __future__ import annotations

_SEED = 20261231 + 1015

Cell = dict[str, int]


class RecencyHeap:
    def __init__(self) -> None:
        self.next_id = 0
        self.mr: Cell | None = None  # most recent: exact fields
        self.old: dict[str, set[int]] = {}  # summary partition: field -> possible vals

    def alloc(self) -> None:
        if self.mr is not None:
            for k, v in self.mr.items():
                self.old.setdefault(k, set()).add(v)
        self.mr = {"tag": self.next_id, "val": -1}
        self.next_id += 1

    def set_field(self, field: str, val: int) -> None:
        assert self.mr is not None
        self.mr[field] = val

    def fresh_val_range(self) -> tuple[int, int]:
        assert self.mr is not None
        return (self.mr["val"], self.mr["val"])

    def old_val_bounds(self) -> tuple[int, int]:
        vs = self.old.get("val", set())
        if not vs:
            return (-1, -1)
        return (min(vs), max(vs))


def bench_recency_abstraction(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    h = RecencyHeap()
    h.alloc()
    h.set_field("val", 5)
    h.alloc()  # 5 drops into summary; new fresh cell val=-1
    checks.append(h.old_val_bounds() == (5, 5))
    checks.append(h.fresh_val_range() == (-1, -1))
    h.set_field("val", 7)
    h.alloc()
    checks.append(h.old_val_bounds() == (5, 7))
    checks.append(h.fresh_val_range() == (-1, -1))
    # tag freshness: fresh tag strictly exceeds any old tag
    h.set_field("val", 1)
    old_tags = h.old.get("tag", set())
    assert h.mr is not None
    checks.append(h.mr["tag"] > max(old_tags))
    # summary never claims exactness on mixed vals -> bounds only
    h2 = RecencyHeap()
    h2.alloc()
    h2.set_field("val", 0)
    h2.alloc()
    h2.set_field("val", 10)
    h2.alloc()
    checks.append(h2.old_val_bounds() == (0, 10))
    return {"synthetic_recency_abstraction": float(sum(checks)) / len(checks)}
