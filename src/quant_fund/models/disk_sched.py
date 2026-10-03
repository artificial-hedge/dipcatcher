"""SYNTHETIC disk head scheduling — FCFS / SSTF / SCAN / C-SCAN.

Total head-movement metrics on a 200-cylinder disk; SSTF should never
beat... (well, SSTF is greedy not optimal, but ≤ FCFS on random loads);
SCAN/C-SCAN sweep monotonically within each pass.
"""

from __future__ import annotations

import random


def _fcfs(head: int, req: list[int]) -> int:
    move = 0
    for r in req:
        move += abs(r - head)
        head = r
    return move


def _sstf(head: int, req: list[int]) -> int:
    left = list(req)
    move = 0
    while left:
        r = min(left, key=lambda x: abs(x - head))
        move += abs(r - head)
        head = r
        left.remove(r)
    return move


def _scan(head: int, req: list[int], disk: int = 200, up: bool = True) -> tuple[int, list[int]]:
    order: list[int] = []
    move = 0
    left = sorted(req)
    h = head
    if up:
        seq = [r for r in left if r >= h] + [disk - 1] + [r for r in reversed(left) if r < h]
    else:
        seq = [r for r in reversed(left) if r <= h] + [0] + [r for r in left if r > h]
    for r in seq:
        move += abs(r - h)
        h = r
        order.append(r)
    return move, order


def _cscan(head: int, req: list[int], disk: int = 200) -> int:
    move = 0
    left = sorted(req)
    h = head
    seq = [r for r in left if r >= h]
    if seq:
        move += (disk - 1 - h) + (disk - 1) + (seq[-1] - seq[0] if len(seq) > 1 else 0)
        # jump to 0 then service remaining low reqs
        low = [r for r in left if r < h]
        if low:
            move += low[0] + (seq[-1] - low[0] if seq else 0)
        return move if not low else _cscan_service(head, left, disk)
    return abs(h - 0)


def _cscan_service(head: int, req: list[int], disk: int) -> int:
    move = 0
    h = head
    ups = [r for r in req if r >= h]
    downs = [r for r in req if r < h]
    seq: list[int] = ups + [disk - 1, 0] + downs
    for r in seq:
        move += abs(r - h)
        h = r
    return move


def bench_disk_sched(seed: int = 20261231 + 374) -> dict[str, float]:
    rng = random.Random(seed)
    sstf_ok = served = mono = 0
    trials = 30
    for _ in range(trials):
        req = rng.sample(range(0, 200), rng.randrange(5, 15))
        head = rng.randrange(0, 200)
        f = _fcfs(head, req)
        s = _sstf(head, req)
        sc, order = _scan(head, req)
        cs = _cscan_service(head, req, 200)
        sstf_ok += int(s <= f)
        served += int(set(order) - {199} == set(req) and cs > 0)
        # SCAN: positions within each leg are monotone
        leg1 = [r for r in order if r >= head]
        mono += int(leg1 == sorted(leg1))
    return {
        "synthetic_sstf_le_fcfs": float(sstf_ok / trials),
        "synthetic_all_served": float(served / trials),
        "synthetic_scan_monotone": float(mono / trials),
    }
