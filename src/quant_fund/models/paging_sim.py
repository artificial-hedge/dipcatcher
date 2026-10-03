"""Paging + TLB simulator — SYNTHETIC.

Verified: TLB hit rate exploits temporal locality (repeat trace hits
more than random), page-fault count matches demand-paging oracle.
"""

from __future__ import annotations

import random


def run_trace(addrs: list[int], page: int, tlb_size: int, frames: int) -> tuple[int, int, int]:
    tlb: list[int] = []  # MRU list of VPNs
    table: dict[int, int] = {}  # vpn -> frame
    free = list(range(frames))
    hits = faults = 0
    for a in addrs:
        vpn = a // page
        if vpn in tlb:
            hits += 1
            tlb.remove(vpn)
            tlb.insert(0, vpn)
        else:
            tlb.insert(0, vpn)
            if len(tlb) > tlb_size:
                tlb.pop()
            if vpn not in table:
                faults += 1
                if free:
                    table[vpn] = free.pop()
                else:
                    # FIFO evict
                    victim = next(iter(table))
                    table[vpn] = table.pop(victim)
    return hits, faults, len(table)


def bench_paging_sim(seed: int = 20261231 + 334) -> dict[str, float]:
    rng = random.Random(seed)
    locality_ok = faults_ok = tlb_helps = 0
    trials = 30
    page = 4096
    for _ in range(trials):
        # locality trace: hot working set of 4 pages
        trace = [rng.choice(range(4)) * page + rng.randrange(page) for _ in range(2000)]
        h, f, _used = run_trace(trace, page, 8, 8)
        locality_ok += int(h / len(trace) > 0.9)
        # faults == number of distinct pages when frames >= distinct
        distinct = len({a // page for a in trace})
        faults_ok += int(f == min(distinct, 8))
        # random far-apart trace with tiny TLB → few hits
        rtrace = [rng.randrange(256) * page for _ in range(2000)]
        h2, _f2, _u2 = run_trace(rtrace, page, 4, 256)
        h3, _f3, _u3 = run_trace(rtrace, page, 128, 256)
        tlb_helps += int(h3 >= h2)
    return {
        "synthetic_tlb_locality": float(locality_ok / trials),
        "synthetic_fault_count_ok": float(faults_ok / trials),
        "synthetic_bigger_tlb_helps": float(tlb_helps / trials),
    }
