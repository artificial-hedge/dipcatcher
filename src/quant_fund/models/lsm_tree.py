"""LSM-tree compaction lite — SYNTHETIC.

Memtable (sorted run) + levels with size-tiered merge. Verified:
lookups find latest version; compaction preserves all keys; level
runs sorted and non-overlapping after merge.
"""

from __future__ import annotations

import bisect
import random


class LSM:
    def __init__(self, memcap: int = 16) -> None:
        self.mem: list[tuple[int, int]] = []
        self.levels: list[list[tuple[int, int]]] = []
        self.memcap = memcap

    def put(self, k: int, v: int) -> None:
        i = bisect.bisect_left([x[0] for x in self.mem], k)
        if i < len(self.mem) and self.mem[i][0] == k:
            self.mem[i] = (k, v)
        else:
            self.mem.insert(i, (k, v))
        if len(self.mem) >= self.memcap:
            self._flush()

    def _flush(self) -> None:
        run = self.mem
        self.mem = []
        # merge into level 0; cascade if overlap of range with existing
        merged = self._merge([run] + self.levels[:1])
        if len(self.levels) > 1:
            merged = self._merge(merged + self.levels[1:])
            self.levels = merged
        else:
            self.levels = merged

    @staticmethod
    def _merge(runs: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
        # k-way merge keeping LATEST (runs[0] is newest)
        allk = sorted({k for r in runs for k, _v in r})
        latest: dict[int, int] = {}
        for r in reversed(runs):  # oldest first, newest overrides
            for k, v in r:
                latest[k] = v
        return [[(k, latest[k]) for k in allk]] if allk else []

    def get(self, k: int) -> int | None:
        for kk, v in self.mem:
            if kk == k:
                return v
        for run in self.levels:
            ks = [x[0] for x in run]
            i = bisect.bisect_left(ks, k)
            if i < len(run) and run[i][0] == k:
                return run[i][1]
        return None


def bench_lsm_tree(seed: int = 20261231 + 365) -> dict[str, float]:
    rng = random.Random(seed)
    latest_ok = sorted_ok = cover_ok = 0
    trials = 30
    for _ in range(trials):
        lsm = LSM(memcap=8)
        ref: dict[int, int] = {}
        for _ in range(rng.randrange(20, 80)):
            k = rng.randrange(0, 30)
            v = rng.randrange(1000)
            lsm.put(k, v)
            ref[k] = v
        latest_ok += int(all(lsm.get(k) == v for k, v in ref.items()))
        sorted_ok += int(
            all(run == sorted(run) for run in lsm.levels) and lsm.mem == sorted(lsm.mem)
        )
        cover_ok += int(all(lsm.get(k) is not None for k in ref))
    return {
        "synthetic_latest_wins": float(latest_ok / trials),
        "synthetic_runs_sorted": float(sorted_ok / trials),
        "synthetic_all_keys_present": float(cover_ok / trials),
    }
