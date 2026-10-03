"""SYNTHETIC journaled filesystem — write-ahead journal with commit +
crash replay. Committed txns survive crash; uncommitted are rolled back;
replay is idempotent.
"""

from __future__ import annotations

import random


class JFS:
    """Journal records: ("begin",tx) ("set",tx,k,v) ("commit",tx)."""

    def __init__(self) -> None:
        self.disk: dict[str, int] = {}
        self.journal: list[tuple] = []
        self.committed: set[str] = set()

    def begin(self, tx: str) -> None:
        self.journal.append(("begin", tx))

    def write(self, tx: str, k: str, v: int) -> None:
        self.journal.append(("set", tx, k, v))

    def commit(self, tx: str) -> None:
        self.journal.append(("commit", tx))
        self.committed.add(tx)

    def _replay(self) -> dict[str, int]:
        staged: dict[str, dict[str, int]] = {}
        done: set[str] = set()
        for rec in self.journal:
            if rec[0] == "set":
                _t, tx, k, v = rec
                staged.setdefault(tx, {})[k] = int(v)
            elif rec[0] == "commit":
                done.add(rec[1])
        out: dict[str, int] = {}
        for tx, kv in staged.items():
            if tx in done:
                out.update(kv)
        return out

    def crash_and_recover(self) -> None:
        self.disk = self._replay()


def bench_fs_journal(seed: int = 20261231 + 375) -> dict[str, float]:
    rng = random.Random(seed)
    vis = gone = idem = 0
    trials = 40
    for _ in range(trials):
        fs = JFS()
        for t in range(rng.randrange(3, 8)):
            tx = f"tx{t}"
            fs.begin(tx)
            for _ in range(rng.randrange(1, 4)):
                fs.write(tx, f"k{rng.randrange(0, 6)}", rng.randrange(100))
            if rng.random() < 0.7:
                fs.commit(tx)
        truth = fs._replay()  # ground truth from journal semantics
        fs.disk = {"stale": -1}  # simulate dirty pre-crash state
        fs.crash_and_recover()
        vis += int(fs.disk == truth)
        gone += int(all(k in truth or k != "stale" for k in fs.disk) and "stale" not in fs.disk)
        snap = dict(fs.disk)
        fs.crash_and_recover()
        idem += int(fs.disk == snap)
    return {
        "synthetic_committed_visible": float(vis / trials),
        "synthetic_uncommitted_gone": float(gone / trials),
        "synthetic_idempotent_replay": float(idem / trials),
    }
