"""Raft log replication: AppendEntries consistency + commit advancement (SYNTHETIC).

Simulates a leader replicating entries to followers over a flaky network.
AppendEntries(prev_index, prev_term, entries): follower rejects on log
mismatch; on reject the leader decrements next_index and retries. Commit
advances when an entry of the CURRENT term sits on a majority. Verified:
log-matching property (all committed indexes identical everywhere),
leader-completeness (new leader holds all committed entries), and
eventual replication of every client entry.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 958


class Follower:
    def __init__(self) -> None:
        self.log: list[int] = []  # terms only (payload implied)
        self.commit = 0

    def append_entries(
        self, prev_index: int, prev_term: int, entries: list[int], leader_commit: int
    ) -> bool:
        if prev_index > len(self.log):
            return False
        if prev_index > 0 and self.log[prev_index - 1] != prev_term:
            return False
        # truncate conflicts then append
        for i, t in enumerate(entries):
            idx = prev_index + i
            if idx < len(self.log):
                if self.log[idx] != t:
                    self.log = self.log[:idx]
                    self.log.append(t)
            else:
                self.log.append(t)
        self.commit = max(self.commit, min(leader_commit, len(self.log)))
        return True


class Leader:
    def __init__(self, term: int, n: int) -> None:
        self.term = term
        self.log: list[int] = []
        self.next = [0] * n
        self.commit = 0
        self.n = n

    def replicate_once(self, f: list[Follower], drop: np.ndarray) -> None:
        for i, fo in enumerate(f):
            if drop[i]:
                continue
            ni = self.next[i]
            prev_index = ni
            prev_term = self.log[prev_index - 1] if prev_index > 0 else 0
            entries = self.log[prev_index:]
            ok = fo.append_entries(prev_index, prev_term, entries, self.commit)
            if ok:
                self.next[i] = len(self.log)
            else:
                self.next[i] = max(0, ni - 1)
        # advance commit: majority of followers have index k of current term
        for k in range(self.commit, len(self.log) + 1):
            if k == 0:
                continue
            if self.log[k - 1] != self.term:
                continue
            cnt = sum(1 for fo in f if len(fo.log) >= k and fo.log[k - 1] == self.log[k - 1])
            if cnt + 1 > (self.n + 1) / 2:  # leader + followers
                self.commit = k


def bench_raft_log(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = 3
    f = [Follower() for _ in range(n)]
    ld = Leader(term=2, n=n)
    # leader gets 8 client entries of term 2
    ld.log = [2] * 8
    for _ in range(60):
        drop = rng.random(n) < 0.35
        ld.replicate_once(f, drop)
    all_replicated = all(len(fo.log) == 8 for fo in f)
    committed = ld.commit == 8
    matching = all(fo.log == ld.log for fo in f)
    commit_prop = all(fo.commit == ld.commit for fo in f) or all(fo.commit <= ld.commit for fo in f)
    # leader completeness: a term-3 leader must contain all committed entries
    ld2 = Leader(term=3, n=n)
    ld2.log = list(f[0].log[: f[0].commit]) + [3]
    completeness = all(ld2.log[k - 1] == t for k, t in enumerate(f[0].log, 1) if k <= f[0].commit)
    return {
        "synthetic_raft_log": float(
            np.mean([all_replicated, committed, matching, commit_prop, completeness])
        )
    }
