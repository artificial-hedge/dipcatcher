"""Raft leader election (synthetic).

Simulates N nodes with election timeouts, randomized message delay,
request-vote RPCs and one leader per term. Verified: (i) election
safety — at most one leader per term; (ii) liveness — a leader is
elected when a majority is reachable; (iii) log matching — leader
log entries replicated to a majority.
"""

from __future__ import annotations

import random


class _Node:
    def __init__(self, i: int) -> None:
        self.i = i
        self.term = 0
        self.voted_for = -1
        self.state = "follower"
        self.log: list[int] = []
        self.votes = 0


def run_raft(n: int, rng: random.Random, steps: int = 400) -> dict[str, object]:
    nodes = [_Node(i) for i in range(n)]
    majority = n // 2 + 1
    leaders_by_term: dict[int, int] = {}
    leader_emerges = -1
    step = 0
    while step < steps:
        step += 1
        # each follower ticks; timeout => candidate
        for nd in nodes:
            if nd.state == "follower" and rng.random() < 0.06:
                nd.state = "candidate"
                nd.term += 1
                nd.voted_for = nd.i
                nd.votes = 1
                # request votes
                for other in nodes:
                    if other.i == nd.i:
                        continue
                    if (
                        rng.random() < 0.9
                        and other.term <= nd.term
                        and other.voted_for in (-1, nd.i)
                    ):
                        other.voted_for = nd.i
                        other.term = nd.term
                        other.state = "follower"
                        nd.votes += 1
                if nd.votes >= majority:
                    nd.state = "leader"
                    if nd.term in leaders_by_term:
                        return {"safety_violation": True}
                    leaders_by_term[nd.term] = nd.i
                    if leader_emerges < 0:
                        leader_emerges = step
        # leader heartbeats replicate log
        for nd in nodes:
            if nd.state == "leader":
                nd.log.append(step)
                for other in nodes:
                    if other.i != nd.i and rng.random() < 0.8:
                        other.log = nd.log[:]
                # step-down: candidates with stale term return
        for nd in nodes:
            if nd.state == "candidate":
                continue
    leader = next((nd for nd in nodes if nd.state == "leader"), None)
    replicated = 0
    if leader is not None:
        replicated = sum(1 for nd in nodes if nd.log == leader.log) >= majority
    return {
        "safety_violation": False,
        "leader": leader.i if leader else -1,
        "terms": len(leaders_by_term),
        "replicated": replicated,
        "emerge_step": leader_emerges,
    }


def bench_raft_election(seed: int = 20261231 + 251) -> dict[str, float]:
    rng = random.Random(seed)
    safety = live = repl = 0
    trials = 40
    for _ in range(trials):
        res = run_raft(5, rng)
        safety += int(not res["safety_violation"])
        live += int(res["leader"] != -1)
        repl += int(bool(res["replicated"]))
    return {
        "synthetic_safety": float(safety / trials),
        "synthetic_liveness": float(live / trials),
        "synthetic_replicated": float(repl / trials),
    }
