"""Total-order broadcast via a sequencer.

The sequencer numbers every broadcast message; recipients buffer messages
that arrive out of order and deliver strictly in sequence order. Verified:
all recipients deliver the identical message sequence, every message is
delivered everywhere, and the sequence order is a linear extension of
Lamport causality (no happens-before edge is inverted).
"""

from __future__ import annotations

import heapq

import numpy as np

_SEED = 20261231 + 960


def run_tot_order(
    n_proc: int, n_msgs: int, rng: np.random.Generator
) -> tuple[list[list[int]], list[int], list[tuple[int, int]]]:
    """n_proc procs each broadcast n_msgs messages; returns (delivered,
    global_seq_order, causal_edges)."""
    order: list[int] = []
    causal: list[tuple[int, int]] = []
    expected = [0] * n_proc  # next seq number to deliver
    buffers: list[list[tuple[int, int]]] = [[] for _ in range(n_proc)]
    delivered: list[list[int]] = [[] for _ in range(n_proc)]
    last_sent: dict[int, int] = {}  # proc -> last msg id sent
    pending = [n_msgs] * n_proc
    # inbox: msg -> set of procs not yet received
    inflight: dict[int, set[int]] = {}
    msg_id = 0
    steps = 0
    while pending != [0] * n_proc or inflight:
        p = int(rng.integers(n_proc))
        if pending[p] > 0 and rng.random() < 0.6:
            if p in last_sent:
                causal.append((last_sent[p], msg_id))
            for m in delivered[p]:  # send follows delivery causally
                causal.append((m, msg_id))
            for q in range(n_proc):
                inflight.setdefault(msg_id, set()).add(q)
            last_sent[p] = msg_id
            order.append(msg_id)
            pending[p] -= 1
            msg_id += 1
        elif inflight:
            m = int(rng.choice(list(inflight.keys())))
            q = int(rng.choice(list(inflight[m])))
            inflight[m].discard(q)
            seq_m = order.index(m)
            heapq.heappush(buffers[q], (seq_m, m))
            # deliver prefix
            while buffers[q] and buffers[q][0][0] == expected[q]:
                _, mm = heapq.heappop(buffers[q])
                delivered[q].append(mm)
                expected[q] += 1
            if not inflight[m]:
                del inflight[m]
        steps += 1
        if steps > 100000:
            break
    return delivered, order, causal


def bench_tot_order(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    delivered, order, causal = run_tot_order(4, 5, rng)
    pos = {m: i for i, m in enumerate(order)}
    checks = [
        all(d == order for d in delivered),  # identical delivery order
        len(set(order)) == len(order),
        all(pos[a] < pos[b] for a, b in causal),  # seq extends causality
        len(delivered) == 4,
    ]
    return {"synthetic_tot_order": float(np.mean(checks))}
