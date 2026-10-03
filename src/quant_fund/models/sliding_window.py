"""SYNTHETIC sliding-window ARQ — go-back-N vs selective repeat over a
lossy channel. GBN retransmits the whole window on a loss; SR resends
only the lost packet. Metrics: delivery completeness, in-order receipt,
SR transmission efficiency.
"""

from __future__ import annotations

import random


def _lost(pkt: int, attempt: int, loss_p: float, salt: int) -> bool:
    """Deterministic per-attempt loss schedule shared by both ARQs."""
    return random.Random(salt * 100003 + pkt * 97 + attempt).random() < loss_p


def gbn_send(n: int, w: int, loss_p: float, salt: int = 0) -> tuple[int, list[int]]:
    sent = 0
    base = 0
    got: list[int] = []
    attempts: dict[int, int] = {}
    while base < n:
        for i in range(base, min(base + w, n)):
            sent += 1
            att = attempts.get(i, 0)
            attempts[i] = att + 1
            if _lost(i, att, loss_p, salt):
                base = i  # retransmit from i
                break
            if i == base:
                got.append(i)
                base = i + 1
            else:
                got.append(i)
    return sent, got


def sr_send(n: int, w: int, loss_p: float, salt: int = 0) -> tuple[int, list[int]]:
    sent = 0
    delivered: set[int] = set()
    base = 0
    got: list[int] = []
    attempts: dict[int, int] = {}
    while base < n:
        for i in range(base, min(base + w, n)):
            if i in delivered:
                continue
            sent += 1
            att = attempts.get(i, 0)
            attempts[i] = att + 1
            if not _lost(i, att, loss_p, salt):
                delivered.add(i)
        while base < n and base in delivered:
            got.append(base)
            base += 1
    return sent, got


def bench_sliding_window(seed: int = 20261231 + 401) -> dict[str, float]:
    rng = random.Random(seed)
    complete = inorder = eff = 0
    trials = 40
    for _ in range(trials):
        n, w = 20, 5
        p = rng.uniform(0.05, 0.2)
        s_g, g_g = gbn_send(n, w, p)
        s_r, g_r = sr_send(n, w, p)
        complete += int(len(g_g) == n and len(g_r) == n)
        inorder += int(g_g == sorted(g_g) and g_r == sorted(g_r))
        # SR never sends more than GBN on the same loss schedule:
        # every GBN retransmission of a delivered packet is extra work
        eff += int(s_r <= s_g)
    return {
        "synthetic_all_delivered": float(complete / trials),
        "synthetic_in_order": float(inorder / trials),
        "synthetic_sr_le_gbn_sends": float(eff / trials),
    }
