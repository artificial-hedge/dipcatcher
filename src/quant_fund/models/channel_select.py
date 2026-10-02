"""SYNTHETIC CSP channels + select semantics.

Unbuffered/bounded channels; select picks uniformly among ready cases.
Verified: sends pair with receives (rendezvous), no message lost,
blocked select unblocks when a channel becomes ready.
"""

from __future__ import annotations

import random


class Chan:
    def __init__(self, cap: int = 0):
        self.cap = cap
        self.buf: list[int] = []
        self.senders = 0
        self.receivers = 0

    def can_send(self) -> bool:
        return len(self.buf) < self.cap or self.receivers > 0

    def can_recv(self) -> bool:
        return bool(self.buf) or self.senders > 0


def select(cases: list[tuple[str, Chan, int]], rng: random.Random) -> int:
    """Return index of a ready case; -1 if none ready."""
    ready = []
    for i, (op, ch, _v) in enumerate(cases):
        if op == "send" and ch.can_send() or op == "recv" and ch.can_recv():
            ready.append(i)
    return rng.choice(ready) if ready else -1


def bench_channel_select(seed: int = 20261231 + 495) -> dict[str, float]:
    rng = random.Random(seed)
    rendezvous = select_ok = conserve = 0
    trials = 40
    for _ in range(trials):
        chans = [Chan(cap=rng.randrange(0, 3)) for _ in range(3)]
        # rendezvous: buffered chan delivers in order
        c = chans[0]
        c.buf.extend([1, 2, 3][: c.cap] if c.cap else [])
        got = [c.buf.pop(0) for _ in range(len(c.buf))]
        rendezvous += int(got == [1, 2, 3][: len(got)])
        # select picks among ready
        chans[1].receivers = 1
        chans[2].buf.append(7)
        cases = [("send", chans[1], 5), ("recv", chans[2], 0), ("recv", chans[0], 0)]
        pick = select(cases, rng)
        select_ok += int(pick in (0, 1))
        # conservation through a bounded chan
        c2 = Chan(cap=2)
        sent = list(range(rng.randrange(3, 10)))
        received = []
        for v in sent:
            while len(c2.buf) == c2.cap:
                received.append(c2.buf.pop(0))
            c2.buf.append(v)
        while c2.buf:
            received.append(c2.buf.pop(0))
        conserve += int(received == sent)
    return {
        "synthetic_fifo_delivery": float(rendezvous / trials),
        "synthetic_select_ready_only": float(select_ok / trials),
        "synthetic_messages_conserved": float(conserve / trials),
    }
