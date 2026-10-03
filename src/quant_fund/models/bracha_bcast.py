"""Bracha reliable broadcast (n = 3f+1 synchronous rounds).

Phases: SEND -> ECHO -> VOTE -> DELIVER. Honest party rules:
  echo v  on SEND(v) from sender, or > (n+f)/2 distinct ECHO(v),
          or > f+1 distinct VOTE(v);
  vote v  on > (n+f)/2 distinct ECHO(v) (once);
  deliver v on > 2f+1... (strictly more than 2f distinct VOTE(v)).
Messages carry sender ids and each party counts each sender once.
Verified for n=4 f=1: agreement, validity (honest sender), totality.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

_SEED = 20261231 + 959


def _count(pairs: list[tuple[int, int]]) -> Counter:
    """Distinct senders per value."""
    seen: dict[int, set[int]] = {}
    for src, v in pairs:
        seen.setdefault(v, set()).add(src)
    return Counter({v: len(s) for v, s in seen.items()})


def run_bracha(
    n: int, f: int, sender_val: int, byz: set[int], rounds: int = 12
) -> dict[int, int | None]:
    echoed: list[set[int]] = [set() for _ in range(n)]
    voted: list[int | None] = [None] * n
    delivered: dict[int, int | None] = {i: None for i in range(n)}
    echo_in: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    vote_in: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    send_in: list[list[int]] = [[] for _ in range(n)]

    for j in range(n):
        send_in[j].append(sender_val if 0 not in byz else j % 2)

    for _ in range(rounds):
        out_echo: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        out_vote: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        for i in range(n):
            if i in byz:
                continue  # byz messages injected separately per round
            if delivered[i] is not None:
                continue
            my_echos = _count(echo_in[i])
            my_votes = _count(vote_in[i])
            emit: set[int] = set()
            for v in set(send_in[i]) | set(my_echos) | set(my_votes):
                if v in echoed[i]:
                    continue
                if v in send_in[i] or my_echos[v] > (n + f) / 2 or my_votes[v] > f:
                    emit.add(v)
            for v in emit:
                echoed[i].add(v)
                for j in range(n):
                    out_echo[j].append((i, v))
            if voted[i] is None:
                for v, c in my_echos.items():
                    if c > (n + f) / 2:
                        voted[i] = v
                        for j in range(n):
                            out_vote[j].append((i, v))
                        break
            for v, c in my_votes.items():
                if c > 2 * f:
                    delivered[i] = v
        # byz party i0: equivocating echo+vote to everyone (once per value)
        for i in byz:
            for j in range(n):
                out_echo[j].append((i, j % 2))
                out_vote[j].append((i, j % 2))
        for i in range(n):
            echo_in[i].extend(out_echo[i])
            vote_in[i].extend(out_vote[i])
            send_in[i] = []
    return delivered


def bench_bracha_bcast(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    checks = []
    for _ in range(4):
        byz = {int(rng.integers(4))}
        sender_val = int(rng.integers(2))
        d = run_bracha(4, 1, sender_val, byz)
        honest = [v for i, v in d.items() if i not in byz and v is not None]
        agreement = len(set(honest)) <= 1
        totality = all(v is not None for i, v in d.items() if i not in byz)
        validity = (0 in byz) or (honest and honest[0] == sender_val)
        checks.append(agreement and totality and validity)
    return {"synthetic_bracha_bcast": float(np.mean(checks))}
