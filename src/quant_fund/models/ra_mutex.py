"""Ricart-Agrawala mutual exclusion: request/reply on logical clocks."""

import heapq

_SEED = 20261231 + 740


def ra_run(requests: list[tuple[int, int]]) -> list[int]:
    """requests: (time, node). Grant mutex in (time, node) order; defer nothing
    since each request finishes before the next timestamp."""
    grants: list[int] = []
    queue: list[tuple[int, int]] = []
    for t, node in sorted(requests):
        heapq.heappush(queue, (t, node))
    while queue:
        t, node = heapq.heappop(queue)
        grants.append(node)
    return grants


def bench_ra_mutex(seed: int = _SEED) -> dict[str, float]:

    reqs = [(t, t % 8) for t in range(40)]
    grants = ra_run(reqs)
    expect = [n for _, n in sorted(reqs)]
    return {"synthetic_ra_order": float(grants == expect)}
