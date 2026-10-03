"""A* maze routing (wave 291).

Lee/A* grid router with Manhattan heuristic and blocked cells; wire
cost vs BFS flood-fill oracle — must find the same minimum length.
"""

import heapq

_SEED = 20261231 + 832


def route(grid: list[list[int]], src: tuple[int, int], dst: tuple[int, int]) -> int:
    n, m = len(grid), len(grid[0])

    def h(c: tuple[int, int]) -> int:
        return abs(c[0] - dst[0]) + abs(c[1] - dst[1])

    pq = [(h(src), 0, src)]
    dist = {src: 0}
    while pq:
        _f, g, c = heapq.heappop(pq)
        if c == dst:
            return g
        if g > dist[c]:
            continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nc = (c[0] + dx, c[1] + dy)
            if 0 <= nc[0] < n and 0 <= nc[1] < m and grid[nc[0]][nc[1]] == 0:
                ng = g + 1
                if ng < dist.get(nc, 1 << 30):
                    dist[nc] = ng
                    heapq.heappush(pq, (ng + h(nc), ng, nc))
    return -1


def _bfs(grid: list[list[int]], src: tuple[int, int], dst: tuple[int, int]) -> int:
    from collections import deque

    n, m = len(grid), len(grid[0])
    dist = {src: 0}
    q = deque([src])
    while q:
        c = q.popleft()
        if c == dst:
            return dist[c]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nc = (c[0] + dx, c[1] + dy)
            if 0 <= nc[0] < n and 0 <= nc[1] < m and grid[nc[0]][nc[1]] == 0 and nc not in dist:
                dist[nc] = dist[c] + 1
                q.append(nc)
    return -1


def bench_a_star_route(seed: int = _SEED) -> dict[str, float]:
    grid = [[0] * 8 for _ in range(8)]
    for i in range(1, 7):
        grid[i][4] = 1  # wall with gap at row 0 and 7
    ok = int(route(grid, (0, 0), (7, 7)) == _bfs(grid, (0, 0), (7, 7)) == 14)
    grid2 = [[0] * 5 for _ in range(5)]
    grid2[2] = [1, 1, 0, 1, 1]
    ok += int(route(grid2, (0, 0), (4, 4)) == _bfs(grid2, (0, 0), (4, 4)) == 8)
    grid3 = [[0, 0, 1], [1, 0, 1], [0, 0, 0]]
    ok += int(route(grid3, (0, 0), (2, 2)) == 4)
    # unreachable
    g4 = [[0, 1, 0], [1, 1, 0], [0, 0, 0]]
    ok += int(route(g4, (0, 0), (0, 2)) == -1)
    return {"synthetic_route": float(ok == 4)}
