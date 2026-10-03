"""Frontier-based exploration: BFS to nearest frontier cell over known-free map."""

import collections

import numpy as np

_SEED = 20261231 + 635


def _explore(grid: np.ndarray, rng: np.random.RandomState, steps: int = 4000) -> float:
    n = grid.shape[0]
    known = np.full((n, n), -1)  # -1 unknown, else cell state
    pos = np.array([n // 2, n // 2])
    if grid[pos[0], pos[1]] == 1:
        free0 = np.argwhere(grid == 0)
        if len(free0) == 0:
            return 0.0
        pos = free0[0]
    start0 = (int(pos[0]), int(pos[1]))

    def sense(p: np.ndarray) -> None:
        for i in range(max(0, p[0] - 4), min(n, p[0] + 5)):
            for j in range(max(0, p[1] - 4), min(n, p[1] + 5)):
                known[i, j] = grid[i, j]

    sense(pos)
    for _ in range(steps):
        unk = known == -1
        free = known == 0
        adj_unk = np.zeros_like(unk)
        adj_unk[1:, :] |= unk[:-1, :]
        adj_unk[:-1, :] |= unk[1:, :]
        adj_unk[:, 1:] |= unk[:, :-1]
        adj_unk[:, :-1] |= unk[:, 1:]
        front = np.argwhere(free & adj_unk)
        if len(front) == 0:
            break
        # BFS over known-free cells from pos to nearest frontier (via parent map)
        parent: dict[tuple[int, int], tuple[int, int] | None] = {(int(pos[0]), int(pos[1])): None}
        qq: collections.deque[tuple[int, int]] = collections.deque([(int(pos[0]), int(pos[1]))])
        tgt: tuple[int, int] | None = None
        front_set = {tuple(int(v) for v in f) for f in front}
        while qq:
            cur = qq.popleft()
            if cur in front_set and cur != tuple(pos):
                tgt = cur
                break
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (cur[0] + di, cur[1] + dj)
                if 0 <= nb[0] < n and 0 <= nb[1] < n and known[nb] == 0 and nb not in parent:
                    parent[nb] = cur
                    qq.append(nb)
        if tgt is None:
            break
        path: list[tuple[int, int]] = [tgt]
        cur_node: tuple[int, int] | None = tgt
        while cur_node is not None:
            cur_node = parent[cur_node]
            if cur_node is not None:
                path.append(cur_node)
        path.reverse()
        pos = np.array(path[1]) if len(path) > 1 else np.array(tgt)
        sense(pos)
    # oracle: reachable free cells via BFS from the agent's start cell
    reach = np.zeros((n, n), bool)
    qq = collections.deque([start0])
    start = start0
    reach[start] = True
    while qq:
        i, j = qq.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n and grid[ni, nj] == 0 and not reach[ni, nj]:
                reach[ni, nj] = True
                qq.append((ni, nj))
    return float(np.mean(known[reach] >= 0))


def bench_frontier_explore(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    covs = []
    for _ in range(8):
        n = 30
        grid = (rng.rand(n, n) < 0.12).astype(int)
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1
        covs.append(_explore(grid, rng))
    return {"synthetic_frontier_coverage": float(np.mean(covs))}
