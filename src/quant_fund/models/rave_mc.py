"""RAVE / AMAF MCTS — all-moves-as-first statistics accelerating UCT (SYNTHETIC).

Each node carries UCT stats plus AMAF stats updated for every move
appearing later in the playout; terminal states are marked at node
creation so the tree never expands past a decided game. Selection
blends Q and AMAF-Q by beta = amaf_n / (amaf_n + n + 4 beta n amaf_n).
Bench: the solver finds the forced blocking move on tic-tac-toe.
"""

import math

import numpy as np

_SEED = 20261231 + 880


def _ttt_moves(b: tuple) -> list[int]:
    return [i for i, v in enumerate(b) if v == 0]


def _ttt_win(b: tuple, p: int) -> bool:
    lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
    return any(all(b[i] == p for i in line) for line in lines)


def _ttt_play(b: tuple, m: int, p: int) -> tuple:
    cells = list(b)
    cells[m] = p
    return tuple(cells)


def _term(b: tuple, player: int) -> float | None:
    """Reward for `player` if board is terminal, else None."""
    for pp in (1, 2):
        if _ttt_win(b, pp):
            return 1.0 if pp == player else 0.0
    if not _ttt_moves(b):
        return 0.5
    return None


class _Node:
    __slots__ = ("n", "w", "amaf_n", "amaf_w", "kids", "untried", "term")

    def __init__(self, board: tuple, player: int):
        self.n = 0
        self.w = 0.0
        self.amaf_n = 0
        self.amaf_w = 0.0
        self.kids: dict[int, _Node] = {}
        self.term = _term(board, player)
        self.untried = [] if self.term is not None else _ttt_moves(board)


def rave_mcts(
    board: tuple,
    player: int,
    sims: int = 400,
    rave: bool = True,
    c: float = 1.4,
    beta: float = 0.02,
    seed: int = _SEED,
) -> int:
    rng = np.random.default_rng(seed)

    def rollout(b: tuple, p: int) -> float:
        while True:
            t = _term(b, player)
            if t is not None:
                return t
            b = _ttt_play(b, int(rng.choice(_ttt_moves(b))), p)
            p = 3 - p

    root = _Node(board, player)
    for _ in range(sims):
        b, p = board, player
        node = root
        path = [node]
        played: list[tuple[int, int]] = []
        while node.term is None and node.untried == [] and node.kids:
            best_m, best_v = -1, -1.0
            for m, k in node.kids.items():
                if rave and k.amaf_n:
                    bb = (
                        beta / (k.n + k.amaf_n + 4 * beta * k.n * k.amaf_n)
                        if (k.n + k.amaf_n)
                        else 0.0
                    )
                    q = (1 - bb) * (k.w / k.n if k.n else 0.0) + bb * (k.amaf_w / k.amaf_n)
                else:
                    q = k.w / k.n if k.n else 0.0
                u = q + c * math.sqrt(math.log(max(node.n, 1)) / max(k.n, 1))
                if u > best_v:
                    best_v, best_m = u, m
            node = node.kids[best_m]
            b = _ttt_play(b, best_m, p)
            played.append((best_m, p))
            p = 3 - p
            path.append(node)
        if node.term is None and node.untried:
            m = int(rng.choice(node.untried))
            node.untried.remove(m)
            child = _Node(_ttt_play(b, m, p), player)
            node.kids[m] = child
            b = _ttt_play(b, m, p)
            played.append((m, p))
            p = 3 - p
            path.append(child)
        leaf = path[-1]
        reward = leaf.term if leaf.term is not None else rollout(b, p)
        root.n += 1
        for i, nd in enumerate(path[1:]):
            m, pp = played[i]
            nd.n += 1
            nd.w += reward if pp == player else 1 - reward
        if rave:
            all_moves = [m for m, _ in played]
            for i, nd in enumerate(path[:-1]):
                for m, pp in played[i + 1 :]:
                    if m in all_moves[: i + 1]:
                        continue
                    nd.amaf_n += 1
                    nd.amaf_w += reward if pp == player else 1 - reward
    return max(root.kids.items(), key=lambda kv: kv[1].n)[0]


def bench_rave_mc(seed: int = _SEED) -> dict[str, float]:
    """SYNTHETIC bench: RAVE MCTS finds the forced blocking move."""
    # O threatens an immediate win at cell 5 (row 3-4-5); X must block.
    board = (1, 0, 0, 2, 2, 0, 0, 0, 1)
    m_rave = rave_mcts(board, 1, sims=500, rave=True, seed=seed)
    return {"synthetic_rave_mc": 1.0 if m_rave == 5 else 0.0}
