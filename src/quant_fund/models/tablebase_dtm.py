"""Endgame tablebase — retrograde distance-to-mate on a finite game (SYNTHETIC).

Game: a linear race. Two pawns on a track of L cells; the side to move
advances its own pawn 1 or 2 cells, or captures (lands on the enemy cell,
sending it back to 0). First pawn to reach the last cell wins. Retrograde
iteration assigns win-in-k / lose-in-k / draw to every state, giving a
full DTM table verified against a memoized minimax oracle.
"""

_SEED = 20261231 + 878

L = 8  # track cells 0..L-1; reaching L-1 wins


def _moves(state: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    a, b, turn = state
    cur, opp = (a, b) if turn == 0 else (b, a)
    out = []
    for step in (1, 2):
        nc = cur + step
        if nc > L - 1:
            continue
        no = 0 if nc == opp else opp
        out.append((nc, no, 1) if turn == 0 else (no, nc, 0))
    return out


def _terminal(state: tuple[int, int, int]) -> int | None:
    a, b, turn = state
    if a == L - 1:
        return 0
    if b == L - 1:
        return 1
    return None


def _all_states() -> list[tuple[int, int, int]]:
    return [(a, b, t) for a in range(L) for b in range(L) for t in range(2) if a != b]


def build_tablebase() -> dict[tuple[int, int, int], tuple[int, int]]:
    """Retrograde: state -> (outcome for mover: +1 win / -1 loss / 0 draw, DTM)."""
    tb: dict[tuple[int, int, int], tuple[int, int]] = {}
    states = _all_states()
    frontier: list[tuple[int, int, int]] = []
    for s in states:
        t = _terminal(s)
        if t is not None:
            tb[s] = (-1, 0) if t == s[2] else (1, 0)
            frontier.append(s)
    for _ in range(200):
        new = []
        for s in states:
            if s in tb:
                continue
            succ = _moves(s)
            vals = [tb[x] for x in succ if x in tb]
            if any(v == -1 for v, _ in vals):
                d = min(d for v, d in vals if v == -1)
                tb[s] = (1, d + 1)
                new.append(s)
            elif len(vals) == len(succ) and all(v == 1 for v, _ in vals):
                tb[s] = (-1, max(d for _, d in vals) + 1)
                new.append(s)
        if not new:
            break
    for s in states:
        tb.setdefault(s, (0, 0))
    return tb


def _oracle(
    state: tuple[int, int, int], depth: int, memo: dict[tuple, tuple[int, int]]
) -> tuple[int, int]:
    """Minimax+DTM oracle; cycles deeper than bound are draws."""
    key = (state, depth)
    if key in memo:
        return memo[key]
    t = _terminal(state)
    if t is not None:
        memo[key] = (-1, 0) if t == state[2] else (1, 0)
        return memo[key]
    if depth <= 0:
        return (0, 0)
    best_v, best_d = -1, 0
    for s in _moves(state):
        v, d = _oracle(s, depth - 1, memo)
        v, d = -v, d + 1
        if v > best_v or (v == best_v == 1 and d < best_d) or (v == best_v == -1 and d > best_d):
            best_v, best_d = v, d
    memo[key] = (best_v, best_d)
    return memo[key]


def bench_tablebase_dtm(seed: int = _SEED) -> dict[str, float]:
    """SYNTHETIC bench: tablebase verdicts+DTM match an unrolled minimax oracle."""
    tb = build_tablebase()
    mismatches = 0
    n_win = 0
    for s in _all_states():
        ov, od = _oracle(s, 64, {})
        tv, td = tb[s]
        if ov != tv or tv == 1 and td != od:
            mismatches += 1
        if tv == 1:
            n_win += 1
    return {"synthetic_tablebase_dtm": 1.0 if mismatches == 0 else 1.0 - mismatches / 20}
