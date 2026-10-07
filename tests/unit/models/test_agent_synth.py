"""Unit tests for quant_fund.models._agent_synth."""

from __future__ import annotations

from quant_fund.models._agent_synth import (
    N_ACTIONS,
    N_STATES,
    bfs_solution,
    is_goal,
    transition,
)


def test_transition_deterministic_and_bounded() -> None:
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            ns = transition(s, a)
            assert 0 <= ns < N_STATES
            assert transition(s, a) == ns


def test_bfs_returns_empty_path_when_start_is_goal() -> None:
    # s0 = 20 is already ≡ 0 mod 10: the honest plan is the empty path,
    # not a pointless cycle to a different goal state.
    assert is_goal(20)
    assert bfs_solution(20) == []


def test_bfs_finds_short_valid_plan() -> None:
    sol = bfs_solution(3)
    assert sol is not None
    s = 3
    for a in sol:
        s = transition(s, a)
    assert is_goal(s)


def test_bfs_respects_max_len() -> None:
    assert bfs_solution(3, max_len=0) is None
    # every start is solvable within the default budget
    assert all(bfs_solution(s) is not None for s in range(0, N_STATES, 7))
