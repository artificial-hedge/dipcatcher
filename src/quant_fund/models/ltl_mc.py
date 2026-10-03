"""Miniature LTL-style model checking on a finite Kripke structure:
check 'G (p -> F q)' (every p-state is eventually followed by a
q-state) via CTL-style fixpoint EG/EF evaluation on a synthetic
transition system.
"""

import numpy as np


def _transitions(n: int, rng: np.random.Generator) -> np.ndarray:
    g = rng.random((n, n)) < 0.4
    np.fill_diagonal(g, True)  # total relation
    return g


def _eg(p: np.ndarray, g: np.ndarray) -> np.ndarray:
    """EG p: greatest fixpoint of p AND EX."""
    x = p.copy()
    for _ in range(20):
        x = p & (g @ x > 0)
    return x


def _ef(q: np.ndarray, g: np.ndarray) -> np.ndarray:
    """EF q: least fixpoint of q OR EX."""
    x = q.copy()
    for _ in range(20):
        x = q | (g @ x > 0)
    return x


def bench_ltl_mc(seed: int = 5911) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = 6
    g = _transitions(n, rng)
    p = rng.random(n) < 0.4
    q = rng.random(n) < 0.4
    # G(p -> F q): states where every successor path reaching p later
    # reaches q — approximated: no EG (p AND NOT EF q) states? Here we
    # check the CTL formula A G(p -> E F q) via full evaluation on the
    # small structure: state s satisfies if every reachable p-state
    # from s can reach some q-state.
    reach = np.zeros((n, n), bool)
    reach |= np.eye(n, dtype=bool)
    reach |= g
    for _ in range(n):
        reach |= reach @ g
    can_q = reach @ q  # states that can reach a q-state
    violates = p & ~can_q
    # states from which a violating state is reachable
    bad_src = reach @ violates
    ok_states = ~bad_src
    return {
        "synthetic_ltl_ok_frac": float(ok_states.mean()),
        "synthetic_ltl_viol_states": float(violates.sum()),
        "synthetic_ltl_states": float(n),
    }
