"""SYNTHETIC deterministic single-tape Turing machine simulator.

Two classic programs: binary increment (input/output in tape alphabet) and
the canonical a^n b^n decider via crossing-off.  Verified against integer /
language oracles over many strings.
"""

from __future__ import annotations

import random

BLANK = "_"


def run_tm(
    transitions: dict[tuple[str, str], tuple[str, str, int]],
    tape_in: str,
    start: str = "q0",
    halt: str = "qh",
    cap: int = 20000,
) -> tuple[str, bool]:
    """Run deterministic TM; returns (final tape string stripped, halted)."""
    tape = list(tape_in) if tape_in else [BLANK]
    head, st, steps = 0, start, 0
    while st != halt and steps < cap:
        sym = tape[head] if head < len(tape) else BLANK
        key = (st, sym)
        if key not in transitions:
            return "".join(tape).strip(BLANK), False
        nst, ws, mv = transitions[key]
        if head < len(tape):
            tape[head] = ws
        else:
            tape.append(ws)
        head += mv
        if head < 0:
            tape.insert(0, BLANK)
            head = 0
        st = nst
        steps += 1
    return "".join(tape).strip(BLANK), st == halt


def _increment_tm() -> dict[tuple[str, str], tuple[str, str, int]]:
    # q0: scan right to end; q1: carry-propagate left, writing result.
    tr: dict[tuple[str, str], tuple[str, str, int]] = {}
    for b in "01":
        tr[("q0", b)] = ("q0", b, 1)
    tr[("q0", BLANK)] = ("q1", BLANK, -1)
    tr[("q1", "0")] = ("qh", "1", 0)
    tr[("q1", "1")] = ("q1", "0", -1)
    tr[("q1", BLANK)] = ("qh", "1", 0)
    return tr


def _anbn_tm() -> dict[tuple[str, str], tuple[str, str, int]]:
    # Cross off leftmost 'a' (→X), scan right for leftmost 'b' (→Y),
    # return; accept when only X/Y remain.
    tr: dict[tuple[str, str], tuple[str, str, int]] = {}
    tr[("q0", "a")] = ("q1", "X", 1)
    tr[("q0", "Y")] = ("q3", "Y", 1)  # no more a's — check all Y
    tr[("q0", BLANK)] = ("qr", BLANK, 0)
    for s in "aY":
        tr[("q1", s)] = ("q1", s, 1)
    tr[("q1", "b")] = ("q2", "Y", -1)
    tr[("q1", BLANK)] = ("qr", BLANK, 0)
    for s in "aYX":
        tr[("q2", s)] = ("q2", s, -1)
    tr[("q2", BLANK)] = ("q0", BLANK, 1)
    tr[("q3", "Y")] = ("q3", "Y", 1)
    tr[("q3", BLANK)] = ("qh", BLANK, 0)
    tr[("q3", "a")] = ("qr", "a", 0)
    tr[("q0", "X")] = ("q0", "X", 1)  # skip crossed-off a's
    tr[("q0", "b")] = ("qr", "b", 0)
    return tr


def bench_turing_machine(seed: int = 20261231 + 500) -> dict[str, float]:
    rng = random.Random(seed)
    inc_ok = 0
    n_inc = 40
    for _ in range(n_inc):
        v = rng.randrange(1, 1 << 12)
        out, halted = run_tm(_increment_tm(), bin(v)[2:])
        inc_ok += int(halted and out == bin(v + 1)[2:])
    anbn_ok = 0
    n_lang = 50
    for _ in range(n_lang):
        n = rng.randrange(0, 6)
        s = "a" * n + "b" * (n if rng.random() < 0.5 else n + rng.choice([-1, 1]))
        if n == 0 and s == "":
            s = "ab"  # ensure non-empty mix
        truth = (
            len(s) > 0
            and s.count("a") == s.count("b") == len(s) // 2
            and s == "a" * s.count("a") + "b" * s.count("b")
            and s.count("a") >= 1
        )
        out, halted = run_tm(_anbn_tm(), s)
        anbn_ok += int(halted == bool(truth))
    bounded = all(run_tm(_increment_tm(), bin(v)[2:], cap=5000)[1] for v in range(1, 64))
    return {
        "synthetic_increment_exact": inc_ok / n_inc,
        "synthetic_anbn_decides": anbn_ok / n_lang,
        "synthetic_halts_bounded": float(bounded),
    }
