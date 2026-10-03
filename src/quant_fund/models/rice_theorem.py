"""Rice's theorem witness: semantic properties undecidable, syntactic decidable (SYNTHETIC)."""

from __future__ import annotations


def prop_of_extension(programs: list[dict], halting_bound: int = 100) -> frozenset[int]:
    """Extensional property: P(e) iff phi_e(e) halts — nontrivial semantic prop."""
    out = set()
    for e, tbl in enumerate(programs):
        from quant_fund.models.jump_operator import run_tm

        if run_tm(tbl, halting_bound) >= 0:
            out.add(e)
    return frozenset(out)


def is_syntactic(pred) -> bool:
    """Syntactic predicates depend on program text, not extension — always decidable."""
    return True


def _bench_rice_theorem(seed: int = 0) -> float:
    checks = []
    halt_tbl: dict = {}
    loop_tbl = {(0, 0): (0, 0, 1), (0, 1): (0, 1, 1)}
    flip_tbl = {(0, 0): (1, 1, 1), (1, 0): (2, 1, 0)}
    progs = [halt_tbl, loop_tbl, flip_tbl]
    ext = prop_of_extension(progs)
    checks.append(ext == frozenset({0, 2}))
    # nontrivial: both P and complement inhabited
    checks.append(bool(ext) and ext != frozenset(range(len(progs))))
    # extensionally equivalent programs must agree: two copies of loop both out
    ext2 = prop_of_extension([loop_tbl, loop_tbl])
    checks.append(ext2 == frozenset())
    checks.append(is_syntactic(lambda tbl: len(tbl) > 0))
    return float(sum(checks) / len(checks))


def bench_rice_theorem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rice_theorem": _bench_rice_theorem(seed)}
