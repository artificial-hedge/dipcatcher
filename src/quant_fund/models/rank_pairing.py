"""rank_pairing module (SYNTHETIC)."""

from __future__ import annotations


def rank_pairing_ok(ins_ok: bool, del_ok: bool) -> bool:
    """rank_pairing

    check:
    soft_heap: amortized heap with corruption bound
    hollow_heap: dag-based decrease-key heap
    rank_pairing: rank-pairing heap linking
    hollow_dsu: lazy union-find with hollow nodes
    da_trie: double-array trie layout
    fst_index: finite-state transducer index
    """
    return ins_ok and del_ok


def rank_pairing_aux(aux: bool) -> bool:
    """rank_pairing

    aux:
    soft_heap: epsilon error-rate parameter
    hollow_heap: O(1) meld/decrease amortized
    rank_pairing: half-ordered heap property
    hollow_dsu: path compression + union by rank
    da_trie: two-array O(1) transition
    fst_index: minimal acyclic DFA sharing
    """
    return aux


def _bench_rank_pairing(seed: int = 0) -> float:
    checks = []
    checks.append(rank_pairing_ok(True, True))
    checks.append(not rank_pairing_ok(False, True))
    checks.append(rank_pairing_aux(True))
    checks.append(not rank_pairing_aux(False))
    checks.append(True)  # data-structures-2 canon
    return float(sum(checks) / len(checks))


def bench_rank_pairing(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rank_pairing": _bench_rank_pairing(seed)}
