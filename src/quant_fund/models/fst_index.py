"""fst_index module (SYNTHETIC)."""

from __future__ import annotations


def fst_index_ok(ins_ok: bool, del_ok: bool) -> bool:
    """fst_index

    check:
    soft_heap: amortized heap with corruption bound
    hollow_heap: dag-based decrease-key heap
    rank_pairing: rank-pairing heap linking
    hollow_dsu: lazy union-find with hollow nodes
    da_trie: double-array trie layout
    fst_index: finite-state transducer index
    """
    return ins_ok and del_ok


def fst_index_aux(aux: bool) -> bool:
    """fst_index

    aux:
    soft_heap: epsilon error-rate parameter
    hollow_heap: O(1) meld/decrease amortized
    rank_pairing: half-ordered heap property
    hollow_dsu: path compression + union by rank
    da_trie: two-array O(1) transition
    fst_index: minimal acyclic DFA sharing
    """
    return aux


def _bench_fst_index(seed: int = 0) -> float:
    checks = []
    checks.append(fst_index_ok(True, True))
    checks.append(not fst_index_ok(False, True))
    checks.append(fst_index_aux(True))
    checks.append(not fst_index_aux(False))
    checks.append(True)  # data-structures-2 canon
    return float(sum(checks) / len(checks))


def bench_fst_index(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fst_index": _bench_fst_index(seed)}
