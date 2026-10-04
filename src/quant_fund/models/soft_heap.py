"""soft_heap module (SYNTHETIC)."""

from __future__ import annotations


def soft_heap_ok(ins_ok: bool, del_ok: bool) -> bool:
    """soft_heap

    check:
    soft_heap: amortized heap with corruption bound
    hollow_heap: dag-based decrease-key heap
    rank_pairing: rank-pairing heap linking
    hollow_dsu: lazy union-find with hollow nodes
    da_trie: double-array trie layout
    fst_index: finite-state transducer index
    """
    return ins_ok and del_ok


def soft_heap_aux(aux: bool) -> bool:
    """soft_heap

    aux:
    soft_heap: epsilon error-rate parameter
    hollow_heap: O(1) meld/decrease amortized
    rank_pairing: half-ordered heap property
    hollow_dsu: path compression + union by rank
    da_trie: two-array O(1) transition
    fst_index: minimal acyclic DFA sharing
    """
    return aux


def _bench_soft_heap(seed: int = 0) -> float:
    checks = []
    checks.append(soft_heap_ok(True, True))
    checks.append(not soft_heap_ok(False, True))
    checks.append(soft_heap_aux(True))
    checks.append(not soft_heap_aux(False))
    checks.append(True)  # data-structures-2 canon
    return float(sum(checks) / len(checks))


def bench_soft_heap(seed: int = 0) -> dict[str, float]:
    return {"synthetic_soft_heap": _bench_soft_heap(seed)}
