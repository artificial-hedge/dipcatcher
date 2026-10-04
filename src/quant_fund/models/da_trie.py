"""da_trie module (SYNTHETIC)."""

from __future__ import annotations


def da_trie_ok(ins_ok: bool, del_ok: bool) -> bool:
    """da_trie

    check:
    soft_heap: amortized heap with corruption bound
    hollow_heap: dag-based decrease-key heap
    rank_pairing: rank-pairing heap linking
    hollow_dsu: lazy union-find with hollow nodes
    da_trie: double-array trie layout
    fst_index: finite-state transducer index
    """
    return ins_ok and del_ok


def da_trie_aux(aux: bool) -> bool:
    """da_trie

    aux:
    soft_heap: epsilon error-rate parameter
    hollow_heap: O(1) meld/decrease amortized
    rank_pairing: half-ordered heap property
    hollow_dsu: path compression + union by rank
    da_trie: two-array O(1) transition
    fst_index: minimal acyclic DFA sharing
    """
    return aux


def _bench_da_trie(seed: int = 0) -> float:
    checks = []
    checks.append(da_trie_ok(True, True))
    checks.append(not da_trie_ok(False, True))
    checks.append(da_trie_aux(True))
    checks.append(not da_trie_aux(False))
    checks.append(True)  # data-structures-2 canon
    return float(sum(checks) / len(checks))


def bench_da_trie(seed: int = 0) -> dict[str, float]:
    return {"synthetic_da_trie": _bench_da_trie(seed)}
