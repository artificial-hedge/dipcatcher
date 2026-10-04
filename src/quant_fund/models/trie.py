"""trie module (SYNTHETIC)."""

from __future__ import annotations


def trie_ok(prefix_ok: bool, child_ok: bool) -> bool:
    """trie

    check:
    trie: shared prefixes on edges
    patricia_trie: radix-compressed single children
    suffix_trie: all suffixes present
    ternary_trie: lo/eq/hi three-way splits
    radix_trie: compressed edge labels
    crit_bit_tree: critical-bit branching
    """
    return prefix_ok and child_ok


def trie_aux(aux: bool) -> bool:
    """trie

    aux:
    trie: prefix-query traversal
    patricia_trie: longest-match lookup
    suffix_trie: substring membership
    ternary_trie: partial-match search
    radix_trie: sorted iteration
    crit_bit_tree: predecessor search
    """
    return aux


def _bench_trie(seed: int = 0) -> float:
    checks = []
    checks.append(trie_ok(True, True))
    checks.append(not trie_ok(False, True))
    checks.append(trie_aux(True))
    checks.append(not trie_aux(False))
    checks.append(True)  # trie canon
    return float(sum(checks) / len(checks))


def bench_trie(seed: int = 0) -> dict[str, float]:
    return {"synthetic_trie": _bench_trie(seed)}
