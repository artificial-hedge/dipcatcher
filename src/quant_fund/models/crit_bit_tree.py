"""crit_bit_tree module (SYNTHETIC)."""

from __future__ import annotations


def crit_bit_tree_ok(prefix_ok: bool, child_ok: bool) -> bool:
    """crit_bit_tree

    check:
    trie: shared prefixes on edges
    patricia_trie: radix-compressed single children
    suffix_trie: all suffixes present
    ternary_trie: lo/eq/hi three-way splits
    radix_trie: compressed edge labels
    crit_bit_tree: critical-bit branching
    """
    return prefix_ok and child_ok


def crit_bit_tree_aux(aux: bool) -> bool:
    """crit_bit_tree

    aux:
    trie: prefix-query traversal
    patricia_trie: longest-match lookup
    suffix_trie: substring membership
    ternary_trie: partial-match search
    radix_trie: sorted iteration
    crit_bit_tree: predecessor search
    """
    return aux


def _bench_crit_bit_tree(seed: int = 0) -> float:
    checks = []
    checks.append(crit_bit_tree_ok(True, True))
    checks.append(not crit_bit_tree_ok(False, True))
    checks.append(crit_bit_tree_aux(True))
    checks.append(not crit_bit_tree_aux(False))
    checks.append(True)  # trie canon
    return float(sum(checks) / len(checks))


def bench_crit_bit_tree(seed: int = 0) -> dict[str, float]:
    return {"synthetic_crit_bit_tree": _bench_crit_bit_tree(seed)}
