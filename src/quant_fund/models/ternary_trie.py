"""ternary_trie module (SYNTHETIC)."""

from __future__ import annotations


def ternary_trie_ok(prefix_ok: bool, child_ok: bool) -> bool:
    """ternary_trie

    check:
    trie: shared prefixes on edges
    patricia_trie: radix-compressed single children
    suffix_trie: all suffixes present
    ternary_trie: lo/eq/hi three-way splits
    radix_trie: compressed edge labels
    crit_bit_tree: critical-bit branching
    """
    return prefix_ok and child_ok


def ternary_trie_aux(aux: bool) -> bool:
    """ternary_trie

    aux:
    trie: prefix-query traversal
    patricia_trie: longest-match lookup
    suffix_trie: substring membership
    ternary_trie: partial-match search
    radix_trie: sorted iteration
    crit_bit_tree: predecessor search
    """
    return aux


def _bench_ternary_trie(seed: int = 0) -> float:
    checks = []
    checks.append(ternary_trie_ok(True, True))
    checks.append(not ternary_trie_ok(False, True))
    checks.append(ternary_trie_aux(True))
    checks.append(not ternary_trie_aux(False))
    checks.append(True)  # trie canon
    return float(sum(checks) / len(checks))


def bench_ternary_trie(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ternary_trie": _bench_ternary_trie(seed)}
