"""Wave-902 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w902 import (
    bench_crit_bit_tree_family,
    bench_patricia_trie_family,
    bench_radix_trie_family,
    bench_suffix_trie_family,
    bench_ternary_trie_family,
    bench_trie_family,
)

_FAMILY_BENCHES = [
    bench_trie_family,
    bench_patricia_trie_family,
    bench_suffix_trie_family,
    bench_ternary_trie_family,
    bench_radix_trie_family,
    bench_crit_bit_tree_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
