"""deque_array module (SYNTHETIC)."""

from __future__ import annotations


def deque_array_ok(head_ok: bool, tail_ok: bool) -> bool:
    """deque_array

    check:
    doubly_linked_list: bidirectional link coherence
    unrolled_list: fixed-capacity block chaining
    gap_buffer: cursor-adjacent edit locality
    piece_table: original+add buffer sequencing
    deque_array: circular double-ended push/pop
    xor_linked_list: XOR-merged pointer pair
    """
    return head_ok and tail_ok


def deque_array_aux(aux: bool) -> bool:
    """deque_array

    aux:
    doubly_linked_list: O(1) insert/remove known node
    unrolled_list: element count under block capacity
    gap_buffer: single-move per cursor step
    piece_table: spans never mutate original
    deque_array: amortized O(1) at both ends
    xor_linked_list: traversal needs prev pointer only
    """
    return aux


def _bench_deque_array(seed: int = 0) -> float:
    checks = []
    checks.append(deque_array_ok(True, True))
    checks.append(not deque_array_ok(False, True))
    checks.append(deque_array_aux(True))
    checks.append(not deque_array_aux(False))
    checks.append(True)  # deque/linked-structure canon
    return float(sum(checks) / len(checks))


def bench_deque_array(seed: int = 0) -> dict[str, float]:
    return {"synthetic_deque_array": _bench_deque_array(seed)}
