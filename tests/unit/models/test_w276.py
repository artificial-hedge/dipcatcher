"""Wave-276 distributed-systems-3 module tests."""

from quant_fund.models.bully_elect import bully
from quant_fund.models.causal_bcast import causal_deliver
from quant_fund.models.chord_look import finger_table, lookup
from quant_fund.models.quorum_rw import quorum_read, quorum_write
from quant_fund.models.ra_mutex import ra_run


def test_ra_order_sorted() -> None:
    grants = ra_run([(5, 2), (1, 0), (3, 1)])
    assert grants == [0, 1, 2]


def test_ra_empty() -> None:
    assert ra_run([]) == []


def test_bully_highest_wins() -> None:
    assert bully([0, 1, 3, 5], 1) == 5


def test_bully_self_when_max() -> None:
    assert bully([0, 4], 4) == 4


def test_chord_finger_in_ring() -> None:
    nodes = [0, 8, 16, 32]
    ft = finger_table(0, nodes)
    assert ft[3] == 8 or ft[3] in nodes


def test_chord_lookup_direct() -> None:
    nodes = [0, 8, 16, 32, 48]
    assert lookup(0, 10, nodes) == 16


def test_quorum_intersection() -> None:
    reps = quorum_write(42, 5, 3)
    assert quorum_read(reps, 5, 3, 42) == 42


def test_causal_respects_deps() -> None:
    order = causal_deliver([(0, []), (1, [0]), (2, [1])])
    assert order == [0, 1, 2]


def test_causal_branching() -> None:
    order = causal_deliver([(0, []), (1, [0]), (2, [0]), (3, [1, 2])])
    pos = {v: i for i, v in enumerate(order)}
    assert pos[1] < pos[3] and pos[2] < pos[3] and pos[0] < pos[1]
