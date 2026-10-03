"""Wave-312 distributed-4 module unit tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.abd_register import ABDRegister
from quant_fund.models.delta_crdt import ORSetDelta
from quant_fund.models.hlc_clock import hlc_local, hlc_recv
from quant_fund.models.quorum_weighted import all_quorums, is_quorum
from quant_fund.models.raft_log import Follower
from quant_fund.models.tot_order import run_tot_order


def test_hlc_rules() -> None:
    assert hlc_local(5, (3, 2)) == (5, 0)
    assert hlc_local(3, (3, 2)) == (3, 3)
    assert hlc_recv(3, (3, 1), (3, 5)) == (3, 6)
    assert hlc_recv(7, (3, 1), (5, 2)) == (7, 0)
    assert hlc_recv(2, (3, 1), (5, 2)) == (5, 3)
    assert hlc_recv(4, (5, 1), (4, 9)) == (5, 2)


def test_delta_crdt_merge_converges() -> None:
    a, b = ORSetDelta(0), ORSetDelta(1)
    d1 = a.add("k")
    d2 = b.remove("k")  # unseen -> empty tomb
    a.merge(d2)
    b.merge(d1)
    assert a.value() == b.value() == {"k"}


def test_raft_reject_and_retry() -> None:
    f = Follower()
    f.log = [1, 1, 2]
    assert not f.append_entries(3, 5, [2], 0)
    assert f.append_entries(3, 2, [3], 3)
    assert f.log == [1, 1, 2, 3]
    assert f.commit == 3


def test_tot_order_prefix_delivery() -> None:
    rng = np.random.default_rng(7)
    delivered, order, causal = run_tot_order(3, 4, rng)
    pos = {m: i for i, m in enumerate(order)}
    assert all(d == order for d in delivered)
    assert all(pos[a] < pos[b] for a, b in causal)


def test_quorum_intersection_small() -> None:
    w = np.array([1, 1, 1])
    wq = all_quorums(w, 2)
    rq = all_quorums(w, 2)
    assert all(len(a & b) > 0 for a in wq for b in rq)
    assert is_quorum(w, {0, 2}, 2) and not is_quorum(w, {0}, 2)


def test_abd_atomicity() -> None:
    rng = np.random.default_rng(3)
    reg = ABDRegister(5)
    reg.write(5, rng)
    assert reg.read(rng)[1] == 5
