"""Wave-271 networking-3 module tests."""

import numpy as np

from quant_fund.models.csma_ca import csma_run
from quant_fund.models.diffserv_qos import diffserv_dequeue
from quant_fund.models.icmp_path import traceroute
from quant_fund.models.ospf_lsa import spf
from quant_fund.models.stp_spanning import stp_run
from quant_fund.models.vlan_tag import Trunk


def test_spf_simple() -> None:
    adj = {0: [(1, 5), (2, 1)], 1: [], 2: [(1, 1)]}
    assert spf(adj, 0) == {0: 0, 1: 2, 2: 1}


def test_spf_disconnected() -> None:
    adj = {0: [], 1: [], 2: []}
    assert spf(adj, 0) == {0: 0}


def test_stp_root_and_edges() -> None:
    adj = {0: {1, 2}, 1: {0, 2}, 2: {0, 1, 3}, 3: {2}}
    root, edges = stp_run(adj, 4)
    assert root == 0
    assert edges == 3


def test_vlan_isolation() -> None:
    t = Trunk()
    t.send(42, 1, 0)
    t.send(43, 2, 1)
    assert 42 in t.domains[1] and 42 not in t.domains[2]


def test_csma_bounds() -> None:
    rng = np.random.RandomState(0)
    d, a = csma_run(4, 16, rng)
    assert 0 <= d <= a


def test_traceroute_exact() -> None:
    assert traceroute([10, 20, 30]) == [10, 20, 30]


def test_diffserv_no_loss() -> None:
    qs = [[1, 2, 3], [4, 5]]
    out = diffserv_dequeue([list(q) for q in qs], np.array([1, 1]), 5)
    assert sorted(out) == [1, 2, 3, 4, 5]


def test_diffserv_priority() -> None:
    qs = [[1, 2, 3, 4, 5], [6]]
    out = diffserv_dequeue([list(q) for q in qs], np.array([3, 1]), 6)
    assert out[:3] == [1, 2, 3]
