"""Wave-259 networking-2 canon tests."""

from quant_fund.models.arp_table import ArpCache, bench_arp_table
from quant_fund.models.bgp_pathvec import bench_bgp_pathvec, bgp_converge
from quant_fund.models.dhcp_lease import DhcpServer, bench_dhcp_lease
from quant_fund.models.dns_resolver import Resolver, bench_dns_resolver
from quant_fund.models.eth_switch import Switch, bench_eth_switch
from quant_fund.models.nat_traversal import NAT, bench_nat_traversal


def test_bgp_basic():
    routes = bgp_converge([(0, 1), (1, 2), (0, 2)], 0, 3)
    assert routes[2] == (0, 2)  # direct beats 2-hop


def test_bgp_bench():
    assert bench_bgp_pathvec()["synthetic_bgp_optimal"] == 1.0


def test_dns_cache():
    r = Resolver({"a.com": "1.2.3.4"}, ttl=5)
    r.resolve("a.com")
    assert r.resolve("a.com") == "1.2.3.4" and r.queries == 1


def test_dns_bench():
    assert bench_dns_resolver()["synthetic_dns_correct"] == 1.0


def test_nat_pinhole():
    n = NAT()
    n.send(("8.8.8.8", 53))
    assert n.inbound_ok(("8.8.8.8", 53))
    assert not n.inbound_ok(("9.9.9.9", 53))


def test_nat_bench():
    r = bench_nat_traversal()
    assert 0.0 <= r["synthetic_nat_punched"] <= 1.0


def test_arp_ttl():
    arp = ArpCache(ttl=5)
    arp.learn("10.0.0.1", "aa")
    assert arp.lookup("10.0.0.1") == "aa"
    arp.now = 10
    assert arp.lookup("10.0.0.1") is None


def test_arp_bench():
    assert bench_arp_table()["synthetic_arp_correct"] == 1.0


def test_dhcp_basic():
    s = DhcpServer(["10.0.0.5"], lease=10)
    assert s.request("m1") == "10.0.0.5"
    assert s.request("m2") is None


def test_dhcp_bench():
    assert bench_dhcp_lease()["synthetic_dhcp_consistent"] == 1.0


def test_switch_learn():
    sw = Switch()
    sw.frame("a", "b", 1)
    assert sw.frame("b", "a", 2) is True


def test_switch_bench():
    assert bench_eth_switch()["synthetic_switch_learn"] == 1.0
