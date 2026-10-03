"""Wave-259 adapter tests."""

from quant_fund.research.benches_w259 import (
    bench_arp_table_family,
    bench_bgp_pathvec_family,
    bench_dhcp_lease_family,
    bench_dns_resolver_family,
    bench_eth_switch_family,
    bench_nat_traversal_family,
)


def test_bench_bgp_pathvec_family():
    out = bench_bgp_pathvec_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_dns_resolver_family():
    out = bench_dns_resolver_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_nat_traversal_family():
    out = bench_nat_traversal_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_arp_table_family():
    out = bench_arp_table_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_dhcp_lease_family():
    out = bench_dhcp_lease_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_eth_switch_family():
    out = bench_eth_switch_family()
    assert out and all(k.startswith("synthetic_") for k in out)
