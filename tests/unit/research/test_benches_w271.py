"""Wave-271 adapter bench tests."""

from quant_fund.research.benches_w271 import (
    bench_csma_ca_family,
    bench_diffserv_qos_family,
    bench_icmp_path_family,
    bench_ospf_lsa_family,
    bench_stp_spanning_family,
    bench_vlan_tag_family,
)

FAMS = [
    bench_ospf_lsa_family,
    bench_stp_spanning_family,
    bench_vlan_tag_family,
    bench_csma_ca_family,
    bench_icmp_path_family,
    bench_diffserv_qos_family,
]


def test_wave271_benches_all_synthetic() -> None:
    for f in FAMS:
        out = f()
        assert out, f.__name__
        for k in out:
            assert k.startswith("synthetic_"), k


def test_wave271_benches_score_high() -> None:
    for f in FAMS:
        assert max(f().values()) >= 0.5, f.__name__
