"""Adapter tests for wave-292 networking-4 canon benches."""

from quant_fund.research.benches_w292 import (
    bench_doh_wire_family,
    bench_qpack_pack_family,
    bench_quic_streams_family,
    bench_sctp_tsn_family,
    bench_tls13_trans_family,
    bench_wg_ik_family,
)


def test_families_return_synthetic_scores():
    for fn in [
        bench_quic_streams_family,
        bench_tls13_trans_family,
        bench_qpack_pack_family,
        bench_wg_ik_family,
        bench_doh_wire_family,
        bench_sctp_tsn_family,
    ]:
        out = fn()
        assert out
        assert all(k.startswith("synthetic_") for k in out)
