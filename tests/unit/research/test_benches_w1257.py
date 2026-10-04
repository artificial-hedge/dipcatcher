"""Wave 1257 bench adapter tests."""

from quant_fund.research.benches_w1257 import (
    bench_culture_studies_family,
    bench_flow_cytometry_studies_family,
    bench_immunoassay_studies_family,
    bench_microscopy_studies_family,
    bench_pcr_studies_family,
    bench_serology_studies_family,
)


def test_w1257_family_benches() -> None:
    for fn in (
        bench_immunoassay_studies_family,
        bench_pcr_studies_family,
        bench_serology_studies_family,
        bench_culture_studies_family,
        bench_microscopy_studies_family,
        bench_flow_cytometry_studies_family,
    ):
        out = fn()
        assert len(out) == 1
        ((k, v),) = out.items()
        assert k.startswith("synthetic_")
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0
