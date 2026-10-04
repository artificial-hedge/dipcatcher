import pytest

from quant_fund.research import benches_w1311


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aegis_studies_family",
        "bench_air_bench_studies_family",
        "bench_overkill_studies_family",
        "bench_salad_bench_studies_family",
        "bench_sorry_bench_studies_family",
        "bench_wildguard_studies_family",
    ],
)
def test_benches_w1311(fam):
    out = getattr(benches_w1311, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
