import pytest

from quant_fund.research import benches_w1319


@pytest.mark.parametrize(
    "fam",
    [
        "bench_math_bench_studies_family",
        "bench_multirc_studies_family",
        "bench_ninco_studies_family",
        "bench_objectnet_studies_family",
        "bench_ood_bench_studies_family",
        "bench_wild_bench_studies_family",
    ],
)
def test_benches_w1319(fam):
    out = getattr(benches_w1319, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
