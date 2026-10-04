import pytest

from quant_fund.research import benches_w1349


@pytest.mark.parametrize(
    "fam",
    [
        "bench_cola_lite_studies_family",
        "bench_qnli_lite_studies_family",
        "bench_qqp_lite_studies_family",
        "bench_sst2_lite_studies_family",
        "bench_stsb_lite_studies_family",
        "bench_wnli_lite_studies_family",
    ],
)
def test_benches_w1349(fam):
    out = getattr(benches_w1349, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
