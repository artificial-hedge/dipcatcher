import pytest

from quant_fund.research import benches_w1941


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ahp_qa_studies_family",
        "bench_arak_qa_studies_family",
        "bench_boramei_qa_studies_family",
        "bench_kmoch_qa_studies_family",
        "bench_mrenh_kongveal_qa_studies_family",
        "bench_neak_ta_qa_studies_family",
    ],
)
def test_benches_w1941(fam):
    out = getattr(benches_w1941, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
