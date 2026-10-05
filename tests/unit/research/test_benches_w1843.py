import pytest

from quant_fund.research import benches_w1843


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ashtoreth_qa_studies_family",
        "bench_baalzebub_qa_studies_family",
        "bench_dagon2_qa_studies_family",
        "bench_delilah_qa_studies_family",
        "bench_goliath_qa_studies_family",
        "bench_samson_qa_studies_family",
    ],
)
def test_benches_w1843(fam):
    out = getattr(benches_w1843, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
