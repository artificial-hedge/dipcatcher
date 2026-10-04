import pytest

from quant_fund.research import benches_w1675


@pytest.mark.parametrize(
    "fam",
    [
        "bench_chaneque_qa_studies_family",
        "bench_cihuateteo_qa_studies_family",
        "bench_nagual_qa_studies_family",
        "bench_tlalocan_qa_studies_family",
        "bench_tzitzimitl_qa_studies_family",
        "bench_xiuhcoatl_qa_studies_family",
    ],
)
def test_benches_w1675(fam):
    out = getattr(benches_w1675, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
