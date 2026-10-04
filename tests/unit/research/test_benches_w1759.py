import pytest

from quant_fund.research import benches_w1759


@pytest.mark.parametrize(
    "fam",
    [
        "bench_citlali_qa_studies_family",
        "bench_malinal_qa_studies_family",
        "bench_metzli_qa_studies_family",
        "bench_tepoz_qa_studies_family",
        "bench_tonaca_qa_studies_family",
        "bench_xochipilli_qa_studies_family",
    ],
)
def test_benches_w1759(fam):
    out = getattr(benches_w1759, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
