import pytest

from quant_fund.research import benches_w1684


@pytest.mark.parametrize(
    "fam",
    [
        "bench_cihuacoatl_qa_studies_family",
        "bench_mayahuel_qa_studies_family",
        "bench_oyohualli_qa_studies_family",
        "bench_quetzalli_qa_studies_family",
        "bench_teteoinnan_qa_studies_family",
        "bench_yaotl_qa_studies_family",
    ],
)
def test_benches_w1684(fam):
    out = getattr(benches_w1684, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
