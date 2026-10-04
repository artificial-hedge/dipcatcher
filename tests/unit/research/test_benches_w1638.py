import pytest

from quant_fund.research import benches_w1638


@pytest.mark.parametrize(
    "fam",
    [
        "bench_akaname_qa_studies_family",
        "bench_hitodama_qa_studies_family",
        "bench_ittanmomen_qa_studies_family",
        "bench_nurikabe_qa_studies_family",
        "bench_shikigami_qa_studies_family",
        "bench_ubume_qa_studies_family",
    ],
)
def test_benches_w1638(fam):
    out = getattr(benches_w1638, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
