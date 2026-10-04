import pytest

from quant_fund.research import benches_w1774


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bastet_qa_studies_family",
        "bench_hathor_qa_studies_family",
        "bench_nut_qa_studies_family",
        "bench_sekhmet_qa_studies_family",
        "bench_sobek_qa_studies_family",
        "bench_thoth_qa_studies_family",
    ],
)
def test_benches_w1774(fam):
    out = getattr(benches_w1774, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
