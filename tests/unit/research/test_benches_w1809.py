import pytest

from quant_fund.research import benches_w1809


@pytest.mark.parametrize(
    "fam",
    [
        "bench_brigid2_qa_studies_family",
        "bench_dagda2_qa_studies_family",
        "bench_lugh2_qa_studies_family",
        "bench_manannan2_qa_studies_family",
        "bench_nuada2_qa_studies_family",
        "bench_ogma2_qa_studies_family",
    ],
)
def test_benches_w1809(fam):
    out = getattr(benches_w1809, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
