import pytest

from quant_fund.research import benches_w1757


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aisling_qa_studies_family",
        "bench_brigid_qa_studies_family",
        "bench_dagda_qa_studies_family",
        "bench_danu_qa_studies_family",
        "bench_manannan_qa_studies_family",
        "bench_morgen_qa_studies_family",
    ],
)
def test_benches_w1757(fam):
    out = getattr(benches_w1757, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
