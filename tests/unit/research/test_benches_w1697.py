import pytest

from quant_fund.research import benches_w1697


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aoqin_qa_studies_family",
        "bench_guandi_qa_studies_family",
        "bench_houyi_qa_studies_family",
        "bench_wenchang_qa_studies_family",
        "bench_yutu_qa_studies_family",
        "bench_zao_qa_studies_family",
    ],
)
def test_benches_w1697(fam):
    out = getattr(benches_w1697, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
