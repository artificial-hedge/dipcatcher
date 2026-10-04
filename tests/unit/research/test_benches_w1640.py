import pytest

from quant_fund.research import benches_w1640


@pytest.mark.parametrize(
    "fam",
    [
        "bench_byakko_qa_studies_family",
        "bench_genbu_qa_studies_family",
        "bench_kirin_2_qa_studies_family",
        "bench_kohryu_qa_studies_family",
        "bench_seiryu_qa_studies_family",
        "bench_suzaku_qa_studies_family",
    ],
)
def test_benches_w1640(fam):
    out = getattr(benches_w1640, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
