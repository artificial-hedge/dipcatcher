import pytest

from quant_fund.research import benches_w1801


@pytest.mark.parametrize(
    "fam",
    [
        "bench_apocatequil2_qa_studies_family",
        "bench_inti2_qa_studies_family",
        "bench_kon2_qa_studies_family",
        "bench_pacamama2_qa_studies_family",
        "bench_supay2_qa_studies_family",
        "bench_viracocha2_qa_studies_family",
    ],
)
def test_benches_w1801(fam):
    out = getattr(benches_w1801, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
