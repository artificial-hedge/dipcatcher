import pytest

from quant_fund.research import benches_w1934


@pytest.mark.parametrize(
    "fam",
    [
        "bench_cherufe_qa_studies_family",
        "bench_chonchon_qa_studies_family",
        "bench_colo_colo_qa_studies_family",
        "bench_kalku_qa_studies_family",
        "bench_peuchen_qa_studies_family",
        "bench_wekufe_qa_studies_family",
    ],
)
def test_benches_w1934(fam):
    out = getattr(benches_w1934, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
