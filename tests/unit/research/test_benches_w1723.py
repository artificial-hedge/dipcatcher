import pytest

from quant_fund.research import benches_w1723


@pytest.mark.parametrize(
    "fam",
    [
        "bench_honir_qa_studies_family",
        "bench_kvasir_qa_studies_family",
        "bench_lodurr_qa_studies_family",
        "bench_mimir_qa_studies_family",
        "bench_ve_qa_studies_family",
        "bench_vili_qa_studies_family",
    ],
)
def test_benches_w1723(fam):
    out = getattr(benches_w1723, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
