import pytest

from quant_fund.research import benches_w1830


@pytest.mark.parametrize(
    "fam",
    [
        "bench_apali2_qa_studies_family",
        "bench_arinniti2_qa_studies_family",
        "bench_kumarbi3_qa_studies_family",
        "bench_siyum2_qa_studies_family",
        "bench_wulukanni2_qa_studies_family",
        "bench_zintuhi2_qa_studies_family",
    ],
)
def test_benches_w1830(fam):
    out = getattr(benches_w1830, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
