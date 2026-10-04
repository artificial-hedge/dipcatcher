import pytest

from quant_fund.research import benches_w1478


@pytest.mark.parametrize(
    "fam",
    [
        "bench_antelope_qa_studies_family",
        "bench_eland_qa_studies_family",
        "bench_impala_qa_studies_family",
        "bench_kudu_qa_studies_family",
        "bench_oryx_qa_studies_family",
        "bench_springbok_qa_studies_family",
    ],
)
def test_benches_w1478(fam):
    out = getattr(benches_w1478, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
