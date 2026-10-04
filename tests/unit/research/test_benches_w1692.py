import pytest

from quant_fund.research import benches_w1692


@pytest.mark.parametrize(
    "fam",
    [
        "bench_alfar_qa_studies_family",
        "bench_draugar_qa_studies_family",
        "bench_hulder_qa_studies_family",
        "bench_muspell_qa_studies_family",
        "bench_svartalf_qa_studies_family",
        "bench_ymir_qa_studies_family",
    ],
)
def test_benches_w1692(fam):
    out = getattr(benches_w1692, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
