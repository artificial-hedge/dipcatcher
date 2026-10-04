import pytest

from quant_fund.research import benches_w1890


@pytest.mark.parametrize(
    "fam",
    [
        "bench_daeva_qa_studies_family",
        "bench_div_qa_studies_family",
        "bench_fravashi_qa_studies_family",
        "bench_khshathra_qa_studies_family",
        "bench_pairika_qa_studies_family",
        "bench_spenta_mainyu_qa_studies_family",
    ],
)
def test_benches_w1890(fam):
    out = getattr(benches_w1890, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
