import pytest

from quant_fund.research import benches_w1900


@pytest.mark.parametrize(
    "fam",
    [
        "bench_enenra_qa_studies_family",
        "bench_goryo_qa_studies_family",
        "bench_kiyohime_qa_studies_family",
        "bench_kodama_shirakawa_qa_studies_family",
        "bench_nure_onna_qa_studies_family",
        "bench_yurei_muzen_qa_studies_family",
    ],
)
def test_benches_w1900(fam):
    out = getattr(benches_w1900, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
