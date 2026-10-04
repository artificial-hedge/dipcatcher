import pytest

from quant_fund.research import benches_w1666


@pytest.mark.parametrize(
    "fam",
    [
        "bench_agta_qa_studies_family",
        "bench_berberoka_qa_studies_family",
        "bench_bungisngis_qa_studies_family",
        "bench_dalaketnon_qa_studies_family",
        "bench_ekek_qa_studies_family",
        "bench_engkanto_qa_studies_family",
    ],
)
def test_benches_w1666(fam):
    out = getattr(benches_w1666, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
