import pytest

from quant_fund.research import benches_w1869


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ankou_qa_studies_family",
        "bench_gwalarn_qa_studies_family",
        "bench_korrigan_qa_studies_family",
        "bench_mari_morgen_qa_studies_family",
        "bench_tangi_qa_studies_family",
        "bench_yeun_elez_qa_studies_family",
    ],
)
def test_benches_w1869(fam):
    out = getattr(benches_w1869, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
