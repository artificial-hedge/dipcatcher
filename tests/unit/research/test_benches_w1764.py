import pytest

from quant_fund.research import benches_w1764


@pytest.mark.parametrize(
    "fam",
    [
        "bench_eileithyia_qa_studies_family",
        "bench_iris_qa_studies_family",
        "bench_leto_qa_studies_family",
        "bench_nemesis_qa_studies_family",
        "bench_nike_qa_studies_family",
        "bench_tyche_qa_studies_family",
    ],
)
def test_benches_w1764(fam):
    out = getattr(benches_w1764, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
