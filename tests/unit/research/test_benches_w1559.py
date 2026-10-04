import pytest

from quant_fund.research import benches_w1559


@pytest.mark.parametrize(
    "fam",
    [
        "bench_anchovy_qa_studies_family",
        "bench_bonito_qa_studies_family",
        "bench_herring_qa_studies_family",
        "bench_kingfish_qa_studies_family",
        "bench_mackerel_qa_studies_family",
        "bench_sardine_qa_studies_family",
    ],
)
def test_benches_w1559(fam):
    out = getattr(benches_w1559, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
