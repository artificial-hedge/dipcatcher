import pytest

from quant_fund.research import benches_w1584


@pytest.mark.parametrize(
    "fam",
    [
        "bench_cottontail_qa_studies_family",
        "bench_hare_qa_studies_family",
        "bench_hedgehog_qa_studies_family",
        "bench_hyrax_qa_studies_family",
        "bench_jackrabbit_qa_studies_family",
        "bench_pika_qa_studies_family",
    ],
)
def test_benches_w1584(fam):
    out = getattr(benches_w1584, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
