import pytest

from quant_fund.research import benches_w1471


@pytest.mark.parametrize(
    "fam",
    [
        "bench_crocus_qa_studies_family",
        "bench_daffodil_qa_studies_family",
        "bench_daisy_qa_studies_family",
        "bench_foxglove_qa_studies_family",
        "bench_iris_qa_studies_family",
        "bench_poppy_qa_studies_family",
    ],
)
def test_benches_w1471(fam):
    out = getattr(benches_w1471, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
