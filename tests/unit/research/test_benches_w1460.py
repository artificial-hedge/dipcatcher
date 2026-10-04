import pytest

from quant_fund.research import benches_w1460


@pytest.mark.parametrize(
    "fam",
    [
        "bench_baboon_qa_studies_family",
        "bench_elephant_qa_studies_family",
        "bench_gazelle_qa_studies_family",
        "bench_giraffe_qa_studies_family",
        "bench_wildebeest_qa_studies_family",
        "bench_zebra_qa_studies_family",
    ],
)
def test_benches_w1460(fam):
    out = getattr(benches_w1460, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
