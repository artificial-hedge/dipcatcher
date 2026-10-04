import pytest

from quant_fund.research import benches_w1574


@pytest.mark.parametrize(
    "fam",
    [
        "bench_gibbon_qa_studies_family",
        "bench_langur_qa_studies_family",
        "bench_lemur_qa_studies_family",
        "bench_macaque_qa_studies_family",
        "bench_marmoset_qa_studies_family",
        "bench_tamarin_qa_studies_family",
    ],
)
def test_benches_w1574(fam):
    out = getattr(benches_w1574, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
