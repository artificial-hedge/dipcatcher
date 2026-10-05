import pytest

from quant_fund.research import benches_w1955


@pytest.mark.parametrize(
    "fam",
    [
        "bench_amdusias_qa_studies_family",
        "bench_andromalius_qa_studies_family",
        "bench_dantalion_qa_studies_family",
        "bench_decarabia_qa_studies_family",
        "bench_malphas_qa_studies_family",
        "bench_seere_qa_studies_family",
    ],
)
def test_benches_w1955(fam):
    out = getattr(benches_w1955, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
