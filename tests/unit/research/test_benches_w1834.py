import pytest

from quant_fund.research import benches_w1834


@pytest.mark.parametrize(
    "fam",
    [
        "bench_hebat2_qa_studies_family",
        "bench_kusuh2_qa_studies_family",
        "bench_sarruma2_qa_studies_family",
        "bench_simige2_qa_studies_family",
        "bench_tasmisu2_qa_studies_family",
        "bench_tessub2_qa_studies_family",
    ],
)
def test_benches_w1834(fam):
    out = getattr(benches_w1834, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
