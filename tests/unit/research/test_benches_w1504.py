import pytest

from quant_fund.research import benches_w1504


@pytest.mark.parametrize(
    "fam",
    [
        "bench_murre_qa_studies_family",
        "bench_noddie_qa_studies_family",
        "bench_prion_qa_studies_family",
        "bench_shag_qa_studies_family",
        "bench_skimmer_qa_studies_family",
        "bench_storm_petrel_qa_studies_family",
    ],
)
def test_benches_w1504(fam):
    out = getattr(benches_w1504, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
