import pytest

from quant_fund.research import benches_w1340


@pytest.mark.parametrize(
    "fam",
    [
        "bench_arc_challenge_studies_family",
        "bench_bio_qa_studies_family",
        "bench_med_qa_studies_family",
        "bench_openbook_qa_studies_family",
        "bench_pubmed_qa_studies_family",
        "bench_sci_q_studies_family",
    ],
)
def test_benches_w1340(fam):
    out = getattr(benches_w1340, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
