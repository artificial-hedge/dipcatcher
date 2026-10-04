import pytest

from quant_fund.research import benches_w1419


@pytest.mark.parametrize(
    "fam",
    [
        "bench_anecdote_qa_studies_family",
        "bench_ballad_qa_studies_family",
        "bench_biography_qa_studies_family",
        "bench_chronicle_qa_studies_family",
        "bench_epic_qa_studies_family",
        "bench_fable_qa_studies_family",
    ],
)
def test_benches_w1419(fam):
    out = getattr(benches_w1419, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
