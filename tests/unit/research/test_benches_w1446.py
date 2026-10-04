import pytest

from quant_fund.research import benches_w1446


@pytest.mark.parametrize(
    "fam",
    [
        "bench_cello_qa_studies_family",
        "bench_drum_qa_studies_family",
        "bench_flute_qa_studies_family",
        "bench_guitar_qa_studies_family",
        "bench_piano_qa_studies_family",
        "bench_violin_qa_studies_family",
    ],
)
def test_benches_w1446(fam):
    out = getattr(benches_w1446, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
