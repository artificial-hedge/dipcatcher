import pytest

from quant_fund.research import benches_w1643


@pytest.mark.parametrize(
    "fam",
    [
        "bench_centaur_2_qa_studies_family",
        "bench_gryphon_qa_studies_family",
        "bench_harpy_2_qa_studies_family",
        "bench_hippogryph_qa_studies_family",
        "bench_minotaur_2_qa_studies_family",
        "bench_satyr_2_qa_studies_family",
    ],
)
def test_benches_w1643(fam):
    out = getattr(benches_w1643, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
