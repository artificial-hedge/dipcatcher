import pytest

from quant_fund.research import benches_w1556


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bristletail_qa_studies_family",
        "bench_pillbug_qa_studies_family",
        "bench_silverfish_qa_studies_family",
        "bench_springtail_qa_studies_family",
        "bench_velvet_worm_qa_studies_family",
        "bench_woodlouse_qa_studies_family",
    ],
)
def test_benches_w1556(fam):
    out = getattr(benches_w1556, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
