import pytest

from quant_fund.research import benches_w1861


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bedivere_qa_studies_family",
        "bench_galahad_qa_studies_family",
        "bench_gawain_qa_studies_family",
        "bench_lancelot_qa_studies_family",
        "bench_percival_qa_studies_family",
        "bench_tristan_qa_studies_family",
    ],
)
def test_benches_w1861(fam):
    out = getattr(benches_w1861, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
