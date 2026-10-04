import pytest

from quant_fund.research import benches_w1589


@pytest.mark.parametrize(
    "fam",
    [
        "bench_porpoise_qa_studies_family",
        "bench_right_whale_qa_studies_family",
        "bench_rissos_qa_studies_family",
        "bench_river_dolphin_qa_studies_family",
        "bench_spinner_qa_studies_family",
        "bench_vaquita_qa_studies_family",
    ],
)
def test_benches_w1589(fam):
    out = getattr(benches_w1589, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
