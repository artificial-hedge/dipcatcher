import pytest

from quant_fund.research import benches_w1405


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ambi_qa_studies_family",
        "bench_audio_qa_lite_studies_family",
        "bench_avsd_lite_studies_family",
        "bench_clotho_qa_studies_family",
        "bench_esc_qa_studies_family",
        "bench_music_avqa_studies_family",
    ],
)
def test_benches_w1405(fam):
    out = getattr(benches_w1405, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
