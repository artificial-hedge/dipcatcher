import pytest

from quant_fund.research import benches_w1775


@pytest.mark.parametrize(
    "fam",
    [
        "bench_inari_qa_studies_family",
        "bench_kaguya_qa_studies_family",
        "bench_momotaro_qa_studies_family",
        "bench_shichifukujin_qa_studies_family",
        "bench_takemikazuchi_qa_studies_family",
        "bench_urashima_qa_studies_family",
    ],
)
def test_benches_w1775(fam):
    out = getattr(benches_w1775, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
