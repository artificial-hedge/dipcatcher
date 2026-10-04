import pytest

from quant_fund.research import benches_w1639


@pytest.mark.parametrize(
    "fam",
    [
        "bench_dorotabo_qa_studies_family",
        "bench_kitsune_3_qa_studies_family",
        "bench_tanuki_3_qa_studies_family",
        "bench_tengu_2_qa_studies_family",
        "bench_yukionna_qa_studies_family",
        "bench_zashiki_warashi_qa_studies_family",
    ],
)
def test_benches_w1639(fam):
    out = getattr(benches_w1639, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
