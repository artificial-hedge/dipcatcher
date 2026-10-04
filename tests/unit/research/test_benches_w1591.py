import pytest

from quant_fund.research import benches_w1591


@pytest.mark.parametrize(
    "fam",
    [
        "bench_capuchin_qa_studies_family",
        "bench_saki_qa_studies_family",
        "bench_squirrel_monkey_qa_studies_family",
        "bench_titi_qa_studies_family",
        "bench_uakari_qa_studies_family",
        "bench_woolly_qa_studies_family",
    ],
)
def test_benches_w1591(fam):
    out = getattr(benches_w1591, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
