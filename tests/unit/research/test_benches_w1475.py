import pytest

from quant_fund.research import benches_w1475


@pytest.mark.parametrize(
    "fam",
    [
        "bench_cicada_qa_studies_family",
        "bench_dragonfly_qa_studies_family",
        "bench_grasshopper_qa_studies_family",
        "bench_ladybug_qa_studies_family",
        "bench_mantis_qa_studies_family",
        "bench_scorpion_qa_studies_family",
    ],
)
def test_benches_w1475(fam):
    out = getattr(benches_w1475, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
