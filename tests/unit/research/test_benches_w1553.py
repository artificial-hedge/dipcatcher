import pytest

from quant_fund.research import benches_w1553


@pytest.mark.parametrize(
    "fam",
    [
        "bench_dart_frog_qa_studies_family",
        "bench_horned_frog_qa_studies_family",
        "bench_leopard_frog_qa_studies_family",
        "bench_spring_peeper_qa_studies_family",
        "bench_treefrog_qa_studies_family",
        "bench_wood_frog_qa_studies_family",
    ],
)
def test_benches_w1553(fam):
    out = getattr(benches_w1553, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
