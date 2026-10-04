import pytest

from quant_fund.research import benches_w1704


@pytest.mark.parametrize(
    "fam",
    [
        "bench_adad_qa_studies_family",
        "bench_ashur_qa_studies_family",
        "bench_ishtar_qa_studies_family",
        "bench_nabu_qa_studies_family",
        "bench_shamash_qa_studies_family",
        "bench_sin_qa_studies_family",
    ],
)
def test_benches_w1704(fam):
    out = getattr(benches_w1704, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
