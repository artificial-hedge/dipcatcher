import pytest

from quant_fund.research import benches_w1621


@pytest.mark.parametrize(
    "fam",
    [
        "bench_cave_beetle_qa_studies_family",
        "bench_cave_cricket_qa_studies_family",
        "bench_cave_fish_qa_studies_family",
        "bench_mudpuppy_qa_studies_family",
        "bench_olm_qa_studies_family",
        "bench_troglobite_qa_studies_family",
    ],
)
def test_benches_w1621(fam):
    out = getattr(benches_w1621, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
