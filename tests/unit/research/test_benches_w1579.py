import pytest

from quant_fund.research import benches_w1579


@pytest.mark.parametrize(
    "fam",
    [
        "bench_dikdik_qa_studies_family",
        "bench_grysbok_qa_studies_family",
        "bench_klipspringer_qa_studies_family",
        "bench_rhebok_qa_studies_family",
        "bench_steenbok_qa_studies_family",
        "bench_suni_qa_studies_family",
    ],
)
def test_benches_w1579(fam):
    out = getattr(benches_w1579, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
