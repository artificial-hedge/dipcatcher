import pytest

from quant_fund.research import benches_w1455


@pytest.mark.parametrize(
    "fam",
    [
        "bench_axolotl_qa_studies_family",
        "bench_bullfrog_qa_studies_family",
        "bench_newt_qa_studies_family",
        "bench_salamander_qa_studies_family",
        "bench_toad_qa_studies_family",
        "bench_tree_frog_qa_studies_family",
    ],
)
def test_benches_w1455(fam):
    out = getattr(benches_w1455, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
