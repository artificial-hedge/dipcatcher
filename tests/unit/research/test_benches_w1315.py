import pytest

from quant_fund.research import benches_w1315


@pytest.mark.parametrize(
    "fam",
    [
        "bench_imagenet_a_studies_family",
        "bench_imagenet_e_studies_family",
        "bench_imagenet_o_studies_family",
        "bench_imagenet_sketch_studies_family",
        "bench_imagenet_v2_studies_family",
        "bench_stylized_studies_family",
    ],
)
def test_benches_w1315(fam):
    out = getattr(benches_w1315, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
