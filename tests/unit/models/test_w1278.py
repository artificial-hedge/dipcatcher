import pytest


@pytest.mark.parametrize(
    "name",
    [
        "chunked_prefill_studies",
        "continuous_batching_studies",
        "disaggregated_serving_studies",
        "early_exit_studies",
        "prefix_caching_studies",
        "tensor_parallel_studies",
    ],
)
def test_w1278_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
