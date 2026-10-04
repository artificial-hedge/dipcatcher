import pytest


@pytest.mark.parametrize(
    "name",
    [
        "activation_checkpoint_studies",
        "fsdp_sharding_studies",
        "hybrid_parallel_studies",
        "pipeline_schedule_studies",
        "sequence_parallel_studies",
        "zero_optimizer_studies",
    ],
)
def test_w1279_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
