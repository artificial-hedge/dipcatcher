import pytest


@pytest.mark.parametrize(
    "name",
    [
        "sungrebe_qa_studies",
        "takhe_qa_studies",
        "swamphen_qa_studies",
        "sora_qa_studies",
        "flufftail_qa_studies",
        "corncrake_qa_studies",
    ],
)
def test_w1546_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
