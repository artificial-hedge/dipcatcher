import pytest


@pytest.mark.parametrize(
    "name",
    [
        "manatee_qa_studies",
        "orca_qa_studies",
        "narwhal_qa_studies",
        "otter_qa_studies",
        "walrus_qa_studies",
        "beluga_qa_studies",
    ],
)
def test_w1449_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
