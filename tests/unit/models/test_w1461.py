import pytest


@pytest.mark.parametrize(
    "name",
    [
        "jaguar_qa_studies",
        "orangutan_qa_studies",
        "macaw_qa_studies",
        "sloth_qa_studies",
        "toucan_qa_studies",
        "gorilla_qa_studies",
    ],
)
def test_w1461_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
