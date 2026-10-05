import pytest


@pytest.mark.parametrize(
    "name",
    [
        "zilalsen_qa_studies",
        "hemmi_qa_studies",
        "maziun_qa_studies",
        "tissardal_qa_studies",
        "aewan_qa_studies",
        "banguilet_qa_studies",
    ],
)
def test_w1883_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
