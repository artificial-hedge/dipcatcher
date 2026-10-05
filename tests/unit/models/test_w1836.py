import pytest


@pytest.mark.parametrize(
    "name",
    [
        "labrandeus2_qa_studies",
        "chrysaoreus2_qa_studies",
        "panamara2_qa_studies",
        "stratios2_qa_studies",
        "axom2_qa_studies",
        "hekate2_qa_studies",
    ],
)
def test_w1836_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
