import pytest


@pytest.mark.parametrize(
    "name",
    [
        "brass_qa_studies",
        "amalgam_qa_studies",
        "pewter_qa_studies",
        "solder_qa_studies",
        "nichrome_qa_studies",
        "bronze_qa_studies",
    ],
)
def test_w1530_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
