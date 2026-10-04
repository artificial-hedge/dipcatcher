import pytest


@pytest.mark.parametrize(
    "name",
    [
        "mayahuel_qa_studies",
        "yaotl_qa_studies",
        "teteoinnan_qa_studies",
        "oyohualli_qa_studies",
        "quetzalli_qa_studies",
        "cihuacoatl_qa_studies",
    ],
)
def test_w1684_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
