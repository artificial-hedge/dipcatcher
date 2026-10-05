import pytest


@pytest.mark.parametrize(
    "name",
    [
        "kumarbi3_qa_studies",
        "arinniti2_qa_studies",
        "wulukanni2_qa_studies",
        "siyum2_qa_studies",
        "zintuhi2_qa_studies",
        "apali2_qa_studies",
    ],
)
def test_w1830_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
