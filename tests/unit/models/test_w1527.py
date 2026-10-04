import pytest


@pytest.mark.parametrize(
    "name",
    [
        "crustose_qa_studies",
        "oakmoss_qa_studies",
        "fruticose_qa_studies",
        "xanthoria_qa_studies",
        "usnea_qa_studies",
        "foliose_qa_studies",
    ],
)
def test_w1527_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
