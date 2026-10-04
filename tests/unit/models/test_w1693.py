import pytest


@pytest.mark.parametrize(
    "name",
    [
        "lamassu_qa_studies",
        "shedu_qa_studies",
        "asag_qa_studies",
        "edimmu_qa_studies",
        "galla_qa_studies",
        "utukku_qa_studies",
    ],
)
def test_w1693_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
