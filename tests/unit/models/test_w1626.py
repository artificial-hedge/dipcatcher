import pytest


@pytest.mark.parametrize(
    "name",
    [
        "bondolo_qa_studies",
        "mittermeier_qa_studies",
        "madame_berthe_qa_studies",
        "northern_qa_studies",
        "southern_qa_studies",
        "western_qa_studies",
    ],
)
def test_w1626_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
