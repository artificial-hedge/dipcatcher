import pytest


@pytest.mark.parametrize(
    "name",
    [
        "arthur_qa_studies",
        "morgan_qa_studies",
        "vivien_qa_studies",
        "gorlois_qa_studies",
        "elaine_qa_studies",
        "igraine_qa_studies",
    ],
)
def test_w1860_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
