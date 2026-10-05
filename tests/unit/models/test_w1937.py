import pytest


@pytest.mark.parametrize(
    "name",
    [
        "trauco_qa_studies",
        "invunche_qa_studies",
        "camahueto_qa_studies",
        "caleuche_qa_studies",
        "pincoya_qa_studies",
        "fiura_qa_studies",
    ],
)
def test_w1937_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
