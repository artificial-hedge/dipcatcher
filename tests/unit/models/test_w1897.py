import pytest


@pytest.mark.parametrize(
    "name",
    [
        "lamia_qa_studies",
        "mormo_qa_studies",
        "strix_qa_studies",
        "striga_qa_studies",
        "vrykolakas_qa_studies",
        "nachzehrer_qa_studies",
    ],
)
def test_w1897_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
