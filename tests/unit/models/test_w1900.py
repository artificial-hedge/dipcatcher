import pytest


@pytest.mark.parametrize(
    "name",
    [
        "enenra_qa_studies",
        "goryo_qa_studies",
        "kiyohime_qa_studies",
        "yurei_muzen_qa_studies",
        "kodama_shirakawa_qa_studies",
        "nure_onna_qa_studies",
    ],
)
def test_w1900_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
