import pytest


@pytest.mark.parametrize(
    "name",
    [
        "pele_qa_studies",
        "kane_qa_studies",
        "kanaloa_qa_studies",
        "ku_qa_studies",
        "lono_qa_studies",
        "hina_qa_studies",
    ],
)
def test_w1741_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
