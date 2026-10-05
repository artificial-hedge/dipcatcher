import pytest


@pytest.mark.parametrize(
    "name",
    [
        "masalai_qa_studies",
        "sanguma_qa_studies",
        "tambaran_qa_studies",
        "kaiaimunu_qa_studies",
        "pukaua_qa_studies",
        "adaro_qa_studies",
    ],
)
def test_w1945_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
