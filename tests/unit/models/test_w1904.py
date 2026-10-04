import pytest


@pytest.mark.parametrize(
    "name",
    [
        "umibozu_qa_studies",
        "ittan_momen_qa_studies",
        "kasa_obake_qa_studies",
        "mikoshi_nyudo_qa_studies",
        "sunakake_babaa_qa_studies",
        "futakuchi_onna_qa_studies",
    ],
)
def test_w1904_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
