import pytest


@pytest.mark.parametrize(
    "name",
    [
        "yucahu_qa_studies",
        "atabei_qa_studies",
        "deminan_qa_studies",
        "juracan_qa_studies",
        "boinayel_qa_studies",
        "karacarol_qa_studies",
    ],
)
def test_w1731_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
