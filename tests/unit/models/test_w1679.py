import pytest


@pytest.mark.parametrize(
    "name",
    [
        "geirahod_qa_studies",
        "alfheim_qa_studies",
        "troll_qa_studies",
        "huldra_qa_studies",
        "vaetter_qa_studies",
        "bergrisi_qa_studies",
    ],
)
def test_w1679_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
