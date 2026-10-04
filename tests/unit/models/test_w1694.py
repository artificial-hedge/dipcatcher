import pytest


@pytest.mark.parametrize(
    "name",
    [
        "karibusa_qa_studies",
        "sokoy_qa_studies",
        "duwende_qa_studies",
        "mambabarang_qa_studies",
        "mangkukulam_qa_studies",
        "tiktik_qa_studies",
    ],
)
def test_w1694_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
