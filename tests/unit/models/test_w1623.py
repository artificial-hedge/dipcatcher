import pytest


@pytest.mark.parametrize(
    "name",
    [
        "chukar_qa_studies",
        "wallcreeper_qa_studies",
        "altai_qa_studies",
        "monal_qa_studies",
        "snow_partridge_qa_studies",
        "blood_pheasant_qa_studies",
    ],
)
def test_w1623_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
