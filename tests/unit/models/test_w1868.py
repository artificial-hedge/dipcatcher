import pytest


@pytest.mark.parametrize(
    "name",
    [
        "hadad_punic_qa_studies",
        "reshef_punic_qa_studies",
        "anat_punic_qa_studies",
        "el_punic_qa_studies",
        "moloch_punic_qa_studies",
        "carthage_punic_qa_studies",
    ],
)
def test_w1868_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
