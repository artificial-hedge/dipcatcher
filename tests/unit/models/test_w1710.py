import pytest


@pytest.mark.parametrize(
    "name",
    [
        "baal_qa_studies",
        "anat_qa_studies",
        "asherah_qa_studies",
        "lotan_qa_studies",
        "yam_qa_studies",
        "mot_qa_studies",
    ],
)
def test_w1710_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
