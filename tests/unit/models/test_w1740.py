import pytest


@pytest.mark.parametrize(
    "name",
    [
        "mot_qa_studies",
        "yam_qa_studies",
        "anat_qa_studies",
        "astarte_qa_studies",
        "resheph_qa_studies",
        "kothar_qa_studies",
    ],
)
def test_w1740_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
