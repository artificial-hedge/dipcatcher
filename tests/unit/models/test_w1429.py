import pytest


@pytest.mark.parametrize(
    "name",
    [
        "electron_qa_studies",
        "molecule_qa_studies",
        "ion_qa_studies",
        "neutron_qa_studies",
        "photon_qa_studies",
        "atom_qa_studies",
    ],
)
def test_w1429_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
