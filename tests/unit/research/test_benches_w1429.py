import pytest

from quant_fund.research import benches_w1429


@pytest.mark.parametrize(
    "fam",
    [
        "bench_atom_qa_studies_family",
        "bench_electron_qa_studies_family",
        "bench_ion_qa_studies_family",
        "bench_molecule_qa_studies_family",
        "bench_neutron_qa_studies_family",
        "bench_photon_qa_studies_family",
    ],
)
def test_benches_w1429(fam):
    out = getattr(benches_w1429, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
