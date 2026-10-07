"""Wave-1429 bench adapters: particle canon (SYNTHETIC only)."""

from quant_fund.models import (
    atom_qa_studies,
    electron_qa_studies,
    ion_qa_studies,
    molecule_qa_studies,
    neutron_qa_studies,
    photon_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14290


def _finite_blob(blob):
    if not (isinstance(blob, dict) and blob):
        raise ValueError("bench blob must be a non-empty dict")
    for k, v in blob.items():
        if not k.startswith("synthetic_"):
            raise ValueError(f"non-synthetic metric key {k}")
        if k in _FORBIDDEN:
            raise ValueError(f"forbidden metric key {k}")
        if not (isinstance(v, float) and 0.0 <= v <= 1.0):
            raise ValueError(f"metric {k} is not a [0,1] float")
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_atom_qa_studies_family(seed: int = _SEED + 0):
    """atom_qa_studies: synthetic correctness bench."""
    return _finite_blob(atom_qa_studies.bench_atom_qa_studies(seed))


def bench_electron_qa_studies_family(seed: int = _SEED + 1):
    """electron_qa_studies: synthetic correctness bench."""
    return _finite_blob(electron_qa_studies.bench_electron_qa_studies(seed))


def bench_ion_qa_studies_family(seed: int = _SEED + 2):
    """ion_qa_studies: synthetic correctness bench."""
    return _finite_blob(ion_qa_studies.bench_ion_qa_studies(seed))


def bench_molecule_qa_studies_family(seed: int = _SEED + 3):
    """molecule_qa_studies: synthetic correctness bench."""
    return _finite_blob(molecule_qa_studies.bench_molecule_qa_studies(seed))


def bench_neutron_qa_studies_family(seed: int = _SEED + 4):
    """neutron_qa_studies: synthetic correctness bench."""
    return _finite_blob(neutron_qa_studies.bench_neutron_qa_studies(seed))


def bench_photon_qa_studies_family(seed: int = _SEED + 5):
    """photon_qa_studies: synthetic correctness bench."""
    return _finite_blob(photon_qa_studies.bench_photon_qa_studies(seed))
