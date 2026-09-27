"""Shared pytest configuration.

Hypothesis profiles: the ``ci`` profile de-randomizes generation so CI runs are
seed-stable across machines (a falsifying example found once re-runs
identically everywhere, instead of living only in one host's example DB).
CI sets HYPOTHESIS_PROFILE=ci (.github/workflows/ci.yml); local runs keep
Hypothesis's default entropy-driven behavior.
"""

import os

from hypothesis import settings

collect_ignore_glob = ["**/__pycache__/**"]

settings.register_profile(
    "ci",
    settings(derandomize=True, deadline=None),
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
