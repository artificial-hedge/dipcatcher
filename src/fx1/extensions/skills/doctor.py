"""Generated skill wrapper for 'doctor'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import SkillExtension

MODULE = SkillExtension(
    kind="skill",
    owner="doctor",
    references=owner_references("skill", "doctor"),
    module=__name__,
)
