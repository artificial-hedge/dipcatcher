"""Generated skill wrapper for 'verify-research'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import SkillExtension

MODULE = SkillExtension(
    kind="skill",
    owner="verify-research",
    references=owner_references("skill", "verify-research"),
    module=__name__,
)
