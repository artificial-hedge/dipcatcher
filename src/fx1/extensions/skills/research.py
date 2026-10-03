"""Generated skill wrapper for 'research'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import SkillExtension

MODULE = SkillExtension(
    kind="skill",
    owner="research",
    references=owner_references("skill", "research"),
    module=__name__,
)
