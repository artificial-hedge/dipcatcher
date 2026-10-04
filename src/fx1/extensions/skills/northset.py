"""Generated skill wrapper for 'northset'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import SkillExtension

MODULE = SkillExtension(
    kind="skill",
    owner="northset",
    references=owner_references("skill", "northset"),
    module=__name__,
)
