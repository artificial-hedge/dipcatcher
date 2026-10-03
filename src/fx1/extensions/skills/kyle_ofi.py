"""Generated skill wrapper for 'kyle-ofi'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import SkillExtension

MODULE = SkillExtension(
    kind="skill",
    owner="kyle-ofi",
    references=owner_references("skill", "kyle-ofi"),
    module=__name__,
)
