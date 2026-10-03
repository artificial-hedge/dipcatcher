"""Generated skill wrapper for 'tearsheet'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import SkillExtension

MODULE = SkillExtension(
    kind="skill",
    owner="tearsheet",
    references=owner_references("skill", "tearsheet"),
    module=__name__,
)
