"""Generated skill wrapper for 'validate'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import SkillExtension

MODULE = SkillExtension(
    kind="skill",
    owner="validate",
    references=owner_references("skill", "validate"),
    module=__name__,
)
