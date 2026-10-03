"""Generated skill wrapper for 'session-book'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import SkillExtension

MODULE = SkillExtension(
    kind="skill",
    owner="session-book",
    references=owner_references("skill", "session-book"),
    module=__name__,
)
