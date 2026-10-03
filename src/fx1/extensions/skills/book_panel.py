"""Generated skill wrapper for 'book-panel'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import SkillExtension

MODULE = SkillExtension(
    kind="skill",
    owner="book-panel",
    references=owner_references("skill", "book-panel"),
    module=__name__,
)
