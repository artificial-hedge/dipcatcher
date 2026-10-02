"""Generated skill wrapper for 'ingest'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import SkillExtension

MODULE = SkillExtension(
    kind="skill",
    owner="ingest",
    references=owner_references("skill", "ingest"),
    module=__name__,
)
