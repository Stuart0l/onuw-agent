from ..types import Role
from .rules import swap_reminder


def build_vote_task(
    valid_targets: list[str],
    dealt_role: Role | None = None,
) -> str:
    prefix = (swap_reminder(dealt_role) + "\n\n") if dealt_role else ""
    return (
        prefix
        + "== VOTING PHASE ==\n"
        "Cast your vote for the player you believe should be killed. Votes are "
        "simultaneous; you will not see others' votes until all are revealed.\n"
        f"Valid targets: {', '.join(valid_targets)}.\n"
        "You MAY vote for yourself (e.g. as Tanner).\n"
        "Respond with JSON ONLY:\n"
        '{"vote": "<player_id>", "rationale": "<one short sentence (private)>"}'
    )
