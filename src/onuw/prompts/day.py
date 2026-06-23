from ..types import Role
from .rules import swap_reminder


def build_day_speech_task(
    round_idx: int,
    total_rounds: int,
    max_chars: int = 600,
    dealt_role: Role | None = None,
) -> str:
    prefix = (swap_reminder(dealt_role) + "\n\n") if dealt_role else ""
    return (
        prefix
        + f"== YOUR TURN: DAY DISCUSSION (round {round_idx + 1} of {total_rounds}) ==\n"
        "It is your turn to speak publicly. Every other player will read your "
        "statement verbatim. You may lie, claim a role, share information, ask "
        "questions. Aim for 1-4 sentences "
        f"(max {max_chars} characters).\n"
        "\n"
        "Before crafting your speech, update belief_state with your current "
        "judgment of every player you have an opinion on. ONE LINE per player, "
        "brief. Use it to stop re-deriving who is who from scratch each round.\n"
        "\n"
        "Respond with JSON ONLY:\n"
        '{"belief_state": {"<player_id>": "<one-line current belief>", ...}, '
        '"speech": "<your public statement>"}'
    )
