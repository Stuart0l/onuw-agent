from ..types import Role
from .rules import thinking_guide_day


def build_day_speech_task(
    round_idx: int,
    total_rounds: int,
    max_chars: int = 600,
    dealt_role: Role | None = None,
    committed_role: Role | None = None,
) -> str:
    prefix = (
        thinking_guide_day(dealt_role, committed_role) + "\n\n"
        if dealt_role
        else ""
    )
    return (
        prefix
        + f"== YOUR TURN: DAY DISCUSSION (round {round_idx + 1} of {total_rounds}) ==\n"
        "It is your turn to speak publicly. Every other player will read your "
        "statement verbatim. You may lie, claim a role, share information, ask "
        "questions. Aim for 1-4 sentences "
        f"(max {max_chars} characters).\n"
        "\n"
        "Update belief_state — ONE LINE per player you have an opinion on. "
        "Combine evidence with your judgment, marking source reliability "
        "(night observations can be swapped, speeches can be bluffs).\n"
        "\n"
        "Respond with JSON ONLY:\n"
        "{\n"
        '  "committed_current_role": "<role name>",  // omit if no change\n'
        '  "belief_state": {"<player_id>": "<one-line current belief>", ...},\n'
        '  "speech": "<your public statement>"\n'
        "}"
    )
