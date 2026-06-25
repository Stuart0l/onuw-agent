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
        "Update per_player_hypothesis — for each player you have an "
        "opinion on, pick the SINGLE most likely current role with a "
        "confidence (high/medium/low) and one line of evidence. Mark "
        "source reliability (night observations can be swapped, "
        "speeches can be bluffs).\n"
        "\n"
        "Respond with JSON ONLY:\n"
        "{\n"
        '  "committed_current_role": "<role name>",  // omit if no change\n'
        '  "per_player_hypothesis": {\n'
        '    "<player_id>": {"role": "<role>", "confidence": "high|medium|low", "evidence": "<one line>"},\n'
        "    ...\n"
        "  },\n"
        '  "speech": "<your public statement>"\n'
        "}"
    )
