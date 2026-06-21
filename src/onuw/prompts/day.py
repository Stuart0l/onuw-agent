from ..memory import PlayerMemory


def build_day_speech_task(
    round_idx: int, total_rounds: int, max_chars: int = 600
) -> str:
    return (
        f"== YOUR TURN: DAY DISCUSSION (round {round_idx + 1} of {total_rounds}) ==\n"
        "It is your turn to speak publicly. Every other player will read your "
        "statement verbatim. You may lie, claim a role, share information, ask "
        "questions, or stay neutral. Aim for 1-4 sentences "
        f"(max {max_chars} characters).\n"
        "Respond with JSON ONLY:\n"
        '{"speech": "<your public statement>"}'
    )


def build_day_speech_prompt(
    memory: PlayerMemory,
    round_idx: int,
    total_rounds: int,
    max_chars: int = 600,
) -> str:
    return memory.to_prompt_context("day") + "\n\n" + build_day_speech_task(
        round_idx, total_rounds, max_chars
    )