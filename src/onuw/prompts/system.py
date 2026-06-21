from ..state import PlayerState
from .rules import (
    GAME_RULES_BLOCK,
    OUTPUT_FORMAT_PREAMBLE,
    ROLE_ABILITY_BLOCKS,
    SYSTEM_PREAMBLE,
    WIN_CONDITIONS_BLOCK,
)


def _persona_block(persona: str | None) -> str:
    if not persona:
        return ""
    return (
        "== YOUR PERSONA ==\n"
        f"{persona}\n"
        "Stay in character; this shapes how you speak and reason."
    )


def build_system_prompt(player: PlayerState, persona: str | None) -> str:
    parts = [
        SYSTEM_PREAMBLE,
        GAME_RULES_BLOCK,
        WIN_CONDITIONS_BLOCK,
        ROLE_ABILITY_BLOCKS[player.original_role],
        _persona_block(persona),
        OUTPUT_FORMAT_PREAMBLE,
    ]
    return "\n\n".join(p for p in parts if p)