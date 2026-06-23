from collections import Counter

from ..state import PlayerState
from ..types import Role
from .rules import (
    GAME_RULES_BLOCK,
    OUTPUT_FORMAT_PREAMBLE,
    ROLE_ABILITY_BLOCKS,
    SYSTEM_PREAMBLE,
    WIN_CONDITIONS_BLOCK,
)


def _table_block(role_pool: list[Role] | None) -> str:
    """Public deck composition — what cards exist in this game."""
    if not role_pool:
        return ""
    counts = Counter(r.value for r in role_pool)
    dealt = len(role_pool) - 3
    lines = [
        "== TABLE ==",
        f"Deck ({len(role_pool)} cards: {dealt} dealt to players, 3 in the center):",
    ]
    for role_name, n in sorted(counts.items()):
        lines.append(f"  - {n}x {role_name}")
    return "\n".join(lines)


def _roles_in_game_block(role_pool: list[Role] | None) -> str:
    """Ability text for every UNIQUE role in the pool — so the agent can
    evaluate any role another player might claim, not just its own."""
    if not role_pool:
        return ""
    seen: list[Role] = []
    for r in role_pool:
        if r not in seen:
            seen.append(r)
    if not seen:
        return ""
    blocks = ["== ROLES IN THIS GAME =="]
    for r in seen:
        block = ROLE_ABILITY_BLOCKS.get(r)
        if block:
            blocks.append(block)
    return "\n\n".join(blocks)


def build_system_prompt(
    player: PlayerState,
    role_pool: list[Role] | None = None,
) -> str:
    # When role_pool is given, the ROLES IN THIS GAME block already
    # covers every role at the table (including the player's own), so
    # the separate "YOUR ROLE" block is omitted. Fall back to the
    # per-player block when no pool was supplied (older test callsites).
    own_role_block = (
        "" if role_pool else ROLE_ABILITY_BLOCKS[player.original_role]
    )
    parts = [
        SYSTEM_PREAMBLE,
        GAME_RULES_BLOCK,
        WIN_CONDITIONS_BLOCK,
        _table_block(role_pool),
        _roles_in_game_block(role_pool),
        own_role_block,
        OUTPUT_FORMAT_PREAMBLE,
    ]
    return "\n\n".join(p for p in parts if p)