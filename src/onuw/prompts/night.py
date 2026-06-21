from ..memory import PlayerMemory
from ..types import Role


def _others(seat_order: list[str], self_id: str) -> str:
    return ", ".join(p for p in seat_order if p != self_id)


def build_night_task(role: Role, self_id: str, seat_order: list[str]) -> str:
    """Phase-specific task block appended to the rendered memory context."""
    if role == Role.WEREWOLF:
        return (
            "== YOUR TURN: WEREWOLF (solo) ==\n"
            "You are the only Werewolf in play. Choose one center card index "
            "(0, 1, or 2) to peek at.\n"
            "Respond with JSON ONLY:\n"
            '{"action": "peek_center", "index": <0|1|2>}'
        )
    if role == Role.SEER:
        return (
            "== YOUR TURN: SEER ==\n"
            "Choose EITHER to view one other player's card OR to view two of "
            "the three center cards.\n"
            f"Valid player targets: {_others(seat_order, self_id)}.\n"
            "Respond with JSON ONLY matching one of:\n"
            '  {"action": "view_player", "target": "<player_id>"}\n'
            '  {"action": "view_center", "indices": [<i>, <j>]}    '
            "# i,j in {0,1,2}, distinct"
        )
    if role == Role.ROBBER:
        return (
            "== YOUR TURN: ROBBER ==\n"
            "Choose one other player to rob. You will swap your card with "
            "theirs and then look at the card you stole.\n"
            f"Valid targets: {_others(seat_order, self_id)}.\n"
            "Respond with JSON ONLY:\n"
            '{"action": "rob", "target": "<player_id>"}'
        )
    if role == Role.TROUBLEMAKER:
        return (
            "== YOUR TURN: TROUBLEMAKER ==\n"
            "Choose two OTHER players (not yourself); their cards will be "
            "swapped. You will NOT see them.\n"
            f"Valid targets: {_others(seat_order, self_id)}.\n"
            "Respond with JSON ONLY:\n"
            '{"action": "swap", "target_a": "<player_id>", '
            '"target_b": "<player_id>"}    # distinct, neither is you'
        )
    if role == Role.DRUNK:
        return (
            "== YOUR TURN: DRUNK ==\n"
            "Choose one center card index (0, 1, or 2). Your card will be "
            "swapped with that center card. You will NOT see your new role.\n"
            "Respond with JSON ONLY:\n"
            '{"action": "swap_center", "index": <0|1|2>}'
        )
    raise ValueError(f"No interactive night task defined for role {role}")


def build_night_prompt(
    memory: PlayerMemory,
    role: Role,
    self_id: str,
    seat_order: list[str],
) -> str:
    return memory.to_prompt_context("night") + "\n\n" + build_night_task(
        role, self_id, seat_order
    )