from ..state import GameState
from ..types import Role, Team


def resolve_winners(state: GameState) -> list[Team]:
    """Evaluate win conditions over current_role.

    Adopted rules (standard ONUW):
      - Village: at least one Werewolf is killed. If no Werewolves are in
        play at all, Village wins only if nobody dies.
      - Werewolf team: at least one Werewolf in play AND no Werewolf is
        killed.
      - Tanner: solo team. Wins iff Tanner is killed. Co-wins with others.
      - Minion edge case: when all Werewolves are in the center, the Minion
        wins iff no non-Minion player dies. (RULE NOTE: easy to flip later.)
    """
    all_current = [p.current_role for p in state.players.values()]
    dead_current = [p.current_role for p in state.players.values() if p.is_dead]

    no_ww_in_play = Role.WEREWOLF not in all_current
    werewolf_killed = Role.WEREWOLF in dead_current
    tanner_killed = Role.TANNER in dead_current
    nobody_died = len(state.deaths) == 0
    minion_in_play = Role.MINION in all_current
    non_minion_died = any(
        p.is_dead and p.current_role != Role.MINION
        for p in state.players.values()
    )

    winners: list[Team] = []

    if tanner_killed:
        winners.append(Team.TANNER)

    if werewolf_killed:
        winners.append(Team.VILLAGE)
    elif no_ww_in_play and nobody_died:
        winners.append(Team.VILLAGE)

    if not no_ww_in_play and not werewolf_killed:
        winners.append(Team.WEREWOLF)

    if no_ww_in_play and minion_in_play and not non_minion_died:
        if Team.WEREWOLF not in winners:
            winners.append(Team.WEREWOLF)

    return winners