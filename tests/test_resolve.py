from onuw.engine.resolve import resolve_winners
from onuw.state import CenterCard, GameState, PlayerState
from onuw.types import Role, Team


def _state(player_roles, dead=None, center_roles=None):
    dead = dead or []
    center_roles = center_roles or [Role.VILLAGER, Role.VILLAGER, Role.VILLAGER]
    players: dict[str, PlayerState] = {}
    seat_order: list[str] = []
    for i, (pid, role) in enumerate(player_roles):
        ps = PlayerState(
            id=pid, name=pid.upper(), seat=i,
            original_role=role, current_role=role,
            is_dead=(pid in dead),
        )
        players[pid] = ps
        seat_order.append(pid)
    return GameState(
        players=players, seat_order=seat_order,
        center=[CenterCard(index=i, role=r) for i, r in enumerate(center_roles)],
        discussion_rounds=3,
        deaths=list(dead),
    )


def test_village_wins_when_a_werewolf_dies():
    state = _state(
        [("p1", Role.WEREWOLF), ("p2", Role.VILLAGER),
         ("p3", Role.SEER), ("p4", Role.VILLAGER)],
        dead=["p1"],
    )
    winners = resolve_winners(state)
    assert Team.VILLAGE in winners
    assert Team.WEREWOLF not in winners


def test_werewolf_team_wins_when_no_werewolf_dies():
    state = _state(
        [("p1", Role.WEREWOLF), ("p2", Role.VILLAGER),
         ("p3", Role.SEER), ("p4", Role.VILLAGER)],
        dead=["p3"],
    )
    winners = resolve_winners(state)
    assert Team.WEREWOLF in winners
    assert Team.VILLAGE not in winners


def test_nobody_dies_with_werewolf_in_play_werewolf_wins():
    state = _state(
        [("p1", Role.WEREWOLF), ("p2", Role.VILLAGER),
         ("p3", Role.SEER), ("p4", Role.VILLAGER)],
        dead=[],
    )
    winners = resolve_winners(state)
    assert Team.WEREWOLF in winners
    assert Team.VILLAGE not in winners


def test_tanner_wins_when_tanner_dies_solo_team():
    state = _state(
        [("p1", Role.TANNER), ("p2", Role.VILLAGER),
         ("p3", Role.WEREWOLF), ("p4", Role.VILLAGER)],
        dead=["p1"],
    )
    winners = resolve_winners(state)
    assert Team.TANNER in winners
    # Werewolf team still wins (no werewolf died)
    assert Team.WEREWOLF in winners
    assert Team.VILLAGE not in winners


def test_tanner_and_village_cowin_when_both_tanner_and_werewolf_die():
    state = _state(
        [("p1", Role.TANNER), ("p2", Role.WEREWOLF),
         ("p3", Role.VILLAGER), ("p4", Role.SEER)],
        dead=["p1", "p2"],
    )
    winners = resolve_winners(state)
    assert Team.TANNER in winners
    assert Team.VILLAGE in winners
    assert Team.WEREWOLF not in winners


def test_village_wins_when_no_werewolves_and_nobody_dies():
    state = _state(
        [("p1", Role.VILLAGER), ("p2", Role.SEER),
         ("p3", Role.MASON), ("p4", Role.HUNTER)],
        dead=[],
        center_roles=[Role.WEREWOLF, Role.WEREWOLF, Role.VILLAGER],
    )
    winners = resolve_winners(state)
    assert Team.VILLAGE in winners
    assert Team.WEREWOLF not in winners


def test_village_loses_when_no_werewolves_and_someone_dies():
    state = _state(
        [("p1", Role.VILLAGER), ("p2", Role.SEER),
         ("p3", Role.MASON), ("p4", Role.HUNTER)],
        dead=["p1"],
        center_roles=[Role.WEREWOLF, Role.WEREWOLF, Role.VILLAGER],
    )
    winners = resolve_winners(state)
    assert Team.VILLAGE not in winners
    assert Team.WEREWOLF not in winners


def test_minion_wins_when_all_werewolves_in_center_and_nobody_dies():
    state = _state(
        [("p1", Role.MINION), ("p2", Role.VILLAGER),
         ("p3", Role.SEER), ("p4", Role.HUNTER)],
        dead=[],
        center_roles=[Role.WEREWOLF, Role.WEREWOLF, Role.VILLAGER],
    )
    winners = resolve_winners(state)
    assert Team.WEREWOLF in winners  # Minion wins on the werewolf team
    assert Team.VILLAGE in winners   # nobody-dies coincidentally satisfies village too


def test_minion_wins_when_all_werewolves_in_center_and_only_minion_dies():
    state = _state(
        [("p1", Role.MINION), ("p2", Role.VILLAGER),
         ("p3", Role.SEER), ("p4", Role.HUNTER)],
        dead=["p1"],
        center_roles=[Role.WEREWOLF, Role.WEREWOLF, Role.VILLAGER],
    )
    winners = resolve_winners(state)
    assert Team.WEREWOLF in winners
    assert Team.VILLAGE not in winners


def test_minion_loses_when_all_werewolves_in_center_and_non_minion_dies():
    state = _state(
        [("p1", Role.MINION), ("p2", Role.VILLAGER),
         ("p3", Role.SEER), ("p4", Role.HUNTER)],
        dead=["p2"],
        center_roles=[Role.WEREWOLF, Role.WEREWOLF, Role.VILLAGER],
    )
    winners = resolve_winners(state)
    assert winners == []  # everyone loses except Tanner (no Tanner here)
