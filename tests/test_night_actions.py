import pytest

from onuw.agents.scripted_agent import ScriptedAgent
from onuw.engine.night import run_night
from onuw.events.bus import EventBus
from onuw.state import CenterCard, GameState, PlayerState
from onuw.types import Role


def _build(player_roles, center_roles=None, overrides=None):
    """Hand-craft a GameState and bound ScriptedAgents."""
    center_roles = center_roles or [Role.VILLAGER, Role.VILLAGER, Role.VILLAGER]
    overrides = overrides or {}
    players: dict[str, PlayerState] = {}
    seat_order: list[str] = []
    for i, (pid, role) in enumerate(player_roles):
        ps = PlayerState(
            id=pid, name=pid.upper(), seat=i,
            original_role=role, current_role=role,
        )
        players[pid] = ps
        seat_order.append(pid)
    center = [CenterCard(index=i, role=r) for i, r in enumerate(center_roles)]
    state = GameState(
        players=players, seat_order=seat_order, center=center,
        discussion_rounds=3,
    )
    agents = {
        pid: overrides.get(pid, ScriptedAgent(pid)) for pid in seat_order
    }
    for pid, agent in agents.items():
        ps = players[pid]
        agent.bind(
            name=ps.name, seat=ps.seat, dealt_role=ps.original_role,
            seat_order=seat_order,
        )
    return state, agents


def _obs_text(agent) -> str:
    return " ".join(o.text for o in agent.night_observations)


pytestmark = pytest.mark.asyncio


async def test_werewolf_pair_see_each_other():
    state, agents = _build([
        ("p1", Role.WEREWOLF), ("p2", Role.WEREWOLF),
        ("p3", Role.VILLAGER), ("p4", Role.VILLAGER),
    ])
    await run_night(state, agents, EventBus())
    assert "p2" in _obs_text(agents["p1"])
    assert "p1" in _obs_text(agents["p2"])
    assert agents["p3"].night_observations == []
    assert agents["p4"].night_observations == []


async def test_solo_werewolf_peeks_center():
    state, agents = _build(
        [("p1", Role.WEREWOLF), ("p2", Role.VILLAGER),
         ("p3", Role.VILLAGER), ("p4", Role.VILLAGER)],
        center_roles=[Role.MASON, Role.SEER, Role.ROBBER],
        overrides={
            "p1": ScriptedAgent("p1", night={"werewolf_solo": {"action": "peek_center", "index": 1}}),
        },
    )
    await run_night(state, agents, EventBus())
    text = _obs_text(agents["p1"])
    assert "seer" in text.lower()


async def test_minion_sees_werewolves_but_werewolves_dont_see_minion():
    state, agents = _build([
        ("p1", Role.WEREWOLF), ("p2", Role.WEREWOLF),
        ("p3", Role.MINION),   ("p4", Role.VILLAGER),
    ])
    await run_night(state, agents, EventBus())
    minion = _obs_text(agents["p3"])
    assert "p1" in minion and "p2" in minion
    for ww in ["p1", "p2"]:
        assert "p3" not in _obs_text(agents[ww])


async def test_minion_when_all_werewolves_in_center():
    state, agents = _build(
        [("p1", Role.MINION), ("p2", Role.VILLAGER),
         ("p3", Role.VILLAGER), ("p4", Role.VILLAGER)],
        center_roles=[Role.WEREWOLF, Role.WEREWOLF, Role.SEER],
    )
    await run_night(state, agents, EventBus())
    text = _obs_text(agents["p1"]).lower()
    assert "no werewolves" in text


async def test_mason_pair_see_each_other():
    state, agents = _build([
        ("p1", Role.MASON), ("p2", Role.MASON),
        ("p3", Role.VILLAGER), ("p4", Role.WEREWOLF),
    ])
    await run_night(state, agents, EventBus())
    assert "p2" in _obs_text(agents["p1"])
    assert "p1" in _obs_text(agents["p2"])


async def test_solo_mason_is_alone():
    state, agents = _build([
        ("p1", Role.MASON), ("p2", Role.VILLAGER),
        ("p3", Role.VILLAGER), ("p4", Role.WEREWOLF),
    ])
    await run_night(state, agents, EventBus())
    assert "only Mason" in _obs_text(agents["p1"])


async def test_seer_views_player():
    state, agents = _build(
        [("p1", Role.SEER), ("p2", Role.WEREWOLF),
         ("p3", Role.VILLAGER), ("p4", Role.VILLAGER)],
        overrides={
            "p1": ScriptedAgent("p1", night={"seer": {"action": "view_player", "target": "p2"}}),
        },
    )
    await run_night(state, agents, EventBus())
    assert "werewolf" in _obs_text(agents["p1"]).lower()


async def test_seer_views_two_center_cards():
    state, agents = _build(
        [("p1", Role.SEER), ("p2", Role.VILLAGER),
         ("p3", Role.VILLAGER), ("p4", Role.VILLAGER)],
        center_roles=[Role.WEREWOLF, Role.MINION, Role.ROBBER],
        overrides={
            "p1": ScriptedAgent("p1", night={"seer": {"action": "view_center", "indices": [0, 2]}}),
        },
    )
    await run_night(state, agents, EventBus())
    text = _obs_text(agents["p1"]).lower()
    assert "werewolf" in text and "robber" in text


async def test_robber_swaps_current_role_and_learns_new_role():
    state, agents = _build(
        [("p1", Role.ROBBER), ("p2", Role.SEER),
         ("p3", Role.VILLAGER), ("p4", Role.WEREWOLF)],
        overrides={
            "p1": ScriptedAgent("p1", night={"robber": {"action": "rob", "target": "p4"}}),
        },
    )
    await run_night(state, agents, EventBus())
    assert state.players["p1"].current_role == Role.WEREWOLF
    assert state.players["p4"].current_role == Role.ROBBER
    assert state.players["p1"].original_role == Role.ROBBER
    text = _obs_text(agents["p1"])
    assert "werewolf" in text.lower()
    assert "does NOT grant" in text


async def test_troublemaker_swaps_blind_no_observation_for_targets():
    state, agents = _build(
        [("p1", Role.TROUBLEMAKER), ("p2", Role.VILLAGER),
         ("p3", Role.WEREWOLF),     ("p4", Role.HUNTER)],
        overrides={
            "p1": ScriptedAgent("p1", night={
                "troublemaker": {"action": "swap", "target_a": "p2", "target_b": "p4"}
            }),
        },
    )
    await run_night(state, agents, EventBus())
    assert state.players["p2"].current_role == Role.HUNTER
    assert state.players["p4"].current_role == Role.VILLAGER
    assert "without looking" in _obs_text(agents["p1"]).lower()
    assert agents["p2"].night_observations == []
    assert agents["p4"].night_observations == []


async def test_drunk_swaps_with_center_no_reveal():
    state, agents = _build(
        [("p1", Role.DRUNK), ("p2", Role.VILLAGER),
         ("p3", Role.VILLAGER), ("p4", Role.VILLAGER)],
        center_roles=[Role.WEREWOLF, Role.MINION, Role.SEER],
        overrides={
            "p1": ScriptedAgent("p1", night={"drunk": {"action": "swap_center", "index": 0}}),
        },
    )
    await run_night(state, agents, EventBus())
    assert state.players["p1"].current_role == Role.WEREWOLF
    assert state.center[0].role == Role.DRUNK
    text = _obs_text(agents["p1"]).lower()
    assert "werewolf" not in text


async def test_insomniac_sees_current_role_after_troublemaker_swap():
    state, agents = _build(
        [("p1", Role.INSOMNIAC), ("p2", Role.TROUBLEMAKER),
         ("p3", Role.WEREWOLF), ("p4", Role.VILLAGER)],
        overrides={
            "p2": ScriptedAgent("p2", night={
                "troublemaker": {"action": "swap", "target_a": "p1", "target_b": "p3"}
            }),
        },
    )
    await run_night(state, agents, EventBus())
    assert state.players["p1"].current_role == Role.WEREWOLF
    assert "werewolf" in _obs_text(agents["p1"]).lower()


async def test_robber_then_insomniac_no_double_wake():
    state, agents = _build(
        [("p1", Role.ROBBER), ("p2", Role.INSOMNIAC),
         ("p3", Role.VILLAGER), ("p4", Role.VILLAGER)],
        overrides={
            "p1": ScriptedAgent("p1", night={"robber": {"action": "rob", "target": "p2"}}),
        },
    )
    await run_night(state, agents, EventBus())
    assert state.players["p1"].current_role == Role.INSOMNIAC
    assert state.players["p2"].current_role == Role.ROBBER
    p1_obs = agents["p1"].night_observations
    assert len(p1_obs) == 1
    assert p1_obs[0].step == "robber_action"
    p2_text = _obs_text(agents["p2"]).lower()
    assert "robber" in p2_text


async def test_villager_and_tanner_and_hunter_have_no_observations():
    state, agents = _build([
        ("p1", Role.VILLAGER), ("p2", Role.TANNER),
        ("p3", Role.HUNTER),   ("p4", Role.WEREWOLF),
    ])
    await run_night(state, agents, EventBus())
    assert agents["p1"].night_observations == []
    assert agents["p2"].night_observations == []
    assert agents["p3"].night_observations == []


# ===== Role-violation defense =====


async def test_robber_self_target_is_clamped_to_legal_target():
    state, agents = _build(
        [("p1", Role.ROBBER), ("p2", Role.SEER),
         ("p3", Role.VILLAGER), ("p4", Role.VILLAGER)],
        overrides={
            "p1": ScriptedAgent("p1", night={"robber": {"action": "rob", "target": "p1"}}),
        },
    )
    await run_night(state, agents, EventBus())
    assert state.players["p1"].current_role == Role.SEER
    assert state.players["p2"].current_role == Role.ROBBER


async def test_robber_unknown_target_is_clamped():
    state, agents = _build(
        [("p1", Role.ROBBER), ("p2", Role.SEER),
         ("p3", Role.VILLAGER), ("p4", Role.VILLAGER)],
        overrides={
            "p1": ScriptedAgent("p1", night={"robber": {"action": "rob", "target": "ghost"}}),
        },
    )
    await run_night(state, agents, EventBus())
    assert state.players["p1"].current_role == Role.SEER


async def test_troublemaker_cannot_swap_self():
    state, agents = _build(
        [("p1", Role.TROUBLEMAKER), ("p2", Role.WEREWOLF),
         ("p3", Role.VILLAGER), ("p4", Role.VILLAGER)],
        overrides={
            "p1": ScriptedAgent("p1", night={
                "troublemaker": {"action": "swap", "target_a": "p1", "target_b": "p2"}
            }),
        },
    )
    await run_night(state, agents, EventBus())
    assert state.players["p1"].current_role == Role.TROUBLEMAKER


async def test_troublemaker_duplicate_target_is_clamped():
    state, agents = _build(
        [("p1", Role.TROUBLEMAKER), ("p2", Role.WEREWOLF),
         ("p3", Role.VILLAGER), ("p4", Role.VILLAGER)],
        overrides={
            "p1": ScriptedAgent("p1", night={
                "troublemaker": {"action": "swap", "target_a": "p2", "target_b": "p2"}
            }),
        },
    )
    await run_night(state, agents, EventBus())
    assert state.players["p1"].current_role == Role.TROUBLEMAKER


async def test_drunk_out_of_range_center_index_is_clamped():
    state, agents = _build(
        [("p1", Role.DRUNK), ("p2", Role.VILLAGER),
         ("p3", Role.VILLAGER), ("p4", Role.VILLAGER)],
        center_roles=[Role.SEER, Role.MINION, Role.MASON],
        overrides={
            "p1": ScriptedAgent("p1", night={"drunk": {"action": "swap_center", "index": 99}}),
        },
    )
    await run_night(state, agents, EventBus())
    assert state.players["p1"].current_role == Role.SEER


async def test_seer_duplicate_center_indices_clamped_to_distinct():
    state, agents = _build(
        [("p1", Role.SEER), ("p2", Role.VILLAGER),
         ("p3", Role.VILLAGER), ("p4", Role.VILLAGER)],
        center_roles=[Role.WEREWOLF, Role.MINION, Role.MASON],
        overrides={
            "p1": ScriptedAgent("p1", night={"seer": {"action": "view_center", "indices": [1, 1]}}),
        },
    )
    await run_night(state, agents, EventBus())
    structured = agents["p1"].night_observations[0].structured
    i, j = structured["indices"]
    assert i != j


async def test_non_acting_role_returning_garbage_does_nothing():
    state, agents = _build(
        [("p1", Role.VILLAGER), ("p2", Role.TANNER),
         ("p3", Role.HUNTER),   ("p4", Role.WEREWOLF)],
        overrides={
            "p1": ScriptedAgent("p1", night={"robber": {"action": "rob", "target": "p4"}}),
            "p2": ScriptedAgent("p2", night={"seer": {"action": "view_player", "target": "p4"}}),
        },
    )
    await run_night(state, agents, EventBus())
    assert state.players["p1"].current_role == Role.VILLAGER
    assert state.players["p4"].current_role == Role.WEREWOLF
    assert agents["p1"].night_observations == []
    assert agents["p2"].night_observations == []
