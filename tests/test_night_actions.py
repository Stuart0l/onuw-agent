import pytest

from onuw.agents.scripted_agent import ScriptedAgent
from onuw.engine.night import run_night
from onuw.events.bus import EventBus
from onuw.memory import PlayerMemory
from onuw.prompts.rules import (
    GAME_RULES_BLOCK,
    ROLE_ABILITY_BLOCKS,
    WIN_CONDITIONS_BLOCK,
)
from onuw.state import CenterCard, GameState, PlayerState
from onuw.types import Role


def _build(player_roles, center_roles=None):
    """Hand-craft a GameState + memories + scripted agents dict."""
    center_roles = center_roles or [Role.VILLAGER, Role.VILLAGER, Role.VILLAGER]
    players: dict[str, PlayerState] = {}
    seat_order: list[str] = []
    memories: dict[str, PlayerMemory] = {}
    for i, (pid, role) in enumerate(player_roles):
        ps = PlayerState(
            id=pid, name=pid.upper(), seat=i,
            original_role=role, current_role=role,
        )
        players[pid] = ps
        seat_order.append(pid)
        memories[pid] = PlayerMemory(
            player_id=pid, seat=i, name=pid.upper(), persona=None,
            rules_text=GAME_RULES_BLOCK,
            role_ability_text=ROLE_ABILITY_BLOCKS[role],
            win_conditions_text=WIN_CONDITIONS_BLOCK,
            assigned_role=role,
        )
    center = [CenterCard(index=i, role=r) for i, r in enumerate(center_roles)]
    state = GameState(
        players=players, seat_order=seat_order, center=center,
        discussion_rounds=3,
    )
    return state, memories


def _agents(state, overrides: dict[str, ScriptedAgent] | None = None):
    overrides = overrides or {}
    return {pid: overrides.get(pid, ScriptedAgent(pid)) for pid in state.seat_order}


def _obs_text(memory: PlayerMemory) -> str:
    return " ".join(o.text for o in memory.night_observations)


pytestmark = pytest.mark.asyncio


async def test_werewolf_pair_see_each_other():
    state, memories = _build([
        ("p1", Role.WEREWOLF), ("p2", Role.WEREWOLF),
        ("p3", Role.VILLAGER), ("p4", Role.VILLAGER),
    ])
    await run_night(state, memories, _agents(state), EventBus())
    assert "p2" in _obs_text(memories["p1"])
    assert "p1" in _obs_text(memories["p2"])
    assert memories["p3"].night_observations == []
    assert memories["p4"].night_observations == []


async def test_solo_werewolf_peeks_center():
    state, memories = _build(
        [("p1", Role.WEREWOLF), ("p2", Role.VILLAGER),
         ("p3", Role.VILLAGER), ("p4", Role.VILLAGER)],
        center_roles=[Role.MASON, Role.SEER, Role.ROBBER],
    )
    agents = _agents(state, {
        "p1": ScriptedAgent("p1", night={"werewolf_solo": {"action": "peek_center", "index": 1}}),
    })
    await run_night(state, memories, agents, EventBus())
    text = _obs_text(memories["p1"])
    assert "seer" in text.lower()


async def test_minion_sees_werewolves_but_werewolves_dont_see_minion():
    state, memories = _build([
        ("p1", Role.WEREWOLF), ("p2", Role.WEREWOLF),
        ("p3", Role.MINION),   ("p4", Role.VILLAGER),
    ])
    await run_night(state, memories, _agents(state), EventBus())
    minion = _obs_text(memories["p3"])
    assert "p1" in minion and "p2" in minion
    for ww in ["p1", "p2"]:
        assert "p3" not in _obs_text(memories[ww])


async def test_minion_when_all_werewolves_in_center():
    state, memories = _build(
        [("p1", Role.MINION), ("p2", Role.VILLAGER),
         ("p3", Role.VILLAGER), ("p4", Role.VILLAGER)],
        center_roles=[Role.WEREWOLF, Role.WEREWOLF, Role.SEER],
    )
    await run_night(state, memories, _agents(state), EventBus())
    text = _obs_text(memories["p1"]).lower()
    assert "no werewolves" in text


async def test_mason_pair_see_each_other():
    state, memories = _build([
        ("p1", Role.MASON), ("p2", Role.MASON),
        ("p3", Role.VILLAGER), ("p4", Role.WEREWOLF),
    ])
    await run_night(state, memories, _agents(state), EventBus())
    assert "p2" in _obs_text(memories["p1"])
    assert "p1" in _obs_text(memories["p2"])


async def test_solo_mason_is_alone():
    state, memories = _build([
        ("p1", Role.MASON), ("p2", Role.VILLAGER),
        ("p3", Role.VILLAGER), ("p4", Role.WEREWOLF),
    ])
    await run_night(state, memories, _agents(state), EventBus())
    assert "only Mason" in _obs_text(memories["p1"])


async def test_seer_views_player():
    state, memories = _build([
        ("p1", Role.SEER), ("p2", Role.WEREWOLF),
        ("p3", Role.VILLAGER), ("p4", Role.VILLAGER),
    ])
    agents = _agents(state, {
        "p1": ScriptedAgent("p1", night={"seer": {"action": "view_player", "target": "p2"}}),
    })
    await run_night(state, memories, agents, EventBus())
    assert "werewolf" in _obs_text(memories["p1"]).lower()


async def test_seer_views_two_center_cards():
    state, memories = _build(
        [("p1", Role.SEER), ("p2", Role.VILLAGER),
         ("p3", Role.VILLAGER), ("p4", Role.VILLAGER)],
        center_roles=[Role.WEREWOLF, Role.MINION, Role.ROBBER],
    )
    agents = _agents(state, {
        "p1": ScriptedAgent("p1", night={"seer": {"action": "view_center", "indices": [0, 2]}}),
    })
    await run_night(state, memories, agents, EventBus())
    text = _obs_text(memories["p1"]).lower()
    assert "werewolf" in text and "robber" in text


async def test_robber_swaps_current_role_and_learns_new_role():
    state, memories = _build([
        ("p1", Role.ROBBER), ("p2", Role.SEER),
        ("p3", Role.VILLAGER), ("p4", Role.WEREWOLF),
    ])
    agents = _agents(state, {
        "p1": ScriptedAgent("p1", night={"robber": {"action": "rob", "target": "p4"}}),
    })
    await run_night(state, memories, agents, EventBus())
    assert state.players["p1"].current_role == Role.WEREWOLF
    assert state.players["p4"].current_role == Role.ROBBER
    # original_role is preserved
    assert state.players["p1"].original_role == Role.ROBBER
    text = _obs_text(memories["p1"])
    assert "werewolf" in text.lower()
    assert "does NOT grant" in text


async def test_troublemaker_swaps_blind_no_observation_for_targets():
    # Targets are non-acting roles so any observation MUST come from the
    # swap (and there should be none).
    state, memories = _build([
        ("p1", Role.TROUBLEMAKER), ("p2", Role.VILLAGER),
        ("p3", Role.WEREWOLF),     ("p4", Role.HUNTER),
    ])
    agents = _agents(state, {
        "p1": ScriptedAgent("p1", night={
            "troublemaker": {"action": "swap", "target_a": "p2", "target_b": "p4"}
        }),
    })
    await run_night(state, memories, agents, EventBus())
    assert state.players["p2"].current_role == Role.HUNTER
    assert state.players["p4"].current_role == Role.VILLAGER
    assert "without looking" in _obs_text(memories["p1"]).lower()
    assert memories["p2"].night_observations == []
    assert memories["p4"].night_observations == []


async def test_drunk_swaps_with_center_no_reveal():
    state, memories = _build(
        [("p1", Role.DRUNK), ("p2", Role.VILLAGER),
         ("p3", Role.VILLAGER), ("p4", Role.VILLAGER)],
        center_roles=[Role.WEREWOLF, Role.MINION, Role.SEER],
    )
    agents = _agents(state, {
        "p1": ScriptedAgent("p1", night={"drunk": {"action": "swap_center", "index": 0}}),
    })
    await run_night(state, memories, agents, EventBus())
    assert state.players["p1"].current_role == Role.WEREWOLF
    assert state.center[0].role == Role.DRUNK
    text = _obs_text(memories["p1"]).lower()
    assert "werewolf" not in text


async def test_insomniac_sees_current_role_after_troublemaker_swap():
    state, memories = _build([
        ("p1", Role.INSOMNIAC), ("p2", Role.TROUBLEMAKER),
        ("p3", Role.WEREWOLF), ("p4", Role.VILLAGER),
    ])
    agents = _agents(state, {
        "p2": ScriptedAgent("p2", night={
            "troublemaker": {"action": "swap", "target_a": "p1", "target_b": "p3"}
        }),
    })
    await run_night(state, memories, agents, EventBus())
    assert state.players["p1"].current_role == Role.WEREWOLF
    assert "werewolf" in _obs_text(memories["p1"]).lower()


async def test_robber_then_insomniac_no_double_wake():
    # p1=Robber robs p2 (Insomniac).
    # At Insomniac's wake, original Insomniac (p2) sees their current_role (now ROBBER).
    # p1, now current Insomniac, must NOT get an Insomniac observation.
    state, memories = _build([
        ("p1", Role.ROBBER), ("p2", Role.INSOMNIAC),
        ("p3", Role.VILLAGER), ("p4", Role.VILLAGER),
    ])
    agents = _agents(state, {
        "p1": ScriptedAgent("p1", night={"robber": {"action": "rob", "target": "p2"}}),
    })
    await run_night(state, memories, agents, EventBus())
    assert state.players["p1"].current_role == Role.INSOMNIAC
    assert state.players["p2"].current_role == Role.ROBBER
    p1_obs = memories["p1"].night_observations
    assert len(p1_obs) == 1
    assert p1_obs[0].step == "robber_action"
    p2_text = _obs_text(memories["p2"]).lower()
    assert "robber" in p2_text


async def test_villager_and_tanner_and_hunter_have_no_observations():
    state, memories = _build([
        ("p1", Role.VILLAGER), ("p2", Role.TANNER),
        ("p3", Role.HUNTER),   ("p4", Role.WEREWOLF),
    ])
    await run_night(state, memories, _agents(state), EventBus())
    assert memories["p1"].night_observations == []
    assert memories["p2"].night_observations == []
    assert memories["p3"].night_observations == []


# ===== Role-violation defense =====
# The engine MUST clamp agent responses so a misbehaving agent can't break
# the rules of the role it was assigned.


async def test_robber_self_target_is_clamped_to_legal_target():
    state, memories = _build([
        ("p1", Role.ROBBER), ("p2", Role.SEER),
        ("p3", Role.VILLAGER), ("p4", Role.VILLAGER),
    ])
    agents = _agents(state, {
        "p1": ScriptedAgent("p1", night={"robber": {"action": "rob", "target": "p1"}}),
    })
    await run_night(state, memories, agents, EventBus())
    # Engine refused self-rob; defaulted to first legal target (p2 = Seer).
    assert state.players["p1"].current_role == Role.SEER
    assert state.players["p2"].current_role == Role.ROBBER


async def test_robber_unknown_target_is_clamped():
    state, memories = _build([
        ("p1", Role.ROBBER), ("p2", Role.SEER),
        ("p3", Role.VILLAGER), ("p4", Role.VILLAGER),
    ])
    agents = _agents(state, {
        "p1": ScriptedAgent("p1", night={"robber": {"action": "rob", "target": "ghost"}}),
    })
    await run_night(state, memories, agents, EventBus())
    assert state.players["p1"].current_role == Role.SEER


async def test_troublemaker_cannot_swap_self():
    state, memories = _build([
        ("p1", Role.TROUBLEMAKER), ("p2", Role.WEREWOLF),
        ("p3", Role.VILLAGER), ("p4", Role.VILLAGER),
    ])
    agents = _agents(state, {
        "p1": ScriptedAgent("p1", night={
            "troublemaker": {"action": "swap", "target_a": "p1", "target_b": "p2"}
        }),
    })
    await run_night(state, memories, agents, EventBus())
    # Engine refused self-swap; the TM's own role is untouched.
    assert state.players["p1"].current_role == Role.TROUBLEMAKER


async def test_troublemaker_duplicate_target_is_clamped():
    state, memories = _build([
        ("p1", Role.TROUBLEMAKER), ("p2", Role.WEREWOLF),
        ("p3", Role.VILLAGER), ("p4", Role.VILLAGER),
    ])
    agents = _agents(state, {
        "p1": ScriptedAgent("p1", night={
            "troublemaker": {"action": "swap", "target_a": "p2", "target_b": "p2"}
        }),
    })
    await run_night(state, memories, agents, EventBus())
    # TM still owns Troublemaker; the duplicate-target was rejected and a
    # legal pair was chosen.
    assert state.players["p1"].current_role == Role.TROUBLEMAKER


async def test_drunk_out_of_range_center_index_is_clamped():
    state, memories = _build(
        [("p1", Role.DRUNK), ("p2", Role.VILLAGER),
         ("p3", Role.VILLAGER), ("p4", Role.VILLAGER)],
        center_roles=[Role.SEER, Role.MINION, Role.MASON],
    )
    agents = _agents(state, {
        "p1": ScriptedAgent("p1", night={"drunk": {"action": "swap_center", "index": 99}}),
    })
    await run_night(state, memories, agents, EventBus())
    # Index clamped to 0 -> took the Seer card.
    assert state.players["p1"].current_role == Role.SEER


async def test_seer_duplicate_center_indices_clamped_to_distinct():
    state, memories = _build(
        [("p1", Role.SEER), ("p2", Role.VILLAGER),
         ("p3", Role.VILLAGER), ("p4", Role.VILLAGER)],
        center_roles=[Role.WEREWOLF, Role.MINION, Role.MASON],
    )
    agents = _agents(state, {
        "p1": ScriptedAgent("p1", night={"seer": {"action": "view_center", "indices": [1, 1]}}),
    })
    await run_night(state, memories, agents, EventBus())
    # Engine produced a distinct pair; observation should mention two
    # different center cards.
    structured = memories["p1"].night_observations[0].structured
    i, j = structured["indices"]
    assert i != j


async def test_non_acting_role_returning_garbage_does_nothing():
    # A Villager / Tanner / Hunter agent is never called for a night
    # action — even if it has a populated night script.
    state, memories = _build([
        ("p1", Role.VILLAGER), ("p2", Role.TANNER),
        ("p3", Role.HUNTER),   ("p4", Role.WEREWOLF),
    ])
    agents = _agents(state, {
        "p1": ScriptedAgent("p1", night={"robber": {"action": "rob", "target": "p4"}}),
        "p2": ScriptedAgent("p2", night={"seer": {"action": "view_player", "target": "p4"}}),
    })
    await run_night(state, memories, agents, EventBus())
    # Roles and state are untouched by the spurious scripts.
    assert state.players["p1"].current_role == Role.VILLAGER
    assert state.players["p4"].current_role == Role.WEREWOLF
    assert memories["p1"].night_observations == []
    assert memories["p2"].night_observations == []
