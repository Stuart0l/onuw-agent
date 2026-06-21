import pytest

from onuw.agents.scripted_agent import ScriptedAgent
from onuw.engine.vote import run_vote
from onuw.events.bus import EventBus
from onuw.memory import PlayerMemory
from onuw.prompts.rules import (
    GAME_RULES_BLOCK,
    ROLE_ABILITY_BLOCKS,
    WIN_CONDITIONS_BLOCK,
)
from onuw.state import CenterCard, GameState, PlayerState
from onuw.types import Role

pytestmark = pytest.mark.asyncio


def _build(player_roles):
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
    state = GameState(
        players=players, seat_order=seat_order,
        center=[CenterCard(index=i, role=Role.VILLAGER) for i in range(3)],
        discussion_rounds=1,
    )
    return state, memories


def _agents_with_votes(state: GameState, votes: dict[str, str]):
    return {pid: ScriptedAgent(pid, vote=votes[pid]) for pid in state.seat_order}


async def test_every_one_vote_each_no_death():
    state, memories = _build([
        ("p1", Role.WEREWOLF), ("p2", Role.VILLAGER),
        ("p3", Role.VILLAGER), ("p4", Role.VILLAGER),
    ])
    # Circular votes: each gets exactly one. Top = 1 < 2 → no death.
    agents = _agents_with_votes(state, {"p1": "p2", "p2": "p3", "p3": "p4", "p4": "p1"})
    await run_vote(state, memories, agents, EventBus())
    assert state.deaths == []


async def test_clear_majority_kills_one_player():
    state, memories = _build([
        ("p1", Role.WEREWOLF), ("p2", Role.VILLAGER),
        ("p3", Role.VILLAGER), ("p4", Role.VILLAGER),
    ])
    agents = _agents_with_votes(state, {"p1": "p2", "p2": "p1", "p3": "p1", "p4": "p1"})
    await run_vote(state, memories, agents, EventBus())
    assert state.deaths == ["p1"]
    assert state.players["p1"].is_dead


async def test_top_tie_kills_all_tied():
    state, memories = _build([
        ("p1", Role.WEREWOLF), ("p2", Role.VILLAGER),
        ("p3", Role.VILLAGER), ("p4", Role.VILLAGER),
    ])
    agents = _agents_with_votes(state, {"p1": "p2", "p2": "p1", "p3": "p1", "p4": "p2"})
    await run_vote(state, memories, agents, EventBus())
    assert sorted(state.deaths) == ["p1", "p2"]


async def test_hunter_chain_triggers():
    # p1 (Hunter) voted for p3 (Werewolf). p1 is killed; Hunter chain kills p3.
    state, memories = _build([
        ("p1", Role.HUNTER), ("p2", Role.VILLAGER),
        ("p3", Role.WEREWOLF), ("p4", Role.VILLAGER),
    ])
    agents = _agents_with_votes(state, {"p1": "p3", "p2": "p1", "p3": "p1", "p4": "p1"})
    await run_vote(state, memories, agents, EventBus())
    assert sorted(state.deaths) == ["p1", "p3"]
    assert state.players["p3"].is_dead


async def test_hunter_chain_is_non_recursive():
    # p1 (Hunter) dies by majority. p1 voted p4 → revenge kills p4 (Hunter).
    # p4 voted p5; the recursion gate must STOP that chain from firing —
    # p5 stays alive.
    state, memories = _build([
        ("p1", Role.HUNTER), ("p2", Role.VILLAGER),
        ("p3", Role.VILLAGER), ("p4", Role.HUNTER),
        ("p5", Role.VILLAGER),
    ])
    agents = _agents_with_votes(state, {
        "p1": "p4", "p2": "p1", "p3": "p1", "p4": "p5", "p5": "p1",
    })
    await run_vote(state, memories, agents, EventBus())
    assert sorted(state.deaths) == ["p1", "p4"]
    assert "p5" not in state.deaths


async def test_hunter_revenge_skips_target_already_dying():
    # p1 and p2 (Hunters) both vote for each other and both die.
    # Their revenge targets are each other - already in primary_deaths.
    # No extra deaths should be added.
    state, memories = _build([
        ("p1", Role.HUNTER), ("p2", Role.HUNTER),
        ("p3", Role.VILLAGER), ("p4", Role.WEREWOLF),
    ])
    agents = _agents_with_votes(state, {"p1": "p2", "p2": "p1", "p3": "p1", "p4": "p2"})
    await run_vote(state, memories, agents, EventBus())
    assert sorted(state.deaths) == ["p1", "p2"]


async def test_unknown_vote_target_defaults_to_self():
    state, memories = _build([
        ("p1", Role.VILLAGER), ("p2", Role.VILLAGER),
        ("p3", Role.VILLAGER), ("p4", Role.WEREWOLF),
    ])
    # p1's vote target is invalid → falls back to self.
    # p2, p3, p4 all vote for p1 → p1 dies anyway.
    agents = _agents_with_votes(state, {"p1": "ghost", "p2": "p1", "p3": "p1", "p4": "p1"})
    await run_vote(state, memories, agents, EventBus())
    assert state.votes["p1"] == "p1"
    assert state.deaths == ["p1"]


async def test_votes_revealed_event_contains_all_votes():
    state, memories = _build([
        ("p1", Role.VILLAGER), ("p2", Role.VILLAGER),
        ("p3", Role.VILLAGER), ("p4", Role.WEREWOLF),
    ])
    captured: list[dict] = []

    class _Capture:
        def on_event(self, event):
            if type(event).__name__ == "VotesRevealedEvent":
                captured.append(dict(event.votes))

    bus = EventBus([_Capture()])
    agents = _agents_with_votes(state, {"p1": "p4", "p2": "p4", "p3": "p4", "p4": "p1"})
    await run_vote(state, memories, agents, bus)
    assert len(captured) == 1
    assert captured[0] == {"p1": "p4", "p2": "p4", "p3": "p4", "p4": "p1"}
