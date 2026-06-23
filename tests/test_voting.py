import pytest

from onuw.agents.scripted_agent import ScriptedAgent
from onuw.engine.vote import run_vote
from onuw.events.bus import EventBus
from onuw.state import CenterCard, GameState, PlayerState
from onuw.types import Role

pytestmark = pytest.mark.asyncio


def _build(player_roles):
    players: dict[str, PlayerState] = {}
    seat_order: list[str] = []
    for i, (pid, role) in enumerate(player_roles):
        ps = PlayerState(
            id=pid, name=pid.upper(), seat=i,
            original_role=role, current_role=role,
        )
        players[pid] = ps
        seat_order.append(pid)
    state = GameState(
        players=players, seat_order=seat_order,
        center=[CenterCard(index=i, role=Role.VILLAGER) for i in range(3)],
        discussion_rounds=1,
    )
    return state


def _agents_with_votes(state: GameState, votes: dict[str, str]):
    agents = {pid: ScriptedAgent(pid, vote=votes[pid]) for pid in state.seat_order}
    for pid, agent in agents.items():
        ps = state.players[pid]
        agent.bind(
            name=ps.name, seat=ps.seat, dealt_role=ps.original_role,
            persona=None, seat_order=state.seat_order,
        )
    return agents


async def test_every_one_vote_each_no_death():
    state = _build([
        ("p1", Role.WEREWOLF), ("p2", Role.VILLAGER),
        ("p3", Role.VILLAGER), ("p4", Role.VILLAGER),
    ])
    agents = _agents_with_votes(state, {"p1": "p2", "p2": "p3", "p3": "p4", "p4": "p1"})
    await run_vote(state, agents, EventBus())
    assert state.deaths == []


async def test_clear_majority_kills_one_player():
    state = _build([
        ("p1", Role.WEREWOLF), ("p2", Role.VILLAGER),
        ("p3", Role.VILLAGER), ("p4", Role.VILLAGER),
    ])
    agents = _agents_with_votes(state, {"p1": "p2", "p2": "p1", "p3": "p1", "p4": "p1"})
    await run_vote(state, agents, EventBus())
    assert state.deaths == ["p1"]
    assert state.players["p1"].is_dead


async def test_top_tie_kills_all_tied():
    state = _build([
        ("p1", Role.WEREWOLF), ("p2", Role.VILLAGER),
        ("p3", Role.VILLAGER), ("p4", Role.VILLAGER),
    ])
    agents = _agents_with_votes(state, {"p1": "p2", "p2": "p1", "p3": "p1", "p4": "p2"})
    await run_vote(state, agents, EventBus())
    assert sorted(state.deaths) == ["p1", "p2"]


async def test_hunter_chain_triggers():
    state = _build([
        ("p1", Role.HUNTER), ("p2", Role.VILLAGER),
        ("p3", Role.WEREWOLF), ("p4", Role.VILLAGER),
    ])
    agents = _agents_with_votes(state, {"p1": "p3", "p2": "p1", "p3": "p1", "p4": "p1"})
    await run_vote(state, agents, EventBus())
    assert sorted(state.deaths) == ["p1", "p3"]
    assert state.players["p3"].is_dead


async def test_hunter_chain_is_non_recursive():
    state = _build([
        ("p1", Role.HUNTER), ("p2", Role.VILLAGER),
        ("p3", Role.VILLAGER), ("p4", Role.HUNTER),
        ("p5", Role.VILLAGER),
    ])
    agents = _agents_with_votes(state, {
        "p1": "p4", "p2": "p1", "p3": "p1", "p4": "p5", "p5": "p1",
    })
    await run_vote(state, agents, EventBus())
    assert sorted(state.deaths) == ["p1", "p4"]
    assert "p5" not in state.deaths


async def test_hunter_revenge_skips_target_already_dying():
    state = _build([
        ("p1", Role.HUNTER), ("p2", Role.HUNTER),
        ("p3", Role.VILLAGER), ("p4", Role.WEREWOLF),
    ])
    agents = _agents_with_votes(state, {"p1": "p2", "p2": "p1", "p3": "p1", "p4": "p2"})
    await run_vote(state, agents, EventBus())
    assert sorted(state.deaths) == ["p1", "p2"]


async def test_unknown_vote_target_defaults_to_self():
    state = _build([
        ("p1", Role.VILLAGER), ("p2", Role.VILLAGER),
        ("p3", Role.VILLAGER), ("p4", Role.WEREWOLF),
    ])
    agents = _agents_with_votes(state, {"p1": "ghost", "p2": "p1", "p3": "p1", "p4": "p1"})
    await run_vote(state, agents, EventBus())
    assert state.votes["p1"] == "p1"
    assert state.deaths == ["p1"]


async def test_votes_revealed_event_contains_all_votes():
    state = _build([
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
    await run_vote(state, agents, bus)
    assert len(captured) == 1
    assert captured[0] == {"p1": "p4", "p2": "p4", "p3": "p4", "p4": "p1"}
