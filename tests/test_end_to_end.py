import json
from pathlib import Path

import pytest

from onuw.agents.scripted_agent import ScriptedAgent
from onuw.config import GameConfig, PlayerConfig
from onuw.engine.engine import GameEngine
from onuw.events.bus import EventBus
from onuw.events.json_log import JsonObserver
from onuw.types import Role

pytestmark = pytest.mark.asyncio


def _scripted_factory(vote: str, day_msg: str = "scripted line"):
    """Returns a factory that hands every seat the same kind of scripted
    agent — one that has a populated night script (the engine picks the
    relevant key for the dealt role) and votes for the given target."""
    night = {
        "werewolf_solo": {"action": "peek_center", "index": 0},
        "seer": {"action": "view_center", "indices": [0, 1]},
        "robber": {"action": "rob", "target": "p2"},
        "troublemaker": {"action": "swap", "target_a": "p2", "target_b": "p3"},
        "drunk": {"action": "swap_center", "index": 0},
    }
    def factory(pcfg):
        return ScriptedAgent(
            pcfg.id,
            night=night,
            day={0: f"{pcfg.id}: {day_msg}", 1: f"{pcfg.id}: {day_msg} (round 2)"},
            vote=vote,
        )
    return factory


async def test_full_scripted_game_produces_winners_and_json_log(tmp_path: Path):
    cfg = GameConfig(
        players=[
            PlayerConfig(id="p1", name="Alice", model="scripted"),
            PlayerConfig(id="p2", name="Bob", model="scripted"),
            PlayerConfig(id="p3", name="Cara", model="scripted"),
            PlayerConfig(id="p4", name="Dan", model="scripted"),
            PlayerConfig(id="p5", name="Eve", model="scripted"),
        ],
        role_pool=[
            Role.WEREWOLF, Role.SEER, Role.ROBBER, Role.VILLAGER,
            Role.VILLAGER, Role.TANNER, Role.HUNTER, Role.MINION,
        ],
        discussion_rounds=2,
        seed=42,
        log_dir=tmp_path,
    )

    bus = EventBus([JsonObserver(tmp_path)])
    engine = GameEngine(cfg, bus, agent_factory=_scripted_factory(vote="p1"), game_id="test_e2e")
    result = await engine.run()

    # Everyone voted p1 → p1 dies. Engine completed without exception.
    # (winners may be empty in the "all werewolves in center + non-tanner
    # dies" scenario; that is a valid ONUW outcome where everyone loses.)
    assert "p1" in result.state.deaths
    assert isinstance(result.winners, list)

    log = tmp_path / "test_e2e.json"
    assert log.exists()
    data = json.loads(log.read_text())
    types = [e["type"] for e in data["events"]]
    assert types[0] == "GameStartEvent"
    assert types[-1] == "GameEndEvent"
    # Speeches: 5 players * 2 rounds = 10
    assert sum(1 for t in types if t == "SpeechEvent") == 10
    assert sum(1 for t in types if t == "RoleAssignedEvent") == 5
    assert sum(1 for t in types if t == "VotesRevealedEvent") == 1
    assert sum(1 for t in types if t == "DeathsEvent") == 1


async def test_same_seed_yields_same_role_assignment(tmp_path: Path):
    cfg = GameConfig(
        players=[
            PlayerConfig(id=f"p{i + 1}", name=f"P{i + 1}", model="scripted")
            for i in range(3)
        ],
        role_pool=[
            Role.WEREWOLF, Role.VILLAGER, Role.SEER,
            Role.ROBBER, Role.VILLAGER, Role.TANNER,
        ],
        discussion_rounds=1,
        seed=99,
        log_dir=tmp_path,
    )

    r1 = await GameEngine(cfg, EventBus(), _scripted_factory(vote="p1"), game_id="a").run()
    r2 = await GameEngine(cfg, EventBus(), _scripted_factory(vote="p1"), game_id="b").run()

    assigned_1 = {pid: p.original_role for pid, p in r1.state.players.items()}
    assigned_2 = {pid: p.original_role for pid, p in r2.state.players.items()}
    assert assigned_1 == assigned_2
