import io
import json
from pathlib import Path
from unittest.mock import MagicMock

from rich.console import Console

from onuw.events.bus import (
    CenterDealtEvent,
    ContentChunkEvent,
    DeathsEvent,
    EventBus,
    GameEndEvent,
    GameStartEvent,
    LLMCallEvent,
    NightActionEvent,
    NightWakeEvent,
    ReasoningChunkEvent,
    RoleAssignedEvent,
    SpeechEvent,
    StateMutationEvent,
    VotesRevealedEvent,
)
from onuw.events.console import ConsoleObserver
from onuw.events.json_log import JsonObserver, _serialize
from onuw.events.observer import Observer
from onuw.types import Role, Team


CONCRETE_EVENTS = [
    GameStartEvent,
    RoleAssignedEvent,
    CenterDealtEvent,
    NightWakeEvent,
    NightActionEvent,
    StateMutationEvent,
    SpeechEvent,
    VotesRevealedEvent,
    DeathsEvent,
    ReasoningChunkEvent,
    ContentChunkEvent,
    LLMCallEvent,
    GameEndEvent,
]


def _plain_console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False, color_system=None, width=200), buf


def test_every_concrete_event_declares_visibility():
    for cls in CONCRETE_EVENTS:
        assert cls.visibility in {"public", "private", "god"}, cls


def test_event_bus_fans_out_to_all_observers():
    a = MagicMock(spec=Observer)
    b = MagicMock(spec=Observer)
    bus = EventBus([a, b])
    evt = NightWakeEvent(role=Role.WEREWOLF, actors=["p1"])
    bus.emit(evt)
    a.on_event.assert_called_once_with(evt)
    b.on_event.assert_called_once_with(evt)


def test_event_bus_add_after_construction():
    a = MagicMock(spec=Observer)
    bus = EventBus()
    bus.add(a)
    evt = NightWakeEvent(role=Role.SEER, actors=[])
    bus.emit(evt)
    a.on_event.assert_called_once_with(evt)


def test_serialize_enum_fields_become_string_values():
    rec = _serialize(RoleAssignedEvent(player_id="p1", role=Role.WEREWOLF))
    assert rec["type"] == "RoleAssignedEvent"
    assert rec["visibility"] == "private"
    assert rec["player_id"] == "p1"
    assert rec["role"] == "werewolf"


def test_serialize_handles_list_of_enums_and_tuples():
    rec = _serialize(CenterDealtEvent(cards=[Role.WEREWOLF, Role.VILLAGER, Role.SEER]))
    assert rec["cards"] == ["werewolf", "villager", "seer"]
    rec = _serialize(DeathsEvent(deaths=["p1"], hunter_revenge=[("p3", "p2")]))
    assert rec["hunter_revenge"] == [["p3", "p2"]]


def test_json_observer_writes_only_on_game_end(tmp_path: Path):
    obs = JsonObserver(tmp_path)
    obs.on_event(
        GameStartEvent(
            game_id="g1",
            players=[{"id": "p1", "name": "Alice"}],
            role_pool=[Role.WEREWOLF, Role.VILLAGER, Role.SEER, Role.ROBBER],
            discussion_rounds=3,
        )
    )
    obs.on_event(SpeechEvent(round_idx=0, speaker_id="p1", text="hi"))
    assert not list(tmp_path.iterdir()), "log must not exist before GameEndEvent"
    obs.on_event(GameEndEvent(winners=[Team.VILLAGE], final_state={"deaths": []}))
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name == "g1.json"
    data = json.loads(files[0].read_text())
    assert data["game_id"] == "g1"
    types = [e["type"] for e in data["events"]]
    assert types == ["GameStartEvent", "SpeechEvent", "GameEndEvent"]
    assert all("ts" in e for e in data["events"])


def test_console_observer_redacts_private_when_not_god():
    console, buf = _plain_console()
    obs = ConsoleObserver(god=False, console=console)
    obs.on_event(RoleAssignedEvent(player_id="p1", role=Role.WEREWOLF))
    out = buf.getvalue()
    assert "werewolf" not in out
    assert "hidden" in out


def test_console_observer_god_mode_reveals_private():
    console, buf = _plain_console()
    obs = ConsoleObserver(god=True, console=console)
    obs.on_event(RoleAssignedEvent(player_id="p1", role=Role.WEREWOLF))
    out = buf.getvalue()
    assert "werewolf" in out
    assert "p1" in out


def test_console_observer_renders_public_events_without_god():
    console, buf = _plain_console()
    obs = ConsoleObserver(god=False, console=console)
    obs.on_event(SpeechEvent(round_idx=0, speaker_id="p1", text="hello world"))
    obs.on_event(
        VotesRevealedEvent(votes={"p1": "p2", "p2": "p1"})
    )
    out = buf.getvalue()
    assert "hello world" in out
    assert "p1 -> p2" in out
