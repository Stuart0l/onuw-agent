from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

from ..types import Role, Team

if TYPE_CHECKING:
    from .observer import Observer


Visibility = Literal["public", "private", "god"]


@dataclass(frozen=True)
class Event:
    visibility: ClassVar[Visibility] = "public"


@dataclass(frozen=True)
class GameStartEvent(Event):
    game_id: str
    players: list[dict]
    role_pool: list[Role]
    discussion_rounds: int
    visibility: ClassVar[Visibility] = "public"


@dataclass(frozen=True)
class RoleAssignedEvent(Event):
    player_id: str
    role: Role
    visibility: ClassVar[Visibility] = "private"


@dataclass(frozen=True)
class CenterDealtEvent(Event):
    cards: list[Role]
    visibility: ClassVar[Visibility] = "god"


@dataclass(frozen=True)
class NightWakeEvent(Event):
    role: Role
    actors: list[str]
    visibility: ClassVar[Visibility] = "public"


@dataclass(frozen=True)
class NightActionEvent(Event):
    player_id: str
    role: Role
    action: dict
    observation: dict | None
    visibility: ClassVar[Visibility] = "private"


@dataclass(frozen=True)
class StateMutationEvent(Event):
    kind: Literal["swap_player_player", "swap_player_center"]
    a: str
    b: str
    visibility: ClassVar[Visibility] = "god"


@dataclass(frozen=True)
class SpeechEvent(Event):
    round_idx: int
    speaker_id: str
    text: str
    visibility: ClassVar[Visibility] = "public"


@dataclass(frozen=True)
class VotesRevealedEvent(Event):
    votes: dict[str, str]
    visibility: ClassVar[Visibility] = "public"


@dataclass(frozen=True)
class DeathsEvent(Event):
    deaths: list[str]
    hunter_revenge: list[tuple[str, str]]
    visibility: ClassVar[Visibility] = "public"


@dataclass(frozen=True)
class ReasoningChunkEvent(Event):
    """One streamed chunk of a reasoning-model's chain of thought."""

    player_id: str
    delta: str
    visibility: ClassVar[Visibility] = "private"


@dataclass(frozen=True)
class LLMCallEvent(Event):
    """One full LLM call's aggregated reasoning + content. Emitted
    after each call so the JSON log captures the whole text in one
    record instead of N chunk records."""

    player_id: str
    reasoning: str
    content: str
    visibility: ClassVar[Visibility] = "private"


@dataclass(frozen=True)
class ContentChunkEvent(Event):
    """One streamed chunk of the final answer."""

    player_id: str
    delta: str
    visibility: ClassVar[Visibility] = "private"


@dataclass(frozen=True)
class GameEndEvent(Event):
    winners: list[Team]
    final_state: dict
    visibility: ClassVar[Visibility] = "public"


class EventBus:
    def __init__(self, observers: list[Observer] | None = None) -> None:
        self._observers: list[Observer] = list(observers or [])

    def add(self, observer: Observer) -> None:
        self._observers.append(observer)

    def emit(self, event: Event) -> None:
        for observer in self._observers:
            observer.on_event(event)
