import random
from dataclasses import dataclass, field

from .types import Role, Team


@dataclass
class PlayerState:
    id: str
    name: str
    seat: int
    original_role: Role
    current_role: Role
    is_dead: bool = False


@dataclass
class CenterCard:
    index: int
    role: Role


@dataclass(frozen=True)
class Speech:
    round_idx: int
    speaker_id: str
    text: str


@dataclass
class GameState:
    players: dict[str, PlayerState]
    seat_order: list[str]
    center: list[CenterCard]
    discussion_rounds: int
    speeches: list[Speech] = field(default_factory=list)
    votes: dict[str, str] = field(default_factory=dict)
    deaths: list[str] = field(default_factory=list)
    winners: list[Team] = field(default_factory=list)
    rng: random.Random = field(default_factory=random.Random)
