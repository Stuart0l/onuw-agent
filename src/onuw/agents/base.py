from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..llm import TokenUsage
from ..types import Role

if TYPE_CHECKING:
    from ..events.bus import EventBus


class Agent(ABC):
    """Player agent. Engine pushes events via ``observe_*`` (no-op
    default) and pulls decisions via ``act_*``. Owns its own memory."""

    def __init__(self, player_id: str) -> None:
        self.player_id = player_id
        self.name: str = player_id
        self.seat: int = 0
        self.dealt_role: Role | None = None
        self.seat_order: list[str] = []
        self.role_pool: list[Role] = []
        self.language: str = "en"
        self.bus: "EventBus | None" = None
        self.token_usage: TokenUsage = TokenUsage()

    def bind(
        self,
        *,
        name: str,
        seat: int,
        dealt_role: Role,
        seat_order: list[str],
        role_pool: list[Role] | None = None,
        language: str = "en",
        bus: "EventBus | None" = None,
    ) -> None:
        self.name = name
        self.seat = seat
        self.dealt_role = dealt_role
        self.seat_order = seat_order
        self.role_pool = list(role_pool or [])
        self.language = language
        self.bus = bus

    # ---- Observations (engine -> agent). Default no-op.

    def observe_night(self, step: str, text: str, structured: dict) -> None:
        return None

    def observe_speech(
        self, round_idx: int, speaker_id: str, text: str
    ) -> None:
        return None

    def observe_votes(self, votes: dict[str, str]) -> None:
        return None

    def observe_deaths(
        self,
        deaths: list[str],
        hunter_revenge: list[tuple[str, str]],
    ) -> None:
        return None

    # ---- Decisions (engine <- agent).

    @abstractmethod
    async def act_night(
        self, action_key: str, valid_targets: list[str]
    ) -> dict:
        """``action_key`` is one of "werewolf_solo", "seer", "robber",
        "troublemaker", "drunk". ``valid_targets`` is the legal target
        list (excludes self where appropriate). Returns the action JSON."""

    @abstractmethod
    async def speak(
        self, round_idx: int, total_rounds: int, max_chars: int
    ) -> str:
        """Returns your public statement for this round."""

    @abstractmethod
    async def vote(self, valid_targets: list[str]) -> str:
        """Returns the player_id you vote for."""

    async def aclose(self) -> None:
        """Optional: release resources at game end. Default no-op."""
        return None
