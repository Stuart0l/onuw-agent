from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..llm import TokenUsage
from ..types import Role

if TYPE_CHECKING:
    from ..events.bus import EventBus


class Agent(ABC):
    """Self-contained player agent.

    The engine never inspects an agent's internal state. Engine pushes
    world events via the ``observe_*`` hooks (default no-op) and pulls
    decisions via the ``act_*`` abstract methods. Each agent is
    responsible for its own memory, prompt rendering, and any state it
    needs across turns.

    Implementations may be backed by LiteLLM, LangChain, LangGraph,
    LlamaIndex, AutoGen, CrewAI, DSPy, the Claude Agent SDK, plain HTTP,
    a human CLI, or scripted responses for tests.

    Lifecycle:
      1. Constructed by an AgentFactory.
      2. ``bind(...)`` is called once at game setup with seat facts.
      3. ``observe_*`` is called as world events happen.
      4. ``act_night`` / ``speak`` / ``vote`` are invoked when it's the
         agent's turn to decide.
      5. ``aclose()`` is called once at game end.
    """

    def __init__(self, player_id: str) -> None:
        self.player_id = player_id
        self.name: str = player_id
        self.seat: int = 0
        self.dealt_role: Role | None = None
        self.seat_order: list[str] = []
        self.role_pool: list[Role] = []
        self.language: str = "en"
        self.bus: "EventBus | None" = None
        # Accumulates token usage across all LLM calls this agent makes
        # during a single game. Engine sums these at game end.
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
        """Engine-supplied facts. Default impl stores them on ``self``.
        Subclasses extend (e.g. to build a system prompt or construct
        internal memory)."""
        self.name = name
        self.seat = seat
        self.dealt_role = dealt_role
        self.seat_order = seat_order
        self.role_pool = list(role_pool or [])
        self.language = language
        self.bus = bus

    # ---- Observations (engine -> agent). Default no-op.

    def observe_night(self, step: str, text: str, structured: dict) -> None:
        """A night-action observation became visible to this agent."""
        return None

    def observe_speech(
        self, round_idx: int, speaker_id: str, text: str
    ) -> None:
        """A public speech was made — broadcast to every agent."""
        return None

    def observe_votes(self, votes: dict[str, str]) -> None:
        """The full vote map was revealed."""
        return None

    def observe_deaths(
        self,
        deaths: list[str],
        hunter_revenge: list[tuple[str, str]],
    ) -> None:
        """The death results were resolved."""
        return None

    # ---- Decisions (engine <- agent).

    @abstractmethod
    async def act_night(
        self, action_key: str, valid_targets: list[str]
    ) -> dict:
        """Return an action JSON object for a night decision.

        ``action_key`` identifies the decision: ``"werewolf_solo"``,
        ``"seer"``, ``"robber"``, ``"troublemaker"``, ``"drunk"``.
        ``valid_targets`` is the engine-enforced legal target list
        (excludes self where appropriate)."""

    @abstractmethod
    async def speak(
        self, round_idx: int, total_rounds: int, max_chars: int
    ) -> str:
        """Return your public statement for this round."""

    @abstractmethod
    async def vote(self, valid_targets: list[str]) -> str:
        """Return the player_id you vote for."""

    async def aclose(self) -> None:
        """Optional: release resources at game end. Default no-op."""
        return None
