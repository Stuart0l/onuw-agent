from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..llm import TokenUsage
from ..memory import PlayerMemory

if TYPE_CHECKING:
    from ..events.bus import EventBus


class Agent(ABC):
    """Interface every player agent must implement.

    The engine never inspects an agent's internal state. Implementations may
    be backed by LiteLLM, LangChain, LangGraph, LlamaIndex, AutoGen, CrewAI,
    DSPy, the Claude Agent SDK, plain HTTP, a human CLI, or scripted
    responses for tests.

    Lifecycle:
      1. Constructed by an AgentFactory.
      2. ``bind(system_prompt, memory)`` is called once at game setup.
      3. ``act_night`` / ``speak`` / ``vote`` are invoked as the game runs.
      4. ``aclose()`` is called once at game end so the agent can release
         any resources it owns (HTTP sessions, etc.).
    """

    def __init__(self, player_id: str) -> None:
        self.player_id = player_id
        self.system_prompt: str = ""
        self.memory: PlayerMemory | None = None
        self.bus: "EventBus | None" = None
        # Accumulates token usage across all LLM calls this agent makes
        # during a single game. Engine sums these at game end.
        self.token_usage: TokenUsage = TokenUsage()

    def bind(
        self,
        system_prompt: str,
        memory: PlayerMemory,
        bus: "EventBus | None" = None,
    ) -> None:
        """Engine-supplied context. Custom agents are free to read
        ``self.memory`` directly and ignore the rendered ``user_prompt``
        passed to the action methods if they prefer their own templating.

        ``bus`` is supplied so backends that produce side-channel signal
        (e.g. reasoning-model chain-of-thought) can emit private events
        without needing direct access to observers.
        """
        self.system_prompt = system_prompt
        self.memory = memory
        self.bus = bus

    @abstractmethod
    async def act_night(self, action_key: str, user_prompt: str) -> dict:
        """Return an action JSON object matching the schema described in
        ``user_prompt``.

        ``action_key`` identifies the decision (e.g. ``"robber"``,
        ``"seer"``, ``"troublemaker"``, ``"drunk"``, ``"werewolf_solo"``).
        """

    @abstractmethod
    async def speak(self, round_idx: int, user_prompt: str) -> str:
        """Return your public statement for this round."""

    @abstractmethod
    async def vote(self, user_prompt: str) -> str:
        """Return the player_id you vote for."""

    async def aclose(self) -> None:
        """Optional: release resources at game end. Default no-op."""
        return None
