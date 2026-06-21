import uuid
from dataclasses import asdict, dataclass

from ..agents import AgentFactory
from ..agents.base import Agent
from ..config import GameConfig
from ..events.bus import EventBus, GameEndEvent
from ..llm import TokenUsage
from ..memory import PlayerMemory
from ..state import GameState
from ..types import Team
from .day import run_day
from .night import run_night
from .resolve import resolve_winners
from .setup import deal
from .vote import run_vote


@dataclass
class GameResult:
    game_id: str
    winners: list[Team]
    state: GameState


class GameEngine:
    def __init__(
        self,
        cfg: GameConfig,
        bus: EventBus,
        agent_factory: AgentFactory,
        game_id: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.bus = bus
        self.agent_factory = agent_factory
        self.game_id = game_id or uuid.uuid4().hex[:8]
        self.state: GameState | None = None
        self.memories: dict[str, PlayerMemory] | None = None
        self.agents: dict[str, Agent] | None = None

    async def run(self) -> GameResult:
        state, memories, agents = deal(
            self.cfg, self.bus, self.agent_factory, self.game_id
        )
        self.state, self.memories, self.agents = state, memories, agents
        try:
            await run_night(state, memories, agents, self.bus)
            await run_day(state, memories, agents, self.bus)
            await run_vote(state, memories, agents, self.bus)
            winners = resolve_winners(state)
            state.winners = winners
            self.bus.emit(
                GameEndEvent(
                    winners=list(winners),
                    final_state=self._final_state(state, agents),
                )
            )
            return GameResult(
                game_id=self.game_id, winners=winners, state=state
            )
        finally:
            for agent in agents.values():
                await agent.aclose()

    @staticmethod
    def _final_state(state: GameState, agents: dict[str, Agent]) -> dict:
        per_player_usage = {
            pid: asdict(agent.token_usage) for pid, agent in agents.items()
        }
        total = TokenUsage()
        for agent in agents.values():
            total += agent.token_usage
        return {
            "players": {
                pid: {
                    "original_role": p.original_role.value,
                    "current_role": p.current_role.value,
                    "is_dead": p.is_dead,
                }
                for pid, p in state.players.items()
            },
            "center_final": [c.role.value for c in state.center],
            "deaths": list(state.deaths),
            "votes": dict(state.votes),
            "winners": [w.value for w in state.winners],
            "token_usage": {
                "per_player": per_player_usage,
                "total": asdict(total),
            },
        }
