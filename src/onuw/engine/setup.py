import random
import uuid

from ..agents import AgentFactory
from ..agents.base import Agent
from ..config import GameConfig
from ..events.bus import (
    CenterDealtEvent,
    EventBus,
    GameStartEvent,
    RoleAssignedEvent,
)
from ..state import CenterCard, GameState, PlayerState


def deal(
    cfg: GameConfig,
    bus: EventBus,
    agent_factory: AgentFactory,
    game_id: str | None = None,
) -> tuple[GameState, dict[str, Agent]]:
    """Set up the game: shuffle the role pool, deal cards to players and
    center, instantiate one Agent per seat via the factory, bind each
    with its seat facts, and emit the start-of-game event stream.

    Agents own their private memory and prompt rendering.
    """
    if game_id is None:
        game_id = uuid.uuid4().hex[:8]

    rng = random.Random(cfg.seed) if cfg.seed is not None else random.Random()
    pool = list(cfg.role_pool)
    rng.shuffle(pool)

    n_players = len(cfg.players)
    player_states: dict[str, PlayerState] = {}
    seat_order: list[str] = []
    agents: dict[str, Agent] = {}

    for i, pcfg in enumerate(cfg.players):
        role = pool[i]
        ps = PlayerState(
            id=pcfg.id,
            name=pcfg.name,
            seat=i,
            original_role=role,
            current_role=role,
        )
        player_states[pcfg.id] = ps
        seat_order.append(pcfg.id)
        agents[pcfg.id] = agent_factory(pcfg)

    # Bind every agent now that we know the full seat order.
    for i, pcfg in enumerate(cfg.players):
        agents[pcfg.id].bind(
            name=pcfg.name,
            seat=i,
            dealt_role=player_states[pcfg.id].original_role,
            seat_order=seat_order,
            role_pool=list(cfg.role_pool),
            language="en",
            bus=bus,
        )

    center = [CenterCard(index=i, role=pool[n_players + i]) for i in range(3)]

    state = GameState(
        players=player_states,
        seat_order=seat_order,
        center=center,
        discussion_rounds=cfg.discussion_rounds,
        max_speech_chars=cfg.max_speech_chars,
        rng=rng,
    )

    bus.emit(
        GameStartEvent(
            game_id=game_id,
            players=[
                {
                    "id": p.id,
                    "name": p.name,
                    "model": p.model,
                }
                for p in cfg.players
            ],
            role_pool=list(cfg.role_pool),
            discussion_rounds=cfg.discussion_rounds,
        )
    )
    for pid, ps in player_states.items():
        bus.emit(RoleAssignedEvent(player_id=pid, role=ps.original_role))
    bus.emit(CenterDealtEvent(cards=[c.role for c in center]))

    return state, agents
