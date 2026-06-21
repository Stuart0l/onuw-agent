import asyncio

from ..agents.base import Agent
from ..events.bus import DeathsEvent, EventBus, VotesRevealedEvent
from ..memory import PlayerMemory
from ..prompts.vote import build_vote_prompt
from ..state import GameState
from ..types import Role


async def run_vote(
    state: GameState,
    memories: dict[str, PlayerMemory],
    agents: dict[str, Agent],
    bus: EventBus,
) -> None:
    valid_targets = list(state.seat_order)

    async def _vote(pid: str) -> str:
        prompt = build_vote_prompt(memories[pid], valid_targets)
        target = await agents[pid].vote(prompt)
        if isinstance(target, str) and target in state.players:
            return target
        return pid  # fallback: vote for self

    coros = [_vote(pid) for pid in state.seat_order]
    results = await asyncio.gather(*coros)
    state.votes = dict(zip(state.seat_order, results))
    bus.emit(VotesRevealedEvent(votes=dict(state.votes)))

    deaths, hunter_revenge = _resolve_deaths(state)
    state.deaths = deaths
    for pid in deaths:
        state.players[pid].is_dead = True
    bus.emit(DeathsEvent(deaths=list(deaths), hunter_revenge=list(hunter_revenge)))


def _resolve_deaths(state: GameState) -> tuple[list[str], list[tuple[str, str]]]:
    """ONUW vote resolution:
      - Count votes per target.
      - If the top count is < 2, no one dies (everyone got exactly one vote).
      - Otherwise every player tied at the top dies.
      - Hunter revenge: any dead player whose CURRENT role is Hunter also
        kills the player they voted for. Non-recursive: if the Hunter's
        target is also a Hunter, that second Hunter's vote does NOT fire.
    """
    counts: dict[str, int] = {}
    for target in state.votes.values():
        counts[target] = counts.get(target, 0) + 1
    if not counts:
        return [], []
    top = max(counts.values())
    if top < 2:
        return [], []
    primary = [pid for pid, c in counts.items() if c == top]

    extra: list[str] = []
    hunter_revenge: list[tuple[str, str]] = []
    for hunter_id in primary:
        if state.players[hunter_id].current_role != Role.HUNTER:
            continue
        target = state.votes.get(hunter_id)
        if not target:
            continue
        if target in primary or target in extra:
            continue  # already dying; don't double-add
        extra.append(target)
        hunter_revenge.append((hunter_id, target))

    return primary + extra, hunter_revenge