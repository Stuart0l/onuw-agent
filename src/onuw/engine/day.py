from ..agents.base import Agent
from ..events.bus import EventBus, SpeechEvent
from ..state import GameState, Speech


async def run_day(
    state: GameState,
    agents: dict[str, Agent],
    bus: EventBus,
) -> None:
    """Strict round-robin: each player produces one statement per round,
    in seat order. After each statement, every agent observes it so
    later speakers see earlier ones.
    """
    cap = state.max_speech_chars
    nothing_to_add = "I have nothing to add."
    for r in range(state.discussion_rounds):
        for pid in state.seat_order:
            raw = await agents[pid].speak(
                round_idx=r,
                total_rounds=state.discussion_rounds,
                max_chars=cap,
            )
            text = (raw or nothing_to_add).strip() or nothing_to_add
            if len(text) > cap:
                text = text[:cap]
            speech = Speech(round_idx=r, speaker_id=pid, text=text)
            state.speeches.append(speech)
            for agent in agents.values():
                agent.observe_speech(round_idx=r, speaker_id=pid, text=text)
            bus.emit(SpeechEvent(round_idx=r, speaker_id=pid, text=text))
