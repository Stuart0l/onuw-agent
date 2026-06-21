from ..agents.base import Agent
from ..events.bus import EventBus, SpeechEvent
from ..memory import PlayerMemory
from ..prompts.day import build_day_speech_prompt
from ..state import GameState, Speech

MAX_SPEECH_CHARS = 600


async def run_day(
    state: GameState,
    memories: dict[str, PlayerMemory],
    agents: dict[str, Agent],
    bus: EventBus,
) -> None:
    """Strict round-robin: each player produces one statement per round,
    in seat order. Later speakers in the same round see earlier speakers.
    """
    for r in range(state.discussion_rounds):
        for pid in state.seat_order:
            prompt = build_day_speech_prompt(
                memories[pid], r, state.discussion_rounds
            )
            raw = await agents[pid].speak(r, prompt)
            text = (raw or "I have nothing to add.").strip()
            if len(text) > MAX_SPEECH_CHARS:
                text = text[:MAX_SPEECH_CHARS]
            speech = Speech(round_idx=r, speaker_id=pid, text=text)
            state.speeches.append(speech)
            for mem in memories.values():
                mem.add_speech(speech)
            bus.emit(
                SpeechEvent(round_idx=r, speaker_id=pid, text=text)
            )
