import json
import warnings
from typing import TYPE_CHECKING, Any

from ..events.bus import ContentChunkEvent, LLMCallEvent, ReasoningChunkEvent
from ..llm.client import LLMClient
from ..memory import PlayerMemory
from ..prompts.day import build_day_speech_task
from ..prompts.night import build_night_task
from ..prompts.system import build_night_system_prompt, build_system_prompt
from ..prompts.vote import build_vote_task
from ..state import Speech
from ..types import Role
from ..utils.json_parse import extract_json
from .base import Agent

if TYPE_CHECKING:
    from ..events.bus import EventBus


_ACTION_KEY_TO_ROLE: dict[str, Role] = {
    "werewolf_solo": Role.WEREWOLF,
    "seer": Role.SEER,
    "robber": Role.ROBBER,
    "troublemaker": Role.TROUBLEMAKER,
    "drunk": Role.DRUNK,
}


class LLMAgent(Agent):
    """Default LiteLLM-backed agent. Owns its private memory and builds
    its own prompts. One JSON-retry on parse failure, then a
    deterministic safe default."""

    def __init__(
        self,
        player_id: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 800,
        json_mode: bool = False,
        extra_body: dict | None = None,
        client: LLMClient | None = None,
    ) -> None:
        super().__init__(player_id)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.json_mode = json_mode
        self.extra_body = extra_body or None
        self.client = client or LLMClient()
        self._memory: PlayerMemory | None = None
        self._system_prompt: str = ""
        self._night_system_prompt: str = ""

    @property
    def memory(self) -> PlayerMemory | None:
        """Read-only view of the agent's private memory (exposed for
        tests / debugging; the engine does not read it)."""
        return self._memory

    # ---- Lifecycle ----

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
        super().bind(
            name=name,
            seat=seat,
            dealt_role=dealt_role,
            seat_order=seat_order,
            role_pool=role_pool,
            language=language,
            bus=bus,
        )
        from ..state import PlayerState  # local import keeps engine off the import path

        ps = PlayerState(
            id=self.player_id, name=name, seat=seat,
            original_role=dealt_role, current_role=dealt_role,
        )
        self._system_prompt = build_system_prompt(ps, role_pool=self.role_pool)
        self._night_system_prompt = build_night_system_prompt(dealt_role)
        self._memory = PlayerMemory(
            player_id=self.player_id,
            seat=seat,
            name=name,
            assigned_role=dealt_role,
        )

    # ---- Observations ----

    def observe_night(self, step: str, text: str, structured: dict) -> None:
        if self._memory is not None:
            self._memory.add_observation(step, text, structured)

    def observe_speech(
        self, round_idx: int, speaker_id: str, text: str
    ) -> None:
        if self._memory is not None:
            self._memory.add_speech(
                Speech(round_idx=round_idx, speaker_id=speaker_id, text=text)
            )

    # observe_votes / observe_deaths use the inherited no-op defaults
    # for now; the LLM agent can read recent history from the speech
    # log it already accumulated.

    # ---- Decisions ----

    async def act_night(
        self, action_key: str, valid_targets: list[str]
    ) -> dict:
        # Night decisions don't need memory context， the task block names the role
        # and lists valid targets.
        role = _ACTION_KEY_TO_ROLE.get(action_key)
        if role is None or self._memory is None:
            return {}
        user_prompt = build_night_task(role, self.player_id, self.seat_order)
        parsed = await self._ask_json(user_prompt, system=self._night_system_prompt)
        if isinstance(parsed, dict):
            return parsed
        return {}

    async def speak(
        self, round_idx: int, total_rounds: int, max_chars: int
    ) -> str:
        if self._memory is None:
            return ""
        task = build_day_speech_task(
            round_idx, total_rounds, max_chars, dealt_role=self.dealt_role
        )
        user_prompt = self._memory.to_prompt_context("day") + "\n\n" + task
        parsed = await self._ask_json(user_prompt)
        if isinstance(parsed, dict):
            # Update beliefs BEFORE returning the speech, so any later
            # prompt rendering sees the freshest state.
            if isinstance(parsed.get("belief_state"), dict):
                self._memory.update_beliefs(parsed["belief_state"])
            if isinstance(parsed.get("speech"), str):
                return parsed["speech"].strip()
        return ""

    async def vote(self, valid_targets: list[str]) -> str:
        if self._memory is None:
            return self.player_id
        task = build_vote_task(valid_targets)
        user_prompt = self._memory.to_prompt_context("vote") + "\n\n" + task
        parsed = await self._ask_json(user_prompt)
        if isinstance(parsed, dict) and isinstance(parsed.get("vote"), str):
            return parsed["vote"]
        return self.player_id

    # ---- LLM plumbing ----

    async def _ask_json(
        self, user_prompt: str, *, system: str | None = None
    ) -> Any | None:
        """Send a completion; one retry with a strict reminder if the
        first response is unparseable. Returns the parsed value or None
        if both attempts fail. Emits a UserWarning carrying both raw
        responses when the fallback fires, so a misbehaving model is
        visible without spamming the console on every successful call.

        ``system`` overrides the default day/vote system prompt; used by
        ``act_night`` to send the trimmed night-only system prompt.
        """
        raw = await self._complete(user_prompt, system=system)
        parsed = _try_parse(raw)
        if parsed is not None:
            return parsed
        retry_prompt = (
            user_prompt
            + "\n\nIMPORTANT: Your previous reply was not valid JSON "
            "matching the required schema. Respond with VALID JSON only "
            "— no prose, no markdown fences."
        )
        raw2 = await self._complete(retry_prompt, system=system)
        parsed2 = _try_parse(raw2)
        if parsed2 is None:
            warnings.warn(
                f"[{self.player_id}] JSON parse failed after retry; "
                f"falling back to defaults. raw#1={raw!r} raw#2={raw2!r}",
                UserWarning,
                stacklevel=2,
            )
        return parsed2

    async def _complete(
        self, user_prompt: str, *, system: str | None = None
    ) -> str:
        # Stream when there is a bus to publish chunks to — otherwise
        # there is no observer that would benefit from live tokens.
        streaming = self.bus is not None

        def _emit_reasoning(delta: str) -> None:
            assert self.bus is not None
            self.bus.emit(
                ReasoningChunkEvent(player_id=self.player_id, delta=delta)
            )

        def _emit_content(delta: str) -> None:
            assert self.bus is not None
            self.bus.emit(
                ContentChunkEvent(player_id=self.player_id, delta=delta)
            )

        result = await self.client.complete(
            system=system if system is not None else self._system_prompt,
            user=user_prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            json_mode=self.json_mode,
            extra_body=self.extra_body,
            stream=streaming,
            on_reasoning_chunk=_emit_reasoning if streaming else None,
            on_content_chunk=_emit_content if streaming else None,
        )
        self.token_usage += result.usage
        if self.bus is not None:
            self.bus.emit(
                LLMCallEvent(
                    player_id=self.player_id,
                    reasoning=result.reasoning,
                    content=result.content,
                )
            )
        return result.content


def _try_parse(raw: str) -> Any | None:
    try:
        return extract_json(raw)
    except json.JSONDecodeError:
        return None
