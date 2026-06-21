import json
from typing import Any

from ..llm.client import LLMClient
from ..utils.json_parse import extract_json
from .base import Agent


class LLMAgent(Agent):
    """Default LiteLLM-backed agent. One JSON-retry on parse failure,
    then a deterministic safe default."""

    def __init__(
        self,
        player_id: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 800,
        client: LLMClient | None = None,
    ) -> None:
        super().__init__(player_id)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = client or LLMClient()

    async def act_night(self, action_key: str, user_prompt: str) -> dict:
        parsed = await self._ask_json(user_prompt)
        if isinstance(parsed, dict):
            return parsed
        return {}

    async def speak(self, round_idx: int, user_prompt: str) -> str:
        parsed = await self._ask_json(user_prompt)
        if isinstance(parsed, dict) and isinstance(parsed.get("speech"), str):
            return parsed["speech"].strip()
        return "I have nothing to add."

    async def vote(self, user_prompt: str) -> str:
        parsed = await self._ask_json(user_prompt)
        if isinstance(parsed, dict) and isinstance(parsed.get("vote"), str):
            return parsed["vote"]
        return self.player_id

    async def _ask_json(self, user_prompt: str) -> Any | None:
        """Send a JSON-mode completion; one retry with a strict reminder
        if the first response is unparseable. Returns the parsed value
        or None if both attempts fail."""
        raw = await self._complete(user_prompt)
        parsed = _try_parse(raw)
        if parsed is not None:
            return parsed
        retry_prompt = (
            user_prompt
            + "\n\nIMPORTANT: Your previous reply was not valid JSON "
            "matching the required schema. Respond with VALID JSON only "
            "— no prose, no markdown fences."
        )
        raw2 = await self._complete(retry_prompt)
        return _try_parse(raw2)

    async def _complete(self, user_prompt: str) -> str:
        return await self.client.complete(
            system=self.system_prompt,
            user=user_prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            json_mode=True,
        )


def _try_parse(raw: str) -> Any | None:
    try:
        return extract_json(raw)
    except json.JSONDecodeError:
        return None
