import asyncio
import random
import re
from typing import Any

from . import LLMResult, TokenUsage


_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _split_think_tags(content: str) -> tuple[str, str]:
    """Return ``(stripped_content, inline_reasoning)``.

    Removes every ``<think>...</think>`` block from the content and
    concatenates their bodies as the reasoning. If no tags are present,
    returns the content unchanged and an empty reasoning string.
    """
    if "<think>" not in content.lower():
        return content, ""
    pieces: list[str] = []

    def _grab(m: re.Match) -> str:
        pieces.append(m.group(1).strip())
        return ""

    stripped = _THINK_TAG_RE.sub(_grab, content).strip()
    return stripped, "\n\n".join(p for p in pieces if p)


class LLMClient:
    """Async wrapper around ``litellm.acompletion`` with exponential
    backoff. Only this module imports litellm so the rest of the codebase
    can be exercised (e.g. with ScriptedAgent) without provider deps.

    Tests inject a fake by subclassing and overriding ``_acompletion``.
    """

    def __init__(
        self,
        default_temperature: float = 0.7,
        max_retries: int = 4,
        backoff_base: float = 1.5,
        backoff_cap: float = 30.0,
    ) -> None:
        self.default_temperature = default_temperature
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap

    async def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int = 800,
        json_mode: bool = False,
        extra_body: dict | None = None,
    ) -> tuple[str, TokenUsage]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": (
                temperature
                if temperature is not None
                else self.default_temperature
            ),
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if extra_body:
            # Forwarded verbatim into the upstream request body — escape
            # hatch for provider-specific fields like MiniMax `thinking`.
            kwargs["extra_body"] = extra_body
        resp = await self._call_with_backoff(kwargs)
        return LLMResult(
            content=self._extract_content(resp),
            usage=TokenUsage.from_response(resp),
            reasoning=self._extract_reasoning(resp),
        )

    async def _call_with_backoff(self, kwargs: dict) -> Any:
        attempt = 0
        while True:
            try:
                return await self._acompletion(kwargs)
            except Exception as exc:  # noqa: BLE001
                if not self._is_retryable(exc) or attempt >= self.max_retries:
                    raise
                delay = min(
                    self.backoff_base**attempt + random.random(),
                    self.backoff_cap,
                )
                await asyncio.sleep(delay)
                attempt += 1

    async def _acompletion(self, kwargs: dict) -> Any:
        import litellm  # lazy import so import-time has no litellm cost

        return await litellm.acompletion(**kwargs)

    @staticmethod
    def _extract_content(resp: Any) -> str:
        """Final answer with any ``<think>...</think>`` block stripped."""
        content = resp.choices[0].message.content or ""
        stripped, _ = _split_think_tags(content)
        return stripped

    @staticmethod
    def _extract_reasoning(resp: Any) -> str:
        """Pull the chain-of-thought out of reasoning-model responses.

        Order of attempts:
          1. ``message.reasoning_content`` — MiniMax-M3, DeepSeek-R1 native API.
          2. ``message.reasoning`` — older / variant providers.
          3. ``<think>...</think>`` block inline in ``message.content`` —
             DeepSeek-R1 via OpenAI-compatible endpoint and similar.
        """
        msg = resp.choices[0].message
        for key in ("reasoning_content", "reasoning"):
            v = getattr(msg, key, None)
            if v:
                return str(v)
        content = getattr(msg, "content", None) or ""
        _, inline = _split_think_tags(content)
        return inline

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        return type(exc).__name__ in {
            "RateLimitError",
            "APIConnectionError",
            "APITimeoutError",
            "ServiceUnavailableError",
            "InternalServerError",
        }
