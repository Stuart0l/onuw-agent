import asyncio
import random
from typing import Any


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
    ) -> str:
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
        return await self._call_with_backoff(kwargs)

    async def _call_with_backoff(self, kwargs: dict) -> str:
        attempt = 0
        while True:
            try:
                resp = await self._acompletion(kwargs)
                return self._extract_content(resp)
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
        return resp.choices[0].message.content or ""

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        return type(exc).__name__ in {
            "RateLimitError",
            "APIConnectionError",
            "APITimeoutError",
            "ServiceUnavailableError",
            "InternalServerError",
        }
