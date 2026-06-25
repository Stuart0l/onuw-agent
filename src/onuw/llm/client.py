import asyncio
import random
import warnings
from collections.abc import Callable
from typing import Any

from . import LLMResult, TokenUsage


class _ThinkSplitter:
    """Stateful splitter for inline ``<think>`` tags. ``feed(delta)``
    returns ``(content_out, reasoning_out)`` for the delta.

    Edge case: a tag split exactly across a chunk boundary routes those
    few bytes to whichever side was current — rare for typical LLM
    chunk sizes, traded for a simpler implementation."""

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self.in_think = False

    def feed(self, delta: str) -> tuple[str, str]:
        c_parts: list[str] = []
        r_parts: list[str] = []
        i = 0
        while i < len(delta):
            if self.in_think:
                idx = delta.find(self.CLOSE, i)
                if idx == -1:
                    r_parts.append(delta[i:])
                    break
                if idx > i:
                    r_parts.append(delta[i:idx])
                i = idx + len(self.CLOSE)
                self.in_think = False
            else:
                idx = delta.find(self.OPEN, i)
                if idx == -1:
                    c_parts.append(delta[i:])
                    break
                if idx > i:
                    c_parts.append(delta[i:idx])
                i = idx + len(self.OPEN)
                self.in_think = True
        return "".join(c_parts), "".join(r_parts)


class _SyntheticDelta:
    """Mimics a streaming ``delta`` so non-streaming responses share
    the chunk loop."""

    def __init__(self, content: str, reasoning_content: str) -> None:
        self.content = content
        self.reasoning_content = reasoning_content


class _SyntheticChoice:
    def __init__(self, delta: _SyntheticDelta) -> None:
        self.delta = delta


class _SyntheticChunk:
    def __init__(self, content: str, reasoning_content: str, usage: Any) -> None:
        self.choices = [_SyntheticChoice(_SyntheticDelta(content, reasoning_content))]
        self.usage = usage


async def _wrap_as_single_chunk(resp: Any):
    """Yield one synthetic chunk carrying the full non-streaming
    response so ``_process_chunks`` is the only code path."""
    msg = resp.choices[0].message
    content = getattr(msg, "content", None) or ""
    reasoning = (
        getattr(msg, "reasoning_content", None)
        or getattr(msg, "reasoning", None)
        or ""
    )
    yield _SyntheticChunk(
        content=content,
        reasoning_content=reasoning,
        usage=getattr(resp, "usage", None),
    )


class LLMClient:
    """Async wrapper around ``litellm.acompletion`` with exponential
    backoff. Only this module imports litellm. Tests inject a fake by
    subclassing and overriding ``_acompletion``."""

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
        stream: bool = False,
        on_reasoning_chunk: Callable[[str], None] | None = None,
        on_content_chunk: Callable[[str], None] | None = None,
    ) -> LLMResult:
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
        if stream:
            kwargs["stream"] = True
            # Ensure the final chunk still carries usage info.
            kwargs["stream_options"] = {"include_usage": True}
            chunks = await self._call_with_backoff(kwargs)
        else:
            resp = await self._call_with_backoff(kwargs)
            chunks = _wrap_as_single_chunk(resp)
        return await self._process_chunks(
            chunks, on_reasoning_chunk, on_content_chunk
        )

    @staticmethod
    async def _process_chunks(
        chunks: Any,
        on_reasoning_chunk: Callable[[str], None] | None,
        on_content_chunk: Callable[[str], None] | None,
    ) -> LLMResult:
        """One chunk loop for both streaming and non-streaming paths
        (non-streaming responses arrive as a one-shot synthetic chunk).
        ``_ThinkSplitter`` strips inline ``<think>`` tags from content
        deltas so callbacks never see them."""
        splitter = _ThinkSplitter()
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage = TokenUsage()
        try:
            async for chunk in chunks:
                choices = getattr(chunk, "choices", None) or []
                if choices:
                    delta = getattr(choices[0], "delta", None)
                    if delta is not None:
                        rd = getattr(delta, "reasoning_content", None) or getattr(
                            delta, "reasoning", None
                        )
                        if rd:
                            reasoning_parts.append(rd)
                            if on_reasoning_chunk is not None:
                                on_reasoning_chunk(rd)
                        cd = getattr(delta, "content", None)
                        if cd:
                            c_out, r_out = splitter.feed(cd)
                            if c_out:
                                content_parts.append(c_out)
                                if on_content_chunk is not None:
                                    on_content_chunk(c_out)
                            if r_out:
                                reasoning_parts.append(r_out)
                                if on_reasoning_chunk is not None:
                                    on_reasoning_chunk(r_out)
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = TokenUsage.from_response(chunk)
        except Exception as exc:  # noqa: BLE001
            # Providers (notably MiniMax) sometimes finish a stream with
            # finish_reason="error" mid-flight. Return whatever we
            # accumulated; the agent's retry-then-default handles it.
            warnings.warn(
                f"Stream interrupted mid-flight ({type(exc).__name__}: {exc}); "
                f"returning partial result ({len(content_parts)} content "
                f"chunks, {len(reasoning_parts)} reasoning chunks accumulated).",
                UserWarning,
                stacklevel=2,
            )

        return LLMResult(
            content="".join(content_parts).strip(),
            usage=usage,
            reasoning="".join(reasoning_parts).strip(),
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
    def _is_retryable(exc: BaseException) -> bool:
        return type(exc).__name__ in {
            "RateLimitError",
            "APIConnectionError",
            "APITimeoutError",
            "ServiceUnavailableError",
            "InternalServerError",
        }
