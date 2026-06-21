from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResult:
    """One complete call's worth of usable signal from the LLM.

    ``reasoning`` carries the chain-of-thought from reasoning models
    (e.g. MiniMax-M3's ``reasoning_content`` field) — empty string for
    non-reasoning models.
    """

    content: str
    usage: "TokenUsage"
    reasoning: str = ""


@dataclass
class TokenUsage:
    """Per-call token accounting. Lives on every Agent so the engine can
    sum across players regardless of which LLM backend (LiteLLM,
    LangChain, etc.) is in use. Non-LLM agents leave it at zero.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __iadd__(self, other: "TokenUsage") -> "TokenUsage":
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        return self

    @classmethod
    def from_response(cls, resp: Any) -> "TokenUsage":
        u = getattr(resp, "usage", None)
        if u is None:
            return cls()
        return cls(
            prompt_tokens=int(getattr(u, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(u, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(u, "total_tokens", 0) or 0),
        )


__all__ = ["LLMResult", "TokenUsage"]
