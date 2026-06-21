from typing import Callable

from ..config import PlayerConfig
from ..llm.client import LLMClient
from .base import Agent
from .llm_agent import LLMAgent

AgentFactory = Callable[[PlayerConfig], Agent]


def default_factory(client: LLMClient | None = None) -> AgentFactory:
    """Return an AgentFactory that builds LiteLLM-backed LLMAgents.

    The same ``LLMClient`` instance is shared across all seats so that
    per-provider rate limits and backoff state are coordinated.
    """
    shared = client or LLMClient()

    def factory(pcfg: PlayerConfig) -> Agent:
        return LLMAgent(
            player_id=pcfg.id,
            model=pcfg.model,
            temperature=pcfg.temperature,
            max_tokens=pcfg.max_tokens,
            client=shared,
        )

    return factory


__all__ = ["Agent", "AgentFactory", "LLMAgent", "default_factory"]
