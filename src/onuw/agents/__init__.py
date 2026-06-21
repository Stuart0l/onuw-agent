from typing import Callable

from ..config import PlayerConfig
from .base import Agent

AgentFactory = Callable[[PlayerConfig], Agent]

__all__ = ["Agent", "AgentFactory"]