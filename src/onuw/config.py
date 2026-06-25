from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from .types import Role


class PlayerConfig(BaseModel):
    id: str
    name: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 800
    # API-level response_format={"type":"json_object"}. Default off —
    # OpenAI / Anthropic accept it but LM Studio and many self-hosted
    # backends reject it; the tolerant parser handles both.
    json_mode: bool = False
    # Provider-specific request body params forwarded verbatim
    # (e.g. MiniMax thinking config, DeepSeek tool toggles).
    extra_body: dict = Field(default_factory=dict)


class GameConfig(BaseModel):
    players: list[PlayerConfig] = Field(min_length=3, max_length=10)
    role_pool: list[Role]
    discussion_rounds: int = 3
    max_speech_chars: int = 600
    seed: int | None = None
    log_dir: Path = Path("logs")
    console: bool = True

    @model_validator(mode="after")
    def _check_invariants(self) -> "GameConfig":
        expected = len(self.players) + 3
        if len(self.role_pool) != expected:
            raise ValueError(
                f"role_pool must contain len(players) + 3 = {expected} entries; "
                f"got {len(self.role_pool)}"
            )
        ids = [p.id for p in self.players]
        if len(set(ids)) != len(ids):
            raise ValueError("player ids must be unique")
        return self
