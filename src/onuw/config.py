from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from .types import Role


class PlayerConfig(BaseModel):
    id: str
    name: str
    model: str
    persona: str | None = None
    temperature: float = 0.7
    max_tokens: int = 800
    # Enable response_format={"type":"json_object"} at the API level.
    # Default off: OpenAI / Anthropic support it (set to true for stronger
    # output guarantees) but LM Studio and many self-hosted servers
    # reject it. The tolerant parser in utils.json_parse covers both.
    json_mode: bool = False


class GameConfig(BaseModel):
    players: list[PlayerConfig] = Field(min_length=3, max_length=10)
    role_pool: list[Role]
    discussion_rounds: int = 3
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
