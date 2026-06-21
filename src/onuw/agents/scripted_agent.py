from .base import Agent


class ScriptedAgent(Agent):
    """Deterministic agent for testing.

    Responses are keyed by phase:
      - night: dict[action_key, action_dict]
      - day:   dict[round_idx, speech]
      - vote:  player_id string
    Missing keys fall back to safe defaults (empty dict / no-op string /
    vote-for-self).
    """

    def __init__(
        self,
        player_id: str,
        night: dict[str, dict] | None = None,
        day: dict[int, str] | None = None,
        vote: str | None = None,
    ) -> None:
        super().__init__(player_id)
        self._night = night or {}
        self._day = day or {}
        self._vote = vote

    async def act_night(self, action_key: str, user_prompt: str) -> dict:
        return self._night.get(action_key, {})

    async def speak(self, round_idx: int, user_prompt: str) -> str:
        return self._day.get(round_idx, "I have nothing to add.")

    async def vote(self, user_prompt: str) -> str:
        if self._vote is None:
            return self.player_id
        return self._vote