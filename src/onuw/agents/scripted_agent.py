from ..memory import NightObservation
from ..state import Speech
from .base import Agent


class ScriptedAgent(Agent):
    """Deterministic agent for testing.

    Returns canned responses keyed by phase:
      - night: dict[action_key, action_dict]
      - day:   dict[round_idx, speech]
      - vote:  player_id string
    Missing keys fall back to safe defaults (empty dict / no-op string /
    vote-for-self).

    Records observed events (night, speeches, votes, deaths) on public
    attributes so tests can verify what the engine pushed to this agent.
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
        self.night_observations: list[NightObservation] = []
        self.speeches: list[Speech] = []
        self.observed_votes: dict[str, str] | None = None
        self.observed_deaths: tuple[list[str], list[tuple[str, str]]] | None = None

    def observe_night(self, step: str, text: str, structured: dict) -> None:
        self.night_observations.append(
            NightObservation(step=step, text=text, structured=structured)
        )

    def observe_speech(
        self, round_idx: int, speaker_id: str, text: str
    ) -> None:
        self.speeches.append(
            Speech(round_idx=round_idx, speaker_id=speaker_id, text=text)
        )

    def observe_votes(self, votes: dict[str, str]) -> None:
        self.observed_votes = dict(votes)

    def observe_deaths(
        self,
        deaths: list[str],
        hunter_revenge: list[tuple[str, str]],
    ) -> None:
        self.observed_deaths = (list(deaths), list(hunter_revenge))

    async def act_night(
        self, action_key: str, valid_targets: list[str]
    ) -> dict:
        return self._night.get(action_key, {})

    async def speak(
        self, round_idx: int, total_rounds: int, max_chars: int
    ) -> str:
        return self._day.get(round_idx, "I have nothing to add.")

    async def vote(self, valid_targets: list[str]) -> str:
        if self._vote is None:
            return self.player_id
        return self._vote
