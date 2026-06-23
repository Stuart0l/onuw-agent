from dataclasses import dataclass, field
from typing import Literal

from .state import Speech
from .types import Role


@dataclass
class NightObservation:
    step: str
    text: str
    structured: dict


@dataclass
class PlayerMemory:
    player_id: str
    seat: int
    name: str
    persona: str | None
    team_summary: str
    assigned_role: Role
    night_observations: list[NightObservation] = field(default_factory=list)
    conversation: list[Speech] = field(default_factory=list)
    own_speeches_drafts: list[str] = field(default_factory=list)
    # Private self-memory: player_id → one-line current belief. Replaced
    # in full each speak() turn so the agent always re-states its
    # current view, never accumulates stale entries.
    belief_state: dict[str, str] = field(default_factory=dict)

    def add_observation(self, step: str, text: str, structured: dict) -> None:
        self.night_observations.append(
            NightObservation(step=step, text=text, structured=structured)
        )

    def add_speech(self, speech: Speech) -> None:
        self.conversation.append(speech)

    def update_beliefs(self, raw: object) -> None:
        """Replace belief_state with the agent's latest dict. Drops non-
        dict input, non-string values, empty strings; caps each entry at
        200 chars to bound prompt growth."""
        if not isinstance(raw, dict):
            return
        cleaned: dict[str, str] = {}
        for pid, val in raw.items():
            if not isinstance(pid, str) or not isinstance(val, str):
                continue
            s = val.strip()
            if not s:
                continue
            cleaned[pid] = s[:200]
        self.belief_state = cleaned

    def to_prompt_context(self, phase: Literal["night", "day", "vote"]) -> str:
        return "\n\n".join(
            [
                self._identity_section(),
                self._team_section(),
                self._observations_section(),
                self._discussion_section(),
                self._beliefs_section(),
            ]
        )

    def _identity_section(self) -> str:
        # Persona is intentionally omitted here — it lives in the system
        # prompt's "== YOUR PERSONA ==" block, built once at setup.
        return (
            "== YOUR IDENTITY ==\n"
            f"You are {self.name} (seat {self.seat}, id {self.player_id}). "
            f"Your dealt role at the start of the night was: {self.assigned_role.value}."
        )

    def _team_section(self) -> str:
        return "== YOUR TEAM ==\n" + self.team_summary

    def _observations_section(self) -> str:
        header = "== WHAT YOU LEARNED DURING THE NIGHT =="
        if not self.night_observations:
            return f"{header}\nNothing."
        body = "\n".join(
            f"{i + 1}. {obs.text}" for i, obs in enumerate(self.night_observations)
        )
        return f"{header}\n{body}"

    def _discussion_section(self) -> str:
        header = "== PUBLIC DISCUSSION SO FAR =="
        if not self.conversation:
            return f"{header}\nNo discussion yet."
        rounds: dict[int, list[Speech]] = {}
        for sp in self.conversation:
            rounds.setdefault(sp.round_idx, []).append(sp)
        parts = [header]
        for r in sorted(rounds.keys()):
            parts.append(f"[Round {r + 1}]")
            for sp in rounds[r]:
                parts.append(f'  - {sp.speaker_id}: "{sp.text}"')
        return "\n".join(parts)

    def _beliefs_section(self) -> str:
        header = "== YOUR PRIVATE BELIEFS (from your last turn; only you see this) =="
        if not self.belief_state:
            return f"{header}\nNothing yet."
        body = "\n".join(
            f"- {pid}: {belief}" for pid, belief in self.belief_state.items()
        )
        return f"{header}\n{body}"
