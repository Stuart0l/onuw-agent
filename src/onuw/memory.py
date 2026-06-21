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
    rules_text: str
    role_ability_text: str
    win_conditions_text: str
    assigned_role: Role
    night_observations: list[NightObservation] = field(default_factory=list)
    conversation: list[Speech] = field(default_factory=list)
    own_speeches_drafts: list[str] = field(default_factory=list)

    def add_observation(self, step: str, text: str, structured: dict) -> None:
        self.night_observations.append(
            NightObservation(step=step, text=text, structured=structured)
        )

    def add_speech(self, speech: Speech) -> None:
        self.conversation.append(speech)

    def to_prompt_context(self, phase: Literal["night", "day", "vote"]) -> str:
        return "\n\n".join(
            [
                self._identity_section(),
                self._role_ability_section(),
                self._win_conditions_section(),
                self._observations_section(),
                self._discussion_section(),
            ]
        )

    def _identity_section(self) -> str:
        lines = [
            "== YOUR IDENTITY ==",
            f"You are {self.name} (seat {self.seat}, id {self.player_id}). "
            f"Your dealt role at the start of the night was: {self.assigned_role.value}.",
        ]
        if self.persona:
            lines.append(f"Persona: {self.persona}")
        return "\n".join(lines)

    def _role_ability_section(self) -> str:
        return "== ROLE ABILITY (your dealt role) ==\n" + self.role_ability_text

    def _win_conditions_section(self) -> str:
        return "== WIN CONDITIONS (all teams) ==\n" + self.win_conditions_text

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
