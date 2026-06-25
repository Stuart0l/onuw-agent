from dataclasses import dataclass, field
from typing import Literal

from .state import Speech
from .types import Role


_LOCKED_ROLES = {Role.TROUBLEMAKER, Role.INSOMNIAC}


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
    assigned_role: Role
    night_observations: list[NightObservation] = field(default_factory=list)
    conversation: list[Speech] = field(default_factory=list)
    own_speeches_drafts: list[str] = field(default_factory=list)
    # Private self-memory: player_id → {role, confidence, evidence}.
    # The agent's structured hypothesis about each speaking player.
    # Replaced in full each speak() turn so the agent always re-states
    # its current view, never accumulates stale entries.
    per_player_hypothesis: dict[str, dict[str, str]] = field(default_factory=dict)
    # The agent's committed CURRENT role. None until the agent picks
    # one in their first speak() turn. Once set, prompt rendering
    # swaps "dealt role + raw night obs" for "committed role only" so
    # the agent stops re-deriving its role each round. Important night
    # observations are expected to be folded into per_player_hypothesis
    # at commit time, so we don't carry a separate raw-fact slot.
    committed_role: Role | None = None

    def __post_init__(self) -> None:
        # TM / Insomniac roles cannot be swapped during the night, so
        # the dealt role IS the committed role for the whole game.
        if self.assigned_role in _LOCKED_ROLES:
            self.committed_role = self.assigned_role

    def add_observation(self, step: str, text: str, structured: dict) -> None:
        self.night_observations.append(
            NightObservation(step=step, text=text, structured=structured)
        )

    def add_speech(self, speech: Speech) -> None:
        self.conversation.append(speech)

    def commit_role(self, role: Role) -> None:
        """Set committed_role. Silently ignored for locked roles
        (TM / Insomniac) since their card cannot be swapped."""
        if self.assigned_role in _LOCKED_ROLES:
            return
        self.committed_role = role

    _CONFIDENCE_VALUES = frozenset({"high", "medium", "low"})

    def update_per_player_hypothesis(self, raw: object) -> None:
        """Replace per_player_hypothesis with the agent's latest dict.
        Each entry must carry a valid role enum value, a confidence in
        {high|medium|low}, and a non-empty evidence string (capped at
        150 chars). Invalid entries are dropped silently."""
        if not isinstance(raw, dict):
            return
        cleaned: dict[str, dict[str, str]] = {}
        for pid, entry in raw.items():
            if not isinstance(pid, str) or not isinstance(entry, dict):
                continue
            role = entry.get("role")
            confidence = entry.get("confidence")
            evidence = entry.get("evidence")
            if not (
                isinstance(role, str)
                and isinstance(confidence, str)
                and isinstance(evidence, str)
            ):
                continue
            role_clean = role.strip().lower()
            try:
                Role(role_clean)
            except ValueError:
                continue
            conf_clean = confidence.strip().lower()
            if conf_clean not in self._CONFIDENCE_VALUES:
                continue
            ev_clean = evidence.strip()[:150]
            if not ev_clean:
                continue
            cleaned[pid] = {
                "role": role_clean,
                "confidence": conf_clean,
                "evidence": ev_clean,
            }
        self.per_player_hypothesis = cleaned

    def to_prompt_context(self, phase: Literal["night", "day", "vote"]) -> str:
        parts = [self._identity_section()]
        # Raw night observations are only useful before commit. After
        # commit, the agent is expected to have folded important night
        # info into per_player_hypothesis — so we drop the observation
        # list entirely.
        if self.committed_role is None:
            parts.append(self._observations_section())
        parts.append(self._discussion_section())
        parts.append(self._hypothesis_section())
        return "\n\n".join(parts)

    def _identity_section(self) -> str:
        header = (
            "== YOUR IDENTITY ==\n"
            f"You are {self.name} (seat {self.seat}, id {self.player_id}). "
        )
        if self.committed_role is None:
            return (
                header
                + "Your dealt role at the start of the night was: "
                + f"{self.assigned_role.value}."
            )
        return (
            header
            + "Your CURRENT (committed) role is: "
            + f"{self.committed_role.value}."
        )

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

    def _hypothesis_section(self) -> str:
        header = "== YOUR PER-PLAYER HYPOTHESIS (from your last turn; only you see this) =="
        if not self.per_player_hypothesis:
            return f"{header}\nNothing yet."
        body = "\n".join(
            f"- {pid}: {h['role']} ({h['confidence']}) — {h['evidence']}"
            for pid, h in self.per_player_hypothesis.items()
        )
        return f"{header}\n{body}"
