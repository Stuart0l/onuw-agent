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
    # Private self-memory: player_id → one-line current belief. Replaced
    # in full each speak() turn so the agent always re-states its
    # current view, never accumulates stale entries.
    belief_state: dict[str, str] = field(default_factory=dict)
    # The agent's committed CURRENT role. None until the agent picks
    # one in their first speak() turn. Once set, prompt rendering
    # swaps "dealt role + raw night obs" for "committed role only" so
    # the agent stops re-deriving its role each round. Important night
    # observations are expected to be folded into belief_state at
    # commit time (with provenance + uncertainty), so we don't carry a
    # separate raw-fact slot — it overlapped with belief_state.
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
        parts = [self._identity_section()]
        # Raw night observations are only useful before commit. After
        # commit, the agent is expected to have folded important night
        # info into belief_state with provenance — so we drop the
        # observation list entirely.
        if self.committed_role is None:
            parts.append(self._observations_section())
        parts.append(self._discussion_section())
        parts.append(self._beliefs_section())
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

    def _beliefs_section(self) -> str:
        header = "== YOUR PRIVATE BELIEFS (from your last turn; only you see this) =="
        if not self.belief_state:
            return f"{header}\nNothing yet."
        body = "\n".join(
            f"- {pid}: {belief}" for pid, belief in self.belief_state.items()
        )
        return f"{header}\n{body}"
