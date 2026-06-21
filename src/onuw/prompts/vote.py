from ..memory import PlayerMemory
from .rules import SWAP_REMINDER


def build_vote_task(valid_targets: list[str]) -> str:
    return (
        SWAP_REMINDER + "\n\n"
        "== VOTING PHASE ==\n"
        "Cast your vote for the player you believe should be killed. Votes are "
        "simultaneous; you will not see others' votes until all are revealed.\n"
        f"Valid targets: {', '.join(valid_targets)}.\n"
        "You MAY vote for yourself (e.g. as Tanner).\n"
        "Respond with JSON ONLY:\n"
        '{"vote": "<player_id>", "rationale": "<one short sentence (private)>"}'
    )


def build_vote_prompt(memory: PlayerMemory, valid_targets: list[str]) -> str:
    return memory.to_prompt_context("vote") + "\n\n" + build_vote_task(valid_targets)