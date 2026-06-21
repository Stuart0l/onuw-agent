from onuw.memory import PlayerMemory
from onuw.prompts.rules import (
    GAME_RULES_BLOCK,
    OUTPUT_FORMAT_PREAMBLE,
    ROLE_ABILITY_BLOCKS,
    WIN_CONDITIONS_BLOCK,
)
from onuw.prompts.system import build_system_prompt
from onuw.state import PlayerState, Speech
from onuw.types import Role


def _player(role: Role = Role.SEER) -> PlayerState:
    return PlayerState(
        id="p1",
        name="Alice",
        seat=1,
        original_role=role,
        current_role=role,
    )


def _memory(role: Role = Role.SEER, persona: str | None = None) -> PlayerMemory:
    return PlayerMemory(
        player_id="p1",
        seat=1,
        name="Alice",
        persona=persona,
        rules_text=GAME_RULES_BLOCK,
        role_ability_text=ROLE_ABILITY_BLOCKS[role],
        win_conditions_text=WIN_CONDITIONS_BLOCK,
        assigned_role=role,
    )


def test_role_ability_blocks_cover_all_roles():
    for r in Role:
        assert r in ROLE_ABILITY_BLOCKS, f"missing ability block for {r}"
        assert ROLE_ABILITY_BLOCKS[r].strip()


def test_to_prompt_context_has_all_sections_when_empty():
    out = _memory().to_prompt_context("night")
    for header in [
        "== YOUR IDENTITY ==",
        "== ROLE ABILITY (your dealt role) ==",
        "== WIN CONDITIONS (all teams) ==",
        "== WHAT YOU LEARNED DURING THE NIGHT ==",
        "== PUBLIC DISCUSSION SO FAR ==",
    ]:
        assert header in out
    assert "Nothing." in out
    assert "No discussion yet." in out


def test_observation_renders_into_what_you_learned_section():
    m = _memory()
    m.add_observation(
        "seer_action",
        "You looked at p3's card; it is WEREWOLF.",
        {"target": "p3", "role": "werewolf"},
    )
    out = m.to_prompt_context("night")
    learned_idx = out.index("== WHAT YOU LEARNED DURING THE NIGHT ==")
    discussion_idx = out.index("== PUBLIC DISCUSSION SO FAR ==")
    section = out[learned_idx:discussion_idx]
    assert "You looked at p3's card; it is WEREWOLF." in section
    assert "Nothing." not in section
    assert "1." in section


def test_speeches_render_grouped_by_round_and_in_order():
    m = _memory()
    m.add_speech(Speech(round_idx=0, speaker_id="p2", text="I am the Villager."))
    m.add_speech(Speech(round_idx=0, speaker_id="p3", text="I doubt that."))
    m.add_speech(Speech(round_idx=1, speaker_id="p2", text="Fine, I'm Robber."))
    out = m.to_prompt_context("day")
    assert "[Round 1]" in out
    assert "[Round 2]" in out
    assert out.index("I am the Villager.") < out.index("I doubt that.") < out.index("Fine, I'm Robber.")
    assert "No discussion yet." not in out


def test_system_prompt_includes_own_role_ability_only():
    out = build_system_prompt(_player(Role.SEER), persona=None)
    assert ROLE_ABILITY_BLOCKS[Role.SEER] in out
    for other in Role:
        if other == Role.SEER:
            continue
        assert ROLE_ABILITY_BLOCKS[other] not in out


def test_system_prompt_persona_block_appears_only_when_set():
    out_none = build_system_prompt(_player(), persona=None)
    assert "YOUR PERSONA" not in out_none
    out_with = build_system_prompt(_player(), persona="A cautious librarian.")
    assert "YOUR PERSONA" in out_with
    assert "A cautious librarian." in out_with


def test_system_prompt_contains_required_layers():
    out = build_system_prompt(_player(), persona=None)
    assert "GAME RULES" in out
    assert "WIN CONDITIONS" in out
    assert OUTPUT_FORMAT_PREAMBLE in out