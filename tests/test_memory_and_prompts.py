from onuw.memory import PlayerMemory
from onuw.prompts.day import build_day_speech_task
from onuw.prompts.rules import (
    OUTPUT_FORMAT_PREAMBLE,
    ROLE_ABILITY_BLOCKS,
    swap_reminder,
    team_summary,
)
from onuw.prompts.system import build_system_prompt
from onuw.prompts.vote import build_vote_task
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
        team_summary=team_summary(role),
        assigned_role=role,
    )


def test_role_ability_blocks_cover_all_roles():
    for r in Role:
        assert r in ROLE_ABILITY_BLOCKS, f"missing ability block for {r}"
        assert ROLE_ABILITY_BLOCKS[r].strip()


def test_to_prompt_context_has_expected_sections_when_empty():
    out = _memory().to_prompt_context("night")
    for header in [
        "== YOUR IDENTITY ==",
        "== YOUR TEAM ==",
        "== WHAT YOU LEARNED DURING THE NIGHT ==",
        "== PUBLIC DISCUSSION SO FAR ==",
    ]:
        assert header in out
    assert "Nothing." in out
    assert "No discussion yet." in out


def test_to_prompt_context_omits_sections_now_in_system_prompt():
    # Role ability, all-teams win conditions, and persona are in the
    # system prompt; the user prompt must not duplicate them.
    out = _memory(persona="A cautious librarian.").to_prompt_context("night")
    assert "== ROLE ABILITY" not in out
    assert "== WIN CONDITIONS (all teams) ==" not in out
    assert "Persona:" not in out
    assert "A cautious librarian." not in out


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


# ----- team_summary -----

def test_team_summary_werewolf_and_minion_are_werewolf_team():
    assert "WEREWOLF team" in team_summary(Role.WEREWOLF)
    assert "WEREWOLF team" in team_summary(Role.MINION)


def test_team_summary_village_roles_say_village_team():
    for r in (Role.VILLAGER, Role.MASON, Role.SEER, Role.INSOMNIAC,
              Role.HUNTER, Role.TROUBLEMAKER):
        assert "VILLAGE team" in team_summary(r), r


def test_team_summary_tanner_is_solo_team():
    assert "TANNER" in team_summary(Role.TANNER)


def test_team_summary_robber_signposts_team_change_from_steal():
    out = team_summary(Role.ROBBER)
    assert "stolen role" in out or "stolen" in out
    assert "night observation" in out


def test_team_summary_drunk_signposts_unknown_team():
    out = team_summary(Role.DRUNK)
    assert "UNKNOWN" in out


# ----- swap reminder -----

def test_day_task_includes_swap_reminder():
    # Without dealt_role the swap reminder is omitted.
    assert "YOUR CURRENT ROLE" not in build_day_speech_task(0, 3, max_chars=600)
    # With dealt_role the role-specific commit instructions are included.
    out = build_day_speech_task(0, 3, max_chars=600, dealt_role=Role.WEREWOLF)
    assert "YOUR CURRENT ROLE" in out
    assert "DO NOT revisit" in out
    # Certain roles get a flat assertion, not the reasoning prompt.
    out_tm = build_day_speech_task(0, 3, max_chars=600, dealt_role=Role.TROUBLEMAKER)
    assert "still the Troublemaker" in out_tm
    out_ins = build_day_speech_task(0, 3, max_chars=600, dealt_role=Role.INSOMNIAC)
    assert "Insomniac wake observation IS your current role" in out_ins


def test_vote_task_includes_swap_reminder():
    assert "YOUR CURRENT ROLE" not in build_vote_task(["p1", "p2", "p3"])
    out = build_vote_task(["p1", "p2", "p3"], dealt_role=Role.SEER)
    assert "YOUR CURRENT ROLE" in out
    assert "DO NOT revisit" in out
