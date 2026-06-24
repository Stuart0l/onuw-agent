from onuw.memory import PlayerMemory
from onuw.prompts.day import build_day_speech_task
from onuw.prompts.rules import (
    OUTPUT_FORMAT_PREAMBLE,
    ROLE_ABILITY_BLOCKS,
)
from onuw.prompts.system import build_night_system_prompt, build_system_prompt
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


def _memory(role: Role = Role.SEER) -> PlayerMemory:
    return PlayerMemory(
        player_id="p1",
        seat=1,
        name="Alice",
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
        "== WHAT YOU LEARNED DURING THE NIGHT ==",
        "== PUBLIC DISCUSSION SO FAR ==",
    ]:
        assert header in out
    # YOUR TEAM section is intentionally absent — the agent derives its
    # team from the role abilities + win conditions in the system prompt.
    assert "== YOUR TEAM ==" not in out
    assert "Nothing." in out
    assert "No discussion yet." in out


def test_to_prompt_context_omits_sections_now_in_system_prompt():
    # Role ability and all-teams win conditions are in the system prompt;
    # the user prompt must not duplicate them.
    out = _memory().to_prompt_context("night")
    assert "== ROLE ABILITY" not in out
    assert "== WIN CONDITIONS (all teams) ==" not in out


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
    out = build_system_prompt(_player(Role.SEER))
    assert ROLE_ABILITY_BLOCKS[Role.SEER] in out
    for other in Role:
        if other == Role.SEER:
            continue
        assert ROLE_ABILITY_BLOCKS[other] not in out


def test_system_prompt_contains_required_layers():
    out = build_system_prompt(_player())
    assert "GAME RULES" in out
    assert "WIN CONDITIONS" in out
    assert OUTPUT_FORMAT_PREAMBLE in out


def test_night_system_prompt_is_minimal():
    out = build_night_system_prompt(Role.ROBBER)
    # Own role only — no rules, win conditions, deck, or other roles.
    assert ROLE_ABILITY_BLOCKS[Role.ROBBER] in out
    assert "GAME RULES" not in out
    assert "WIN CONDITIONS" not in out
    assert "TABLE" not in out
    assert "ROLES IN THIS GAME" not in out
    for other in Role:
        if other == Role.ROBBER:
            continue
        assert ROLE_ABILITY_BLOCKS[other] not in out
    assert OUTPUT_FORMAT_PREAMBLE in out


# ----- thinking guide -----

def test_day_task_uncommitted_guide_for_unlocked_role():
    # Without dealt_role the guide is omitted (older callsites).
    assert "THINKING GUIDE" not in build_day_speech_task(0, 3, max_chars=600)
    # Unlocked role with no committed role → 6-step uncommitted guide.
    out = build_day_speech_task(0, 3, max_chars=600, dealt_role=Role.WEREWOLF)
    assert "THINKING GUIDE" in out
    assert "STEP 1 — RECALL" in out
    assert "STEP 3 — COMMIT CURRENT ROLE" in out
    assert "STEP 5 — STRATEGY" in out
    assert "STEP 6 — WRITE THE JSON" in out
    assert "Do NOT loop back" in out


def test_day_task_locked_guide_for_troublemaker_or_insomniac():
    # TM / Insomniac roles cannot be swapped — they get the locked
    # 4-step guide with no commit / no review step.
    out_tm = build_day_speech_task(0, 3, max_chars=600, dealt_role=Role.TROUBLEMAKER)
    assert "your role is troublemaker" in out_tm.lower()
    assert "fixed for the whole game" in out_tm
    assert "COMMIT CURRENT ROLE" not in out_tm
    assert "STEP 4 — WRITE THE JSON" in out_tm

    out_ins = build_day_speech_task(0, 3, max_chars=600, dealt_role=Role.INSOMNIAC)
    assert "your role is insomniac" in out_ins.lower()
    assert "COMMIT CURRENT ROLE" not in out_ins


def test_day_guide_uncommitted_injects_role_specific_commit_rule():
    # Robber + generic-other path through _commit_role_rule.
    out_rob = build_day_speech_task(0, 3, max_chars=600, dealt_role=Role.ROBBER)
    assert "the card you stole" in out_rob
    out_other = build_day_speech_task(0, 3, max_chars=600, dealt_role=Role.SEER)
    assert "may have been swapped" in out_other


def test_day_task_committed_guide_uses_committed_role():
    # Non-locked role + committed_role set → 5-step refine guide.
    out = build_day_speech_task(
        0, 3, max_chars=600,
        dealt_role=Role.ROBBER, committed_role=Role.WEREWOLF,
    )
    assert "your committed role is werewolf" in out.lower()
    assert "STEP 3 — REVIEW COMMITTED ROLE" in out
    assert "STEP 5 — WRITE THE JSON" in out
    assert "COMMIT CURRENT ROLE" not in out


def test_vote_task_includes_thinking_guide():
    out = build_vote_task(["p1", "p2", "p3"])
    assert "THINKING GUIDE" in out
    assert "STEP 1 — RECALL" in out
    assert "STEP 3 — PICK" in out
    assert "STEP 4 — WRITE" in out
    # The whole point of the rewrite: pull from prior beliefs, don't
    # re-derive the game state at vote time.
    assert "PRIVATE BELIEFS" in out
    assert "do NOT re-derive" in out


# ----- belief state -----

def test_update_beliefs_accepts_clean_dict():
    m = _memory()
    m.update_beliefs({"p2": "likely Werewolf", "p3": "Mason, trusted"})
    assert m.belief_state == {"p2": "likely Werewolf", "p3": "Mason, trusted"}


def test_update_beliefs_rejects_non_dict():
    m = _memory()
    m.update_beliefs("not a dict")  # type: ignore[arg-type]
    assert m.belief_state == {}
    m.update_beliefs(["nope"])  # type: ignore[arg-type]
    assert m.belief_state == {}


def test_update_beliefs_skips_non_string_values_and_empty_strings():
    m = _memory()
    m.update_beliefs({"p1": "ok", "p2": 5, "p3": "  ", "p4": "fine"})
    assert m.belief_state == {"p1": "ok", "p4": "fine"}


def test_update_beliefs_caps_long_entries_at_200_chars():
    m = _memory()
    long = "x" * 500
    m.update_beliefs({"p1": long})
    assert len(m.belief_state["p1"]) == 200


def test_update_beliefs_replaces_full_dict_not_merges():
    m = _memory()
    m.update_beliefs({"p1": "first"})
    m.update_beliefs({"p2": "second"})
    assert m.belief_state == {"p2": "second"}  # p1 dropped


def test_beliefs_section_empty_renders_nothing_yet():
    out = _memory().to_prompt_context("day")
    assert "== YOUR PRIVATE BELIEFS" in out
    assert "Nothing yet." in out


def test_beliefs_section_renders_entries():
    m = _memory()
    m.update_beliefs({"p2": "likely Werewolf", "p3": "Mason"})
    out = m.to_prompt_context("day")
    assert "== YOUR PRIVATE BELIEFS" in out
    assert "- p2: likely Werewolf" in out
    assert "- p3: Mason" in out


def test_day_task_schema_mentions_belief_state():
    out = build_day_speech_task(0, 3, max_chars=600)
    assert "belief_state" in out


def test_day_task_schema_mentions_committed_role():
    out = build_day_speech_task(0, 3, max_chars=600)
    assert "committed_current_role" in out


def test_belief_state_instruction_mentions_reliability():
    # Per-entry provenance/uncertainty is what replaces the separate
    # key_facts slot — make sure the instruction text steers there.
    out = build_day_speech_task(0, 3, max_chars=600)
    assert "swapped" in out and "bluffs" in out


# ----- committed role -----

def test_locked_roles_auto_commit_at_construction():
    tm = _memory(role=Role.TROUBLEMAKER)
    assert tm.committed_role == Role.TROUBLEMAKER
    ins = _memory(role=Role.INSOMNIAC)
    assert ins.committed_role == Role.INSOMNIAC


def test_unlocked_roles_start_uncommitted():
    m = _memory(role=Role.WEREWOLF)
    assert m.committed_role is None


def test_commit_role_sets_for_unlocked_role():
    m = _memory(role=Role.WEREWOLF)
    m.commit_role(Role.ROBBER)
    assert m.committed_role == Role.ROBBER


def test_commit_role_ignored_for_locked_roles():
    tm = _memory(role=Role.TROUBLEMAKER)
    tm.commit_role(Role.WEREWOLF)  # attempt to overwrite
    assert tm.committed_role == Role.TROUBLEMAKER


def test_uncommitted_memory_renders_dealt_role_and_observations():
    m = _memory(role=Role.WEREWOLF)
    m.add_observation("ww_pair", "You see your partner: p3.", {})
    out = m.to_prompt_context("day")
    assert "dealt role at the start of the night was: werewolf" in out
    assert "== WHAT YOU LEARNED DURING THE NIGHT ==" in out
    assert "You see your partner: p3." in out


def test_committed_memory_drops_raw_observations():
    # Once the agent commits, raw night obs are gone from the prompt —
    # the agent is expected to have folded important info into
    # belief_state with provenance.
    m = _memory(role=Role.WEREWOLF)
    m.add_observation("ww_pair", "You see your partner: p3.", {})
    m.commit_role(Role.WEREWOLF)
    out = m.to_prompt_context("day")
    assert "CURRENT (committed) role is: werewolf" in out
    assert "== WHAT YOU LEARNED DURING THE NIGHT ==" not in out
    assert "You see your partner: p3." not in out
    assert "dealt role at the start" not in out


def test_locked_role_memory_renders_committed_view_from_start():
    # TM auto-locks at construction → IDENTITY shows committed role
    # even before any speak() turn, and raw obs aren't shown.
    tm = _memory(role=Role.TROUBLEMAKER)
    out = tm.to_prompt_context("day")
    assert "CURRENT (committed) role is: troublemaker" in out
    assert "== WHAT YOU LEARNED DURING THE NIGHT ==" not in out
