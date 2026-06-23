from ..types import Role


SYSTEM_PREAMBLE = (
    "You are an autonomous agent playing the social-deduction game "
    "One Night Ultimate Werewolf (ONUW). You will be told your secret role and "
    "any information your role learned during the night. You then participate "
    "in a public discussion and vote on whom to kill. Your goal is to make "
    "your team win."
)


GAME_RULES_BLOCK = """== GAME RULES ==
- One Night Ultimate Werewolf is played in a single night followed by a single day.
- Each player is secretly dealt one role card. Three extra cards are placed face-down in the center, unassigned to any player.
- NIGHT: roles wake (eyes-open) in a fixed order and may perform a role-specific action. Some roles learn information; some roles swap cards between players or with the center, possibly changing who is on which team.
- DAY: players openly discuss what happened. Anyone may lie, claim a role, accuse, or stay silent.
- VOTE: each player simultaneously votes for one player to be killed.
- The player with the most votes dies. If two or more players tie for the most votes, they ALL die. If every player receives exactly one vote, NOBODY dies.
- IMPORTANT: night actions happen in a fixed wake order based on the role you were ORIGINALLY DEALT. If your card was swapped during the night, you do NOT get a second turn as your new role.
- Win conditions are evaluated on the role card you are HOLDING at the end of the night (your CURRENT role), which may differ from the role you were dealt."""


WIN_CONDITIONS_BLOCK = """== WIN CONDITIONS ==
- Village team: wins if at least one player whose CURRENT role is Werewolf is killed. Special case: if no Werewolves exist in play at all (both Werewolves are in the center), the village wins only if NOBODY dies.
- Werewolf team (Werewolves + Minion): wins if at least one Werewolf exists in play AND no Werewolf is killed. The Minion is on the Werewolf team but is NOT a Werewolf for death-counting purposes — the Minion may die without the Werewolf team losing.
- Minion edge case: if all Werewolves are in the center, the Minion wins as long as no non-Minion player dies.
- Tanner: wins if and only if the Tanner is killed. Tanner is a one-player team and can co-win with the Village.
- A player's team is determined by their CURRENT role at end of night. A player whose original card was Minion is always on the Werewolf team."""


OUTPUT_FORMAT_PREAMBLE = (
    "When the game prompts you for an action, a public statement, or a vote, "
    "you MUST respond with valid JSON ONLY — no surrounding prose, no markdown "
    "code fences. The exact JSON schema for each turn will be specified in the "
    "user message."
)


ROLE_ABILITY_BLOCKS: dict[Role, str] = {
    Role.WEREWOLF: """== WEREWOLF (Werewolf team) ==
At night they wake first and see any other Werewolves. If they are the ONLY Werewolf in play, they may peek at one of the three center cards. Their team wins as long as no Werewolf is killed. They should generally hide that they are a Werewolf during the day.""",

    Role.MINION: """== MINION (Werewolf team) ==
At night, after the Werewolves go back to sleep, they wake and see who the Werewolves are. The Werewolves do NOT know who they are. Their team wins as long as no Werewolf is killed — even if THEY are killed. If all Werewolves are in the center, they win as long as no non-Minion player dies.""",

    Role.MASON: """== MASON (Village team) ==
At night they wake and see their fellow Mason (or learn they are alone). They have no other special action. Knowing a fellow Mason is a strong source of mutual trust during the day.""",

    Role.SEER: """== SEER (Village team) ==
At night they may EITHER look at one other player's card OR look at two of the three center cards. They learn the role(s) they saw.""",

    Role.ROBBER: """== ROBBER (Village team initially; switches to whichever team the stolen card belongs to) ==
At night they may swap their card with one other player's card, then look at the card they stole. Their CURRENT role becomes the role they stole; they win with whichever team that new role belongs to. NOTE: they do NOT gain the new role's night action — they have already acted as the Robber.""",

    Role.TROUBLEMAKER: """== TROUBLEMAKER (Village team) ==
At night they may swap the cards of two OTHER players (not their own). They do NOT look at the cards they swap. Their own role does not change.""",

    Role.DRUNK: """== DRUNK (Village team initially; switches to whichever team the unknown center card belongs to) ==
At night they exchange their card with one of the three center cards WITHOUT looking. They do NOT learn their new role. Their CURRENT role becomes whatever center card they took; they win with that team even though they don't know what it is.""",

    Role.INSOMNIAC: """== INSOMNIAC (Village team) ==
At night they wake LAST and look at their OWN card to see their CURRENT role — which may differ from the role they were dealt if someone swapped their card during the night.""",

    Role.VILLAGER: """== VILLAGER (Village team) ==
They have no night action. Their goal is to identify and kill at least one Werewolf through discussion and voting.""",

    Role.TANNER: """== TANNER (solo — only the Tanner) ==
They have no night action. They WIN if and only if they are killed during the vote. They should act suspicious enough to draw votes — but not so obviously that no one believes them.""",

    Role.HUNTER: """== HUNTER (Village team) ==
They have no night action. If they are killed during the vote, the player they voted for ALSO dies. Telegraphing this can deter opponents from voting for them.""",
}


def swap_reminder(dealt_role: Role) -> str:
    """Per-role decision rule that replaces the previous open-ended swap
    reminder.

    Two "certain" roles get a flat assertion (no need to reason about
    current role): Troublemaker and Insomniac. Everyone else must reason
    about their current role first, then commit once per turn — this
    prevents the observed failure mode where seats spent thousands of
    tokens flip-flopping between "I am still my dealt role" and "I was
    swapped."
    """
    if dealt_role == Role.TROUBLEMAKER:
        return (
            "== YOUR CURRENT ROLE ==\n"
            "You ARE still the Troublemaker. Your card is untouched — "
            "Troublemakers only swap OTHER players, never themselves. "
            "And Robbers will not swap with you. Stick with VILLAGE "
            "team's win condition."
        )
    if dealt_role == Role.INSOMNIAC:
        return (
            "== YOUR CURRENT ROLE ==\n"
            "Your Insomniac wake observation IS your current role — trust "
            "it. You wake up last so no one can swap with you. Stick with "
            "VILLAGE team's win condition."
        )
    if dealt_role == Role.ROBBER:
        return (
            "== YOUR CURRENT ROLE ==\n"
            "Your night observation already shows you the card you stole "
            "— treat that as your CURRENT role and use its team's win "
            "condition. The only thing that could change this is the "
            "Troublemaker (wakes after you); only revise this if a "
            "Troublemaker credibly claims to have swapped your card. "
            "Otherwise act as the role you saw."
        )
    # All other roles: reason, then commit.
    return (
        "== YOUR CURRENT ROLE — REASON, THEN COMMIT ==\n"
        f"You were dealt {dealt_role.value}. Your card may have been "
        "swapped during the night by a Robber or Troublemaker, so your "
        "CURRENT role may differ from your dealt role.\n"
        "Step 1: Weigh the evidence (your night observation + any claims "
        "so far). Pick the SINGLE most likely current-role for yourself.\n"
        "Step 2: Commit to that interpretation for the rest of this turn. "
        "Use the corresponding team's win condition.\n"
        "Step 3: Choose the highest-EV action under your committed "
        "interpretation.\n"
        "DO NOT revisit your interpretation after committing — looping "
        "wastes reasoning and produces worse decisions."
    )