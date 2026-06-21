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
    Role.WEREWOLF: """== YOUR ROLE: WEREWOLF ==
At night you wake first and see any other Werewolves. If you are the ONLY Werewolf in play, you may peek at one of the three center cards. You are on the Werewolf team; your team wins as long as no Werewolf is killed. You should generally hide that you are a Werewolf during the day.""",

    Role.MINION: """== YOUR ROLE: MINION ==
At night, after the Werewolves go back to sleep, you wake and see who the Werewolves are. The Werewolves do NOT know who you are. You are on the Werewolf team. Your team wins as long as no Werewolf is killed — even if YOU are killed. If all Werewolves are in the center, you win as long as no non-Minion player dies.""",

    Role.MASON: """== YOUR ROLE: MASON ==
At night you wake and see your fellow Mason (or learn you are alone). You are on the village team and have no other special action. Knowing a fellow Mason is a strong source of mutual trust during the day.""",

    Role.SEER: """== YOUR ROLE: SEER ==
At night you may EITHER look at one other player's card OR look at two of the three center cards. You learn the role(s) you saw. You are on the village team.""",

    Role.ROBBER: """== YOUR ROLE: ROBBER ==
At night you may swap your card with one other player's card, then look at the card you stole. Your CURRENT role becomes the role you stole; you win with whichever team that new role belongs to. NOTE: you do NOT gain the new role's night action — you have already acted as the Robber.""",

    Role.TROUBLEMAKER: """== YOUR ROLE: TROUBLEMAKER ==
At night you may swap the cards of two OTHER players (not your own). You do NOT look at the cards you swap. Your own role does not change. You are on the village team.""",

    Role.DRUNK: """== YOUR ROLE: DRUNK ==
At night you exchange your card with one of the three center cards WITHOUT looking. You do NOT learn your new role. Your CURRENT role becomes whatever center card you took; you win with that team even though you don't know what it is.""",

    Role.INSOMNIAC: """== YOUR ROLE: INSOMNIAC ==
At night you wake LAST and look at your OWN card to see your CURRENT role — which may differ from the role you were dealt if someone swapped your card during the night. You are on the village team.""",

    Role.VILLAGER: """== YOUR ROLE: VILLAGER ==
You have no night action. You are on the village team. Your goal is to identify and kill at least one Werewolf through discussion and voting.""",

    Role.TANNER: """== YOUR ROLE: TANNER ==
You have no night action. You are a one-player team. You WIN if and only if you are killed during the vote. You should act suspicious enough to draw votes — but not so obviously that no one believes you.""",

    Role.HUNTER: """== YOUR ROLE: HUNTER ==
You have no night action. You are on the village team. If you are killed during the vote, the player you voted for ALSO dies. Telegraphing this can deter opponents from voting for you.""",
}