from enum import StrEnum


class Role(StrEnum):
    WEREWOLF = "werewolf"
    MINION = "minion"
    MASON = "mason"
    SEER = "seer"
    ROBBER = "robber"
    TROUBLEMAKER = "troublemaker"
    DRUNK = "drunk"
    INSOMNIAC = "insomniac"
    VILLAGER = "villager"
    TANNER = "tanner"
    HUNTER = "hunter"


class Team(StrEnum):
    WEREWOLF = "werewolf"
    VILLAGE = "village"
    TANNER = "tanner"


WAKE_ORDER: list[Role] = [
    Role.WEREWOLF,
    Role.MINION,
    Role.MASON,
    Role.SEER,
    Role.ROBBER,
    Role.TROUBLEMAKER,
    Role.DRUNK,
    Role.INSOMNIAC,
]
