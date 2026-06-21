import pytest
from pydantic import ValidationError

from onuw.config import GameConfig, PlayerConfig
from onuw.types import Role, Team, WAKE_ORDER


def _player(i: int) -> PlayerConfig:
    return PlayerConfig(id=f"p{i}", name=f"Player{i}", model="gpt-4o")


def test_role_enum_has_eleven_base_roles():
    assert len(list(Role)) == 11


def test_team_enum():
    assert {t.value for t in Team} == {"werewolf", "village", "tanner"}


def test_wake_order_excludes_no_action_roles():
    assert Role.VILLAGER not in WAKE_ORDER
    assert Role.TANNER not in WAKE_ORDER
    assert Role.HUNTER not in WAKE_ORDER
    assert WAKE_ORDER[0] == Role.WEREWOLF
    assert WAKE_ORDER[-1] == Role.INSOMNIAC


def test_valid_game_config():
    cfg = GameConfig(
        players=[_player(i) for i in range(1, 4)],
        role_pool=[
            Role.WEREWOLF, Role.VILLAGER, Role.SEER,
            Role.ROBBER, Role.VILLAGER, Role.TANNER,
        ],
    )
    assert len(cfg.role_pool) == len(cfg.players) + 3


def test_role_pool_size_rejected_when_too_small():
    with pytest.raises(ValidationError):
        GameConfig(
            players=[_player(i) for i in range(1, 4)],
            role_pool=[Role.WEREWOLF, Role.VILLAGER, Role.SEER],
        )


def test_role_pool_size_rejected_when_too_large():
    with pytest.raises(ValidationError):
        GameConfig(
            players=[_player(i) for i in range(1, 4)],
            role_pool=[Role.WEREWOLF] * 10,
        )


def test_duplicate_player_ids_rejected():
    with pytest.raises(ValidationError):
        GameConfig(
            players=[_player(1), _player(1), _player(2)],
            role_pool=[
                Role.WEREWOLF, Role.VILLAGER, Role.SEER,
                Role.ROBBER, Role.VILLAGER, Role.TANNER,
            ],
        )


def test_player_count_bounds():
    with pytest.raises(ValidationError):
        GameConfig(
            players=[_player(1), _player(2)],
            role_pool=[Role.WEREWOLF] * 5,
        )
