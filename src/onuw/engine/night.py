from ..agents.base import Agent
from ..events.bus import (
    EventBus,
    NightActionEvent,
    NightWakeEvent,
    StateMutationEvent,
)
from ..state import GameState
from ..types import WAKE_ORDER, Role


async def run_night(
    state: GameState,
    agents: dict[str, Agent],
    bus: EventBus,
) -> None:
    for role in WAKE_ORDER:
        actor_ids = [
            pid
            for pid in state.seat_order
            if state.players[pid].original_role == role
        ]
        bus.emit(NightWakeEvent(role=role, actors=actor_ids))
        if not actor_ids:
            continue
        await _handle_role(role, actor_ids, state, agents, bus)


async def _handle_role(role, actor_ids, state, agents, bus):
    if role == Role.WEREWOLF:
        await _handle_werewolves(actor_ids, state, agents, bus)
    elif role == Role.MINION:
        _handle_minion(actor_ids, state, agents, bus)
    elif role == Role.MASON:
        _handle_masons(actor_ids, agents, bus)
    elif role == Role.SEER:
        await _handle_seer(actor_ids, state, agents, bus)
    elif role == Role.ROBBER:
        await _handle_robber(actor_ids, state, agents, bus)
    elif role == Role.TROUBLEMAKER:
        await _handle_troublemaker(actor_ids, state, agents, bus)
    elif role == Role.DRUNK:
        await _handle_drunk(actor_ids, state, agents, bus)
    elif role == Role.INSOMNIAC:
        _handle_insomniac(actor_ids, state, agents, bus)


def _swap_player_player(state: GameState, a: str, b: str, bus: EventBus) -> None:
    state.players[a].current_role, state.players[b].current_role = (
        state.players[b].current_role,
        state.players[a].current_role,
    )
    bus.emit(StateMutationEvent(kind="swap_player_player", a=a, b=b))


def _swap_player_center(
    state: GameState, pid: str, index: int, bus: EventBus
) -> None:
    state.players[pid].current_role, state.center[index].role = (
        state.center[index].role,
        state.players[pid].current_role,
    )
    bus.emit(
        StateMutationEvent(
            kind="swap_player_center", a=pid, b=f"center[{index}]"
        )
    )


def _validate_center_index(v) -> int:
    try:
        i = int(v)
    except (TypeError, ValueError):
        return 0
    return i if 0 <= i <= 2 else 0


def _validate_center_pair(indices) -> tuple[int, int]:
    if not isinstance(indices, (list, tuple)) or len(indices) < 2:
        return 0, 1
    i = _validate_center_index(indices[0])
    j = _validate_center_index(indices[1])
    if i == j:
        for alt in range(3):
            if alt != i:
                return i, alt
    return i, j


def _others(seat_order: list[str], self_id: str) -> list[str]:
    return [p for p in seat_order if p != self_id]


async def _handle_werewolves(actor_ids, state, agents, bus):
    for pid in actor_ids:
        others = [p for p in actor_ids if p != pid]
        if others:
            text = f"You see the other Werewolves: {', '.join(others)}."
            structured = {"other_werewolves": others}
            agents[pid].observe_night("werewolf_wake", text, structured)
            bus.emit(
                NightActionEvent(
                    player_id=pid,
                    role=Role.WEREWOLF,
                    action={"step": "werewolf_wake"},
                    observation=structured,
                )
            )
    if len(actor_ids) == 1:
        pid = actor_ids[0]
        action = await agents[pid].act_night("werewolf_solo", valid_targets=[])
        index = _validate_center_index(action.get("index", 0))
        peeked = state.center[index].role
        text = (
            f"You are the only Werewolf. You peeked at center[{index}] "
            f"and saw: {peeked.value}."
        )
        structured = {"index": index, "role": peeked.value}
        agents[pid].observe_night("werewolf_solo_peek", text, structured)
        bus.emit(
            NightActionEvent(
                player_id=pid,
                role=Role.WEREWOLF,
                action={"action": "peek_center", "index": index},
                observation=structured,
            )
        )


def _handle_minion(actor_ids, state, agents, bus):
    werewolf_ids = [
        pid
        for pid in state.seat_order
        if state.players[pid].original_role == Role.WEREWOLF
    ]
    for pid in actor_ids:
        if werewolf_ids:
            text = f"You see the Werewolves: {', '.join(werewolf_ids)}."
        else:
            text = (
                "There are no Werewolves in play (both Werewolf cards are in "
                "the center)."
            )
        structured = {"werewolves": werewolf_ids}
        agents[pid].observe_night("minion_wake", text, structured)
        bus.emit(
            NightActionEvent(
                player_id=pid,
                role=Role.MINION,
                action={"step": "minion_wake"},
                observation=structured,
            )
        )


def _handle_masons(actor_ids, agents, bus):
    for pid in actor_ids:
        others = [p for p in actor_ids if p != pid]
        if others:
            text = f"You see the other Mason(s): {', '.join(others)}."
        else:
            text = "You are the only Mason."
        structured = {"other_masons": others}
        agents[pid].observe_night("mason_wake", text, structured)
        bus.emit(
            NightActionEvent(
                player_id=pid,
                role=Role.MASON,
                action={"step": "mason_wake"},
                observation=structured,
            )
        )


async def _handle_seer(actor_ids, state, agents, bus):
    for pid in actor_ids:
        valid = _others(state.seat_order, pid)
        action = await agents[pid].act_night("seer", valid_targets=valid)
        if action.get("action") == "view_center":
            i, j = _validate_center_pair(action.get("indices", [0, 1]))
            ri = state.center[i].role.value
            rj = state.center[j].role.value
            text = (
                f"You viewed two center cards: center[{i}]={ri}, "
                f"center[{j}]={rj}."
            )
            structured = {
                "mode": "center",
                "indices": [i, j],
                "roles": {f"center[{i}]": ri, f"center[{j}]": rj},
            }
        else:
            target = action.get("target")
            if target not in state.players or target == pid:
                target = valid[0]
            seen = state.players[target].current_role.value
            text = f"You looked at {target}'s card; it is {seen}."
            structured = {"mode": "player", "target": target, "role": seen}
        agents[pid].observe_night("seer_action", text, structured)
        bus.emit(
            NightActionEvent(
                player_id=pid,
                role=Role.SEER,
                action=action,
                observation=structured,
            )
        )


async def _handle_robber(actor_ids, state, agents, bus):
    for pid in actor_ids:
        valid = _others(state.seat_order, pid)
        action = await agents[pid].act_night("robber", valid_targets=valid)
        target = action.get("target")
        if target not in state.players or target == pid:
            target = valid[0]
        _swap_player_player(state, pid, target, bus)
        new_role = state.players[pid].current_role.value
        text = (
            f"You swapped your card with {target}'s. Your CURRENT role is "
            f"now {new_role}. Note: this does NOT grant you the {new_role}'s "
            "night action — you have already acted as Robber."
        )
        structured = {"target": target, "new_role": new_role}
        agents[pid].observe_night("robber_action", text, structured)
        bus.emit(
            NightActionEvent(
                player_id=pid,
                role=Role.ROBBER,
                action=action,
                observation=structured,
            )
        )


async def _handle_troublemaker(actor_ids, state, agents, bus):
    for pid in actor_ids:
        valid = _others(state.seat_order, pid)
        action = await agents[pid].act_night("troublemaker", valid_targets=valid)
        a = action.get("target_a")
        b = action.get("target_b")
        if a not in valid or b not in valid or a == b:
            a, b = valid[0], valid[1]
        _swap_player_player(state, a, b, bus)
        text = "You swapped two other players' cards without looking."
        structured = {"target_a": a, "target_b": b}
        agents[pid].observe_night("troublemaker_action", text, structured)
        bus.emit(
            NightActionEvent(
                player_id=pid,
                role=Role.TROUBLEMAKER,
                action=action,
                observation=structured,
            )
        )


async def _handle_drunk(actor_ids, state, agents, bus):
    for pid in actor_ids:
        action = await agents[pid].act_night("drunk", valid_targets=[])
        index = _validate_center_index(action.get("index", 0))
        _swap_player_center(state, pid, index, bus)
        text = (
            f"You swapped your card with center[{index}] without looking. "
            "You do NOT know your new role."
        )
        structured = {"index": index}
        agents[pid].observe_night("drunk_action", text, structured)
        bus.emit(
            NightActionEvent(
                player_id=pid,
                role=Role.DRUNK,
                action=action,
                observation=structured,
            )
        )


def _handle_insomniac(actor_ids, state, agents, bus):
    for pid in actor_ids:
        current = state.players[pid].current_role.value
        text = f"You looked at your card. Your CURRENT role is {current}."
        structured = {"current_role": current}
        agents[pid].observe_night("insomniac_wake", text, structured)
        bus.emit(
            NightActionEvent(
                player_id=pid,
                role=Role.INSOMNIAC,
                action={"step": "insomniac_wake"},
                observation=structured,
            )
        )
