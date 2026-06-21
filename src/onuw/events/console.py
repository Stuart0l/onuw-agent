from rich.console import Console

from .bus import (
    CenterDealtEvent,
    DeathsEvent,
    Event,
    GameEndEvent,
    GameStartEvent,
    NightActionEvent,
    NightWakeEvent,
    RoleAssignedEvent,
    SpeechEvent,
    StateMutationEvent,
    VotesRevealedEvent,
)
from .observer import Observer


class ConsoleObserver(Observer):
    def __init__(self, god: bool = False, console: Console | None = None) -> None:
        self.god = god
        self.console = console or Console()

    def on_event(self, event: Event) -> None:
        if event.visibility != "public" and not self.god:
            self._render_redacted(event)
            return
        self._render(event)

    def _render_redacted(self, event: Event) -> None:
        name = type(event).__name__.removesuffix("Event")
        self.console.print(f"[dim]({name} — hidden)[/dim]")

    def _render(self, event: Event) -> None:
        if isinstance(event, GameStartEvent):
            self.console.rule(f"[bold cyan]Game {event.game_id}[/bold cyan]")
            names = ", ".join(p.get("name", p.get("id", "?")) for p in event.players)
            self.console.print(f"Players: {names}")
            self.console.print(f"Role pool: {', '.join(r.value for r in event.role_pool)}")
            self.console.print(f"Discussion rounds: {event.discussion_rounds}")
        elif isinstance(event, RoleAssignedEvent):
            self.console.print(
                f"  [yellow]{event.player_id}[/yellow] dealt "
                f"[magenta]{event.role.value}[/magenta]"
            )
        elif isinstance(event, CenterDealtEvent):
            cards = ", ".join(r.value for r in event.cards)
            self.console.print(f"  Center: [magenta]{cards}[/magenta]")
        elif isinstance(event, NightWakeEvent):
            actors = ", ".join(event.actors) if event.actors else "no one"
            self.console.print(f"[blue]Night:[/blue] {event.role.value} wakes ({actors})")
        elif isinstance(event, NightActionEvent):
            self.console.print(
                f"  [yellow]{event.player_id}[/yellow] ({event.role.value}) "
                f"action={event.action}"
            )
            if event.observation:
                self.console.print(f"    learned: {event.observation}")
        elif isinstance(event, StateMutationEvent):
            self.console.print(
                f"  [dim]{event.kind}: {event.a} <-> {event.b}[/dim]"
            )
        elif isinstance(event, SpeechEvent):
            self.console.print(
                f"[green]R{event.round_idx + 1} {event.speaker_id}:[/green] "
                f"{event.text}"
            )
        elif isinstance(event, VotesRevealedEvent):
            self.console.print("[bold]Votes:[/bold]")
            for voter, target in event.votes.items():
                self.console.print(f"  {voter} -> {target}")
        elif isinstance(event, DeathsEvent):
            if event.deaths:
                self.console.print(f"[red]Deaths:[/red] {', '.join(event.deaths)}")
            else:
                self.console.print("[red]Deaths:[/red] none")
            for hunter, target in event.hunter_revenge:
                self.console.print(f"  [red]Hunter {hunter} drags {target}[/red]")
        elif isinstance(event, GameEndEvent):
            self.console.rule("[bold green]Game Over[/bold green]")
            self.console.print(f"Winners: {[w.value for w in event.winners]}")
            self.console.print(f"Final state: {event.final_state}")
        else:
            self.console.print(f"{type(event).__name__}: {event}")
