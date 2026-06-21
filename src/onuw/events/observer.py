from abc import ABC, abstractmethod

from .bus import Event


class Observer(ABC):
    @abstractmethod
    def on_event(self, event: Event) -> None: ...
