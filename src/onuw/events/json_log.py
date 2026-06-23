import json
import uuid
from dataclasses import fields
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .bus import ContentChunkEvent, Event, GameEndEvent, ReasoningChunkEvent
from .observer import Observer


def _to_jsonable(v: Any) -> Any:
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _to_jsonable(x) for k, x in v.items()}
    return v


def _serialize(event: Event) -> dict:
    record: dict = {
        "type": type(event).__name__,
        "visibility": event.visibility,
    }
    for f in fields(event):
        record[f.name] = _to_jsonable(getattr(event, f.name))
    return record


class JsonObserver(Observer):
    def __init__(self, log_dir: Path | str) -> None:
        self.log_dir = Path(log_dir)
        self.events: list[dict] = []
        self.game_id: str | None = None
        self.output_path: Path | None = None

    def on_event(self, event: Event) -> None:
        if self.game_id is None:
            self.game_id = getattr(event, "game_id", None) or uuid.uuid4().hex[:8]
        # Skip streaming chunks — the per-call aggregate arrives as a
        # single LLMCallEvent from LLMAgent and that's what we log.
        if isinstance(event, (ReasoningChunkEvent, ContentChunkEvent)):
            return
        record = _serialize(event)
        record["ts"] = datetime.now(timezone.utc).isoformat()
        self.events.append(record)
        if isinstance(event, GameEndEvent):
            self._flush()

    def _flush(self) -> None:
        assert self.game_id is not None
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.log_dir / f"{self.game_id}.json"
        payload = {"game_id": self.game_id, "events": self.events}
        self.output_path.write_text(json.dumps(payload, indent=2))
