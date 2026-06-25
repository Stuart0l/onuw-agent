import uuid
from datetime import datetime


def default_game_id() -> str:
    """Generate a sortable, readable game id of the form
    ``game_YYYYMMDD_HHMMSS_<4hex>``. The hex suffix protects against
    collisions for games started in the same second."""
    return datetime.now().strftime("game_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:4]
