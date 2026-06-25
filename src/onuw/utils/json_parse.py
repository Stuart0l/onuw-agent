import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Tolerant JSON extraction. Tries: (1) raw json.loads, (2) strip
    a surrounding ```/```json fence, (3) first balanced {...}/[...]
    block (ignoring string-literal contents). Raises
    json.JSONDecodeError on total failure."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    chunk = _first_balanced(text)
    if chunk is not None:
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("no valid JSON object found in response", text or "", 0)


def _first_balanced(text: str) -> str | None:
    """Return the first balanced {...} or [...] substring, ignoring
    braces that appear inside JSON string literals."""
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None
