from __future__ import annotations

import json
import logging
from typing import Any

_BANNED_KEYS = {"vector", "vectors", "image_bytes", "raw_image", "embedding_vector"}


def log_search_event(logger: logging.Logger, **fields: Any) -> None:
    safe: dict[str, Any] = {}
    for key, value in fields.items():
        lowered = key.lower()
        if lowered in _BANNED_KEYS or "vector" in lowered or "image_bytes" in lowered:
            continue
        if value is None or isinstance(value, (str, int, float, bool)):
            safe[key] = value
        elif isinstance(value, (list, tuple)) and all(item is None or isinstance(item, (str, int, float, bool)) for item in value):
            safe[key] = list(value)
        else:
            safe[key] = str(value)
    logger.info(json.dumps(safe, ensure_ascii=False, sort_keys=True))
