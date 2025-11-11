from __future__ import annotations

from typing import Callable, Any


class Step:
    """Represents a single node in a Component DAG."""

    def __init__(
        self, name: str, func: Callable[..., Any], depends_on: list[str] | None = None
    ):
        self.name = name
        self.func = func
        self.depends_on = depends_on or []

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Step(name={self.name!r}, depends_on={self.depends_on!r})"
