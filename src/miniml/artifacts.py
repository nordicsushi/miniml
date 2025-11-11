from __future__ import annotations

from typing import Any


class ArtifactStore:
    """Abstract interface for storing step outputs."""

    def get(self, component: str, step_name: str) -> Any | None:
        raise NotImplementedError

    def set(self, component: str, step_name: str, value: Any) -> None:
        raise NotImplementedError

    def delete(self, component: str, step_name: str) -> None:
        raise NotImplementedError


class InMemoryArtifactStore(ArtifactStore):
    """Simple in-memory implementation. Good for local/dev."""

    def __init__(self) -> None:
        # key: (component_name, step_name)
        self._store: dict[tuple[str, str], Any] = {}

    def get(self, component: str, step_name: str) -> Any | None:
        return self._store.get((component, step_name))

    def set(self, component: str, step_name: str, value: Any) -> None:
        self._store[(component, step_name)] = value

    def delete(self, component: str, step_name: str) -> None:
        self._store.pop((component, step_name), None)

