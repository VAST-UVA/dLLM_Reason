"""Global registry for models, schedulers, search methods, etc."""

from typing import Any


class Registry:
    """Simple string-keyed registry for factory pattern."""

    def __init__(self, name: str):
        self.name = name
        self._entries: dict[str, Any] = {}

    def register(self, key: str):
        """Decorator to register a class/function under `key`."""
        def decorator(obj):
            if key in self._entries:
                raise KeyError(f"{self.name} already has entry '{key}'")
            self._entries[key] = obj
            return obj
        return decorator

    def get(self, key: str) -> Any:
        if key not in self._entries:
            available = ", ".join(sorted(self._entries))
            raise KeyError(f"{self.name} has no entry '{key}'. Available: {available}")
        return self._entries[key]

    def keys(self):
        return self._entries.keys()

    def __contains__(self, key: str):
        return key in self._entries

    def __repr__(self):
        return f"Registry(name={self.name}, entries={list(self._entries.keys())})"


MODEL_REGISTRY = Registry("models")
SCHEDULER_REGISTRY = Registry("schedulers")
SEARCH_REGISTRY = Registry("search")
