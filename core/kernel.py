from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from threading import RLock
from typing import Any


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    entrypoint: str


@dataclass(frozen=True)
class ModuleStatus:
    name: str
    entrypoint: str
    loaded: bool


class Kernel:
    """
    Central runtime that lazily loads modules by name.
    A module must expose `create_module()` in its entrypoint.
    """

    def __init__(self, specs: list[ModuleSpec]) -> None:
        self._specs = {spec.name: spec for spec in specs}
        self._instances: dict[str, Any] = {}
        self._lock = RLock()

    def list_modules(self) -> list[str]:
        with self._lock:
            return sorted(self._specs.keys())

    def list_module_status(self) -> list[ModuleStatus]:
        with self._lock:
            statuses: list[ModuleStatus] = []
            for name in sorted(self._specs.keys()):
                spec = self._specs[name]
                statuses.append(
                    ModuleStatus(name=name, entrypoint=spec.entrypoint, loaded=name in self._instances)
                )
            return statuses

    def is_loaded(self, module_name: str) -> bool:
        with self._lock:
            return module_name in self._instances

    def load(self, module_name: str) -> Any:
        with self._lock:
            if module_name in self._instances:
                return self._instances[module_name]

            spec = self._specs.get(module_name)
            if spec is None:
                available = ", ".join(self.list_modules()) or "(none)"
                raise ValueError(f"Unknown module '{module_name}'. Available: {available}")

            instance = self._create_instance(spec)
            self._instances[module_name] = instance
            return instance

    def unload(self, module_name: str) -> bool:
        with self._lock:
            instance = self._instances.pop(module_name, None)

        if instance is None:
            return False

        self._call_optional_hook(instance, "on_unload")
        return True

    def reload(self, module_name: str) -> Any:
        self.unload(module_name)
        return self.load(module_name)

    def unload_all(self) -> None:
        with self._lock:
            names = list(self._instances.keys())
        for name in names:
            self.unload(name)

    @staticmethod
    def _call_optional_hook(instance: Any, hook_name: str) -> None:
        hook = getattr(instance, hook_name, None)
        if callable(hook):
            hook()

    def _create_instance(self, spec: ModuleSpec) -> Any:
        mod = import_module(spec.entrypoint)
        create = getattr(mod, "create_module", None)
        if create is None:
            raise RuntimeError(
                f"Module '{spec.name}' entrypoint '{spec.entrypoint}' has no create_module()."
            )

        instance = create()
        # Optional lifecycle hook for modules that need access to kernel/runtime.
        on_load = getattr(instance, "on_load", None)
        if callable(on_load):
            on_load(self)
        return instance
