from __future__ import annotations

import importlib.util
import inspect
from importlib.metadata import entry_points
from pathlib import Path

from .types import Indicator, Strategy

type PluginClass = type[Strategy] | type[Indicator]
type PluginMap = dict[str, PluginClass]


def _discover(group: str) -> PluginMap:
    discovered: PluginMap = {}
    for ep in entry_points(group=group):
        loaded = ep.load()
        if inspect.isclass(loaded):
            discovered[ep.name] = loaded
    return discovered


def discover_strategies() -> PluginMap:
    return _discover("polymarket_algo.strategies")


def discover_indicators() -> PluginMap:
    return _discover("polymarket_algo.indicators")


def load_local_plugins() -> PluginMap:
    plugin_dir = Path.home() / ".polymarket-algo" / "plugins"
    loaded: PluginMap = {}
    if not plugin_dir.exists():
        return loaded

    for file in plugin_dir.glob("*.py"):
        spec = importlib.util.spec_from_file_location(file.stem, file)
        if not (spec and spec.loader):
            continue

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            if callable(getattr(obj, "evaluate", None)) or callable(getattr(obj, "compute", None)):
                loaded[name] = obj

    return loaded


class PluginRegistry:
    def __init__(self) -> None:
        self.strategies: PluginMap = {}
        self.indicators: PluginMap = {}

    def load(self) -> PluginRegistry:
        self.strategies = discover_strategies()
        self.indicators = discover_indicators()

        for name, plugin in load_local_plugins().items():
            if callable(getattr(plugin, "evaluate", None)):
                self.strategies[name] = plugin
            elif callable(getattr(plugin, "compute", None)):
                self.indicators[name] = plugin

        return self
