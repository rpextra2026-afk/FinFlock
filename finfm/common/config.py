"""Configuration loading. Every constant of the experiment comes through here.

Each ``config/<name>.yaml`` becomes the top-level section ``<name>``. ``measured.yaml`` is applied
last and may only override keys that already exist (so a typo cannot silently add a new key).
The config hash covers every yaml value plus the bytes of every non-yaml file in the directory
(csv tables) and is written into every output.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from finfm.common.hashing import sha256_file, stable_hash

MEASURED_STEM = "measured"
YAML_SUFFIXES = (".yaml", ".yml")
HASHED_DATA_SUFFIXES = (".csv", ".txt")
PATH_SEPARATOR = "."
UNHASHED_SECTIONS = ("cpu", "ablation", "clean", "build")  # CPU-stage settings: readable like any section, never part of the hash the GPU notebooks and saved dev fits carry


class ConfigError(KeyError):
    """Raised for missing or malformed configuration."""


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "<MISSING>"


_MISSING = _Missing()


def _walk(data: Mapping[str, Any], path: str) -> Any:
    node: Any = data
    for part in path.split(PATH_SEPARATOR):
        if not isinstance(node, Mapping) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _set(data: dict[str, Any], path: str, value: Any, *, must_exist: bool) -> None:
    parts = path.split(PATH_SEPARATOR)
    node = data
    for part in parts[:-1]:
        nxt = node.get(part, _MISSING)
        if nxt is _MISSING:
            if must_exist:
                raise ConfigError(f"unknown config path: {path}")
            nxt = node[part] = {}
        if not isinstance(nxt, dict):
            raise ConfigError(f"config path {path} passes through a non-mapping value")
        node = nxt
    if must_exist and parts[-1] not in node:
        raise ConfigError(f"unknown config path: {path}")
    node[parts[-1]] = value


class Config:
    """Immutable-style nested configuration with dotted-path access and a stable hash."""

    def __init__(self, data: Mapping[str, Any], file_digests: Mapping[str, str] | None = None):
        self._data: dict[str, Any] = copy.deepcopy(dict(data))
        self._file_digests: dict[str, str] = dict(file_digests or {})

    @classmethod
    def load(cls, config_dir: str | Path, overrides: Mapping[str, Any] | None = None) -> Config:
        """Load every yaml in ``config_dir``; apply ``measured.yaml`` then ``overrides``."""
        directory = Path(config_dir)
        if not directory.is_dir():
            raise ConfigError(f"config directory not found: {directory}")
        data: dict[str, Any] = {}
        measured: Mapping[str, Any] = {}
        digests: dict[str, str] = {}
        for path in sorted(directory.iterdir()):
            if path.suffix in YAML_SUFFIXES:
                content = yaml.safe_load(path.read_text(encoding="utf-8"))
                if path.stem == MEASURED_STEM:
                    measured = content or {}
                    continue
                if not isinstance(content, Mapping):
                    raise ConfigError(f"{path.name} must contain a mapping at the top level")
                data[path.stem] = content
            elif path.suffix in HASHED_DATA_SUFFIXES:
                digests[path.name] = sha256_file(path)
        config = cls(data, digests)
        if not isinstance(measured, Mapping):
            raise ConfigError(f"{MEASURED_STEM}.yaml must contain a mapping")
        config = config.with_overrides(_flatten(measured))
        if overrides:
            config = config.with_overrides(overrides)
        return config

    def get(self, path: str, default: Any = _MISSING) -> Any:
        """Value at a dotted path; raises ``ConfigError`` when absent and no default is given."""
        value = _walk(self._data, path)
        if value is _MISSING:
            if default is _MISSING:
                raise ConfigError(f"missing config key: {path}")
            return default
        return copy.deepcopy(value)

    def section(self, name: str) -> Mapping[str, Any]:
        """A top-level section as a plain mapping."""
        value = self.get(name)
        if not isinstance(value, Mapping):
            raise ConfigError(f"config section {name} is not a mapping")
        return value

    def with_overrides(self, overrides: Mapping[str, Any]) -> Config:
        """A copy with dotted-path overrides applied; every path must already exist."""
        data = copy.deepcopy(self._data)
        for path, value in overrides.items():
            _set(data, path, copy.deepcopy(value), must_exist=True)
        return Config(data, self._file_digests)

    def to_dict(self) -> dict[str, Any]:
        """A deep copy of the full configuration."""
        return copy.deepcopy(self._data)

    def hash_without(self, sections: tuple[str, ...]) -> str:
        """Hash that ignores whole top-level sections (infrastructure settings that cannot change results)."""
        data = {k: v for k, v in self._data.items() if k not in sections and k not in UNHASHED_SECTIONS}
        return stable_hash({"values": data, "files": self._file_digests})

    @property
    def hash(self) -> str:
        """Stable sha256 of all hashed values and of the non-yaml config files (``UNHASHED_SECTIONS`` excluded)."""
        return self.hash_without(())

    @property
    def unhashed_hash(self) -> str:
        """Hash of the sections the main hash leaves out, so CPU-stage settings are still on record."""
        return stable_hash({k: v for k, v in self._data.items() if k in UNHASHED_SECTIONS})


def _flatten(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested mappings into dotted paths so measured.yaml may be nested or flat."""
    flat: dict[str, Any] = {}
    for key, value in mapping.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            flat.update(_flatten(value, f"{path}{PATH_SEPARATOR}"))
        else:
            flat[path] = value
    return flat


def load_config(config_dir: str | Path, overrides: Mapping[str, Any] | None = None) -> Config:
    """Convenience wrapper for ``Config.load``."""
    return Config.load(config_dir, overrides)
