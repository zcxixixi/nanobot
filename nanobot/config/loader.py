"""Configuration loading utilities."""

import json
import os
from pathlib import Path
from typing import Any, get_args, get_origin

from nanobot.config.schema import Config


def get_config_path() -> Path:
    """Get the default configuration file path."""
    return Path.home() / ".nanobot" / "config.json"


def get_data_dir() -> Path:
    """Get the nanobot data directory."""
    from nanobot.utils.helpers import get_data_path
    return get_data_path()


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from file or create default.
    
    Args:
        config_path: Optional path to config file. Uses default if not provided.
    
    Returns:
        Loaded configuration object.
    """
    path = config_path or get_config_path()
    
    data: dict[str, Any] = {}

    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            data = _migrate_config(data)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to load config from {path}: {e}")
            print("Using default configuration.")

    normalized = convert_keys(data)
    env_overrides = _collect_env_overrides()
    if env_overrides:
        normalized = _deep_merge(normalized, env_overrides)
    return Config.model_validate(normalized)


def save_config(config: Config, config_path: Path | None = None) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration to save.
        config_path: Optional path to save to. Uses default if not provided.
    """
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to camelCase format
    data = config.model_dump()
    data = convert_to_camel(data)
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _migrate_config(data: dict) -> dict:
    """Migrate old config formats to current."""
    # Move tools.exec.restrictToWorkspace → tools.restrictToWorkspace
    tools = data.get("tools", {})
    exec_cfg = tools.get("exec", {})
    if "restrictToWorkspace" in exec_cfg and "restrictToWorkspace" not in tools:
        tools["restrictToWorkspace"] = exec_cfg.pop("restrictToWorkspace")
    return data


def convert_keys(data: Any) -> Any:
    """Convert camelCase keys to snake_case for Pydantic."""
    if isinstance(data, dict):
        return {camel_to_snake(k): convert_keys(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_keys(item) for item in data]
    return data


def convert_to_camel(data: Any) -> Any:
    """Convert snake_case keys to camelCase."""
    if isinstance(data, dict):
        return {snake_to_camel(k): convert_to_camel(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_to_camel(item) for item in data]
    return data


def _collect_env_overrides() -> dict[str, Any]:
    """Collect NANOBOT_* env vars into nested config overrides."""
    prefix = "NANOBOT_"
    overrides: dict[str, Any] = {}
    for key, raw in os.environ.items():
        if not key.startswith(prefix):
            continue
        tail = key[len(prefix):]
        if not tail:
            continue
        path = [segment.lower() for segment in tail.split("__") if segment]
        if not path:
            continue
        if not _is_valid_config_path(path):
            continue
        _set_nested(overrides, path, _parse_env_value(raw))
    return overrides


def _set_nested(target: dict[str, Any], path: list[str], value: Any) -> None:
    node = target
    for part in path[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[path[-1]] = value


def _parse_env_value(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    low = text.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return raw
    return raw


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _is_valid_config_path(path: list[str]) -> bool:
    """Check whether an env override path exists in Config schema."""
    model: Any = Config
    for idx, part in enumerate(path):
        fields = getattr(model, "model_fields", {})
        if part not in fields:
            return False
        if idx == len(path) - 1:
            return True
        annotation = fields[part].annotation
        next_model = _resolve_model_type(annotation)
        if next_model is None:
            return False
        model = next_model
    return True


def _resolve_model_type(annotation: Any) -> type | None:
    """Resolve nested Pydantic model type from annotation."""
    if isinstance(annotation, type) and hasattr(annotation, "model_fields"):
        return annotation

    origin = get_origin(annotation)
    if origin is None:
        return None

    for arg in get_args(annotation):
        resolved = _resolve_model_type(arg)
        if resolved is not None:
            return resolved
    return None


def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    result = []
    for i, char in enumerate(name):
        if char.isupper() and i > 0:
            result.append("_")
        result.append(char.lower())
    return "".join(result)


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])
