import json
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.config.loader import load_config


class SampleTool(Tool):
    @property
    def name(self) -> str:
        return "sample"

    @property
    def description(self) -> str:
        return "sample tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2},
                "count": {"type": "integer", "minimum": 1, "maximum": 10},
                "mode": {"type": "string", "enum": ["fast", "full"]},
                "meta": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "flags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["tag"],
                },
            },
            "required": ["query", "count"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


def test_validate_params_missing_required() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi"})
    assert "missing required count" in "; ".join(errors)


def test_validate_params_type_and_range() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 0})
    assert any("count must be >= 1" in e for e in errors)

    errors = tool.validate_params({"query": "hi", "count": "2"})
    assert any("count should be integer" in e for e in errors)


def test_validate_params_enum_and_min_length() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "h", "count": 2, "mode": "slow"})
    assert any("query must be at least 2 chars" in e for e in errors)
    assert any("mode must be one of" in e for e in errors)


def test_validate_params_nested_object_and_array() -> None:
    tool = SampleTool()
    errors = tool.validate_params(
        {
            "query": "hi",
            "count": 2,
            "meta": {"flags": [1, "ok"]},
        }
    )
    assert any("missing required meta.tag" in e for e in errors)
    assert any("meta.flags[0] should be string" in e for e in errors)


def test_validate_params_ignores_unknown_fields() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 2, "extra": "x"})
    assert errors == []


async def test_registry_returns_validation_error() -> None:
    reg = ToolRegistry()
    reg.register(SampleTool())
    result = await reg.execute("sample", {"query": "hi"})
    assert "Invalid parameters" in result


async def test_exec_tool_returns_stdout() -> None:
    tool = ExecTool(timeout=5)
    result = await tool.execute("printf 'hello'")
    assert "hello" in result


async def test_exec_tool_timeout_includes_partial_output() -> None:
    tool = ExecTool(timeout=1)
    result = await tool.execute("echo start && sleep 2")
    assert "start" in result
    assert "timed out" in result.lower()


async def test_exec_tool_timeout_override() -> None:
    tool = ExecTool(timeout=1)
    result = await tool.execute("echo start && sleep 1", timeout=3)
    assert "start" in result
    assert "timed out" not in result.lower()


def test_exec_tool_sanitizes_live_output_control_sequences() -> None:
    raw = "\x1b[2J\x1b[Hhello\rworld\x1b]0;title\x07"
    cleaned = ExecTool._sanitize_for_live_output(raw)
    assert "\x1b" not in cleaned
    assert "hello" in cleaned
    assert "world" in cleaned


def test_load_config_supports_nested_env_override(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "agents": {"defaults": {"workspace": "~/.nanobot/workspace"}},
                "tools": {"exec": {"timeout": 60}},
            }
        ),
        encoding="utf-8",
    )

    expected_workspace = tmp_path / "workspace_from_env"
    monkeypatch.setenv("NANOBOT_AGENTS__DEFAULTS__WORKSPACE", str(expected_workspace))
    monkeypatch.setenv("NANOBOT_TOOLS__EXEC__TIMEOUT", "123")

    config = load_config(config_path=config_path)
    assert config.workspace_path == expected_workspace.resolve()
    assert config.tools.exec.timeout == 123


def test_load_config_supports_env_api_key_override(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "providers": {
                    "openai": {"apiKey": ""},
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("NANOBOT_PROVIDERS__OPENAI__API_KEY", "env-test-key")
    config = load_config(config_path=config_path)
    assert config.providers.openai.api_key == "env-test-key"
