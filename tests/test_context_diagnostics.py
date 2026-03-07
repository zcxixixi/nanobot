"""Tests for debug context and synthetic benchmark diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from nanobot.agent.diagnostics import build_context_debug_snapshot, run_synthetic_context_benchmark
from nanobot.cli.commands import app
from nanobot.config.schema import Config
from nanobot.session.manager import SessionManager
from nanobot.utils.helpers import sync_workspace_templates

runner = CliRunner()


def _make_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    sync_workspace_templates(workspace, silent=True)
    return workspace


def test_build_context_debug_snapshot_reports_injected_files_and_pruning(tmp_path: Path) -> None:
    workspace = _make_workspace(tmp_path)
    (workspace / "memory" / "PINNED.md").write_text(
        "# Pinned Context\n- Follow the pinned rule.\n",
        encoding="utf-8",
    )
    (workspace / "WORKFLOW.md").write_text(
        "# Workflow\nCurrent step: inspect diagnostics.\nNext step: review output.\n",
        encoding="utf-8",
    )

    session_manager = SessionManager(workspace)
    session = session_manager.get_or_create("cli:test")
    for idx in range(4):
        session.messages.extend([
            {"role": "user", "content": f"user-{idx}"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": f"call-{idx}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"call-{idx}", "name": "read_file", "content": "line\n" * 120},
            {"role": "assistant", "content": f"done-{idx}"},
        ])
    session_manager.save(session)

    snapshot = build_context_debug_snapshot(workspace, session_key="cli:test", max_messages=100)

    assert snapshot["files"]["pinned_exists"] is True
    assert snapshot["files"]["workflow_exists"] is True
    assert "Follow the pinned rule." in snapshot["pinned_preview"]
    assert "Current step: inspect diagnostics." in snapshot["workflow_preview"]
    assert snapshot["history"]["tool_messages"] == 4
    assert snapshot["history"]["pruned_tool_messages"] == 2
    assert snapshot["history"]["messages"][2]["content_preview"].startswith("[older tool result pruned: read_file]")


def test_run_synthetic_context_benchmark_returns_pass_summary() -> None:
    result = run_synthetic_context_benchmark(turns=6, block_count=120)

    assert result["passed"] is True
    assert result["completed_turns"] == 6
    assert result["prompt_checks"]["pinned_injected"] is True
    assert result["prompt_checks"]["workflow_injected"] is True
    assert result["history"]["pruned_tool_messages"] > 0
    assert result["history"]["recent_detailed_tool_messages"] == 2
    assert result["readable_summary"]["rules_visible"] is True
    assert result["readable_summary"]["workflow_visible"] is True
    assert result["readable_summary"]["older_history_trimmed"] is True


def test_run_synthetic_context_benchmark_defaults_to_medium_pressure() -> None:
    result = run_synthetic_context_benchmark()

    assert result["requested_turns"] == 20
    assert result["completed_turns"] == 20
    assert result["passed"] is True


def test_debug_context_command_outputs_json(tmp_path: Path) -> None:
    workspace = _make_workspace(tmp_path)
    (workspace / "memory" / "PINNED.md").write_text(
        "# Pinned Context\n- Stay concise.\n",
        encoding="utf-8",
    )
    (workspace / "WORKFLOW.md").write_text(
        "# Workflow\nCurrent step: run debug command.\nNext step: inspect JSON.\n",
        encoding="utf-8",
    )

    session_manager = SessionManager(workspace)
    session = session_manager.get_or_create("cli:test")
    session.add_message("user", "hello")
    session.add_message("assistant", "world")
    session_manager.save(session)

    config = Config()
    config.agents.defaults.workspace = str(workspace)

    with patch("nanobot.config.loader.load_config", return_value=config):
        result = runner.invoke(app, ["debug-context", "--session-id", "cli:test", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["session_key"] == "cli:test"
    assert payload["history"]["message_count"] == 2
    assert "Stay concise." in payload["pinned_preview"]
    assert payload["readable_summary"]["rules_visible"] is True


def test_benchmark_context_command_outputs_json() -> None:
    result = runner.invoke(app, ["benchmark-context", "--turns", "4", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert payload["completed_turns"] == 4


def test_debug_context_command_plain_output_is_human_readable(tmp_path: Path) -> None:
    workspace = _make_workspace(tmp_path)
    (workspace / "memory" / "PINNED.md").write_text(
        "# Pinned Context\n- Explain constraints clearly.\n",
        encoding="utf-8",
    )
    (workspace / "WORKFLOW.md").write_text(
        "# Workflow\nCurrent step: explain the current state.\nNext step: explain what to do next.\n",
        encoding="utf-8",
    )

    config = Config()
    config.agents.defaults.workspace = str(workspace)

    with patch("nanobot.config.loader.load_config", return_value=config):
        result = runner.invoke(app, ["debug-context"])

    assert result.exit_code == 0
    assert "What This Means" in result.stdout
    assert "Rules are visible" in result.stdout
    assert "Workflow status is visible" in result.stdout


def test_benchmark_context_command_plain_output_mentions_medium_pressure() -> None:
    result = runner.invoke(app, ["benchmark-context"])

    assert result.exit_code == 0
    assert "20-turn medium-pressure benchmark" in result.stdout
    assert "Rules stayed visible" in result.stdout
    assert "Older tool output was trimmed" in result.stdout
