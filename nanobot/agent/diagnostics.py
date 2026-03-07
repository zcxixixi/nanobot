"""Diagnostics for inspecting context state and running synthetic benchmarks."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from nanobot.agent.context import ContextBuilder
from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.session.manager import SessionManager
from nanobot.utils.helpers import sync_workspace_templates


def _read_optional(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _preview_text(value: str, limit: int = 240) -> str:
    compact = value.strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def _preview_content(content: Any, limit: int = 140) -> str:
    if isinstance(content, str):
        return _preview_text(content, limit=limit)
    if isinstance(content, list):
        return _preview_text(json.dumps(content, ensure_ascii=False), limit=limit)
    if content is None:
        return ""
    return _preview_text(str(content), limit=limit)


def _build_readable_summary(
    *,
    pinned_exists: bool,
    workflow_exists: bool,
    tool_messages: int,
    pruned_tool_messages: int,
    recent_detailed_tool_messages: int,
) -> dict[str, Any]:
    """Produce a plain-language summary for non-technical readers."""
    rules_visible = pinned_exists
    workflow_visible = workflow_exists
    older_history_trimmed = pruned_tool_messages > 0
    explanation_parts = []

    explanation_parts.append(
        "Rules are visible to the agent." if rules_visible
        else "Rules are not visible because PINNED.md is missing."
    )
    explanation_parts.append(
        "Workflow status is visible to the agent." if workflow_visible
        else "Workflow status is not visible because WORKFLOW.md is missing."
    )
    if tool_messages == 0:
        explanation_parts.append("No tool output is stored in this session yet.")
    elif older_history_trimmed:
        explanation_parts.append(
            f"Older tool output was trimmed ({pruned_tool_messages} older tool messages compressed) "
            f"while {recent_detailed_tool_messages} recent tool messages stayed detailed."
        )
    else:
        explanation_parts.append("Tool output has not needed trimming yet.")

    return {
        "rules_visible": rules_visible,
        "workflow_visible": workflow_visible,
        "older_history_trimmed": older_history_trimmed,
        "explanation": " ".join(explanation_parts),
    }


def build_context_debug_snapshot(
    workspace: Path,
    *,
    session_key: str = "cli:direct",
    max_messages: int = 40,
) -> dict[str, Any]:
    """Build a compact snapshot of the current prompt inputs and pruned history."""
    builder = ContextBuilder(workspace)
    session_manager = SessionManager(workspace)
    session = session_manager.get_or_create(session_key)
    history = session.get_history(max_messages=max_messages)
    system_prompt = builder.build_system_prompt()

    pinned_path = workspace / "memory" / "PINNED.md"
    workflow_path = workspace / "WORKFLOW.md"
    memory_path = workspace / "memory" / "MEMORY.md"

    tool_messages = [m for m in history if m.get("role") == "tool"]
    pruned_tool_messages = [
        m for m in tool_messages
        if isinstance(m.get("content"), str)
        and m["content"].startswith("[older tool result pruned:")
    ]
    recent_detailed_tool_messages = len(tool_messages) - len(pruned_tool_messages)

    return {
        "workspace": str(workspace),
        "session_key": session_key,
        "prompt": {
            "characters": len(system_prompt),
            "preview": _preview_text(system_prompt, limit=500),
        },
        "files": {
            "pinned_exists": pinned_path.exists(),
            "workflow_exists": workflow_path.exists(),
            "memory_exists": memory_path.exists(),
        },
        "readable_summary": _build_readable_summary(
            pinned_exists=pinned_path.exists(),
            workflow_exists=workflow_path.exists(),
            tool_messages=len(tool_messages),
            pruned_tool_messages=len(pruned_tool_messages),
            recent_detailed_tool_messages=recent_detailed_tool_messages,
        ),
        "pinned_preview": _preview_text(_read_optional(pinned_path)),
        "workflow_preview": _preview_text(_read_optional(workflow_path)),
        "memory_preview": _preview_text(_read_optional(memory_path)),
        "history": {
            "message_count": len(history),
            "tool_messages": len(tool_messages),
            "pruned_tool_messages": len(pruned_tool_messages),
            "recent_detailed_tool_messages": recent_detailed_tool_messages,
            "messages": [
                {
                    "role": msg.get("role"),
                    "name": msg.get("name"),
                    "content_preview": _preview_content(msg.get("content")),
                }
                for msg in history[-12:]
            ],
        },
    }


class _SyntheticBenchmarkProvider(LLMProvider):
    """Deterministic provider for synthetic long-workflow validation."""

    def __init__(self, workspace: Path):
        super().__init__(api_key="synthetic")
        self.workspace = workspace
        self.calls = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        self.calls += 1
        turn = (self.calls + 1) // 2
        if self.calls % 2 == 1:
            return LLMResponse(
                content="Inspect the large workflow artifact.",
                tool_calls=[
                    ToolCallRequest(
                        id=f"call-{self.calls}",
                        name="read_file",
                        arguments={"path": str(self.workspace / "large.txt")},
                    )
                ],
            )
        return LLMResponse(content=f"turn {turn} complete", tool_calls=[])

    def get_default_model(self) -> str:
        return "synthetic-benchmark"


async def _run_synthetic_context_benchmark_async(
    turns: int,
    block_count: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = Path(tmp_dir) / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        sync_workspace_templates(workspace, silent=True)
        (workspace / "memory" / "PINNED.md").write_text(
            "# Pinned Context\n- Always keep the workflow next step accurate.\n",
            encoding="utf-8",
        )
        (workspace / "WORKFLOW.md").write_text(
            "# Workflow\nCurrent step: synthetic benchmark.\nNext step: verify pruning stays lightweight.\n",
            encoding="utf-8",
        )
        (workspace / "large.txt").write_text(
            "\n".join(f"BLOCK-{idx:04d}" for idx in range(block_count)),
            encoding="utf-8",
        )

        provider = _SyntheticBenchmarkProvider(workspace)
        session_manager = SessionManager(workspace)
        loop = AgentLoop(
            bus=MessageBus(),
            provider=provider,
            workspace=workspace,
            session_manager=session_manager,
            max_iterations=4,
            memory_window=100,
        )

        prompt = ContextBuilder(workspace).build_system_prompt()
        pinned_injected = "Always keep the workflow next step accurate." in prompt
        workflow_injected = "Current step: synthetic benchmark." in prompt

        completed_turns = 0
        for idx in range(turns):
            result = await loop.process_direct(
                f"Run synthetic turn {idx + 1}",
                session_key="cli:benchmark",
                channel="cli",
                chat_id="benchmark",
            )
            if result != f"turn {idx + 1} complete":
                break
            completed_turns += 1

        snapshot = build_context_debug_snapshot(
            workspace,
            session_key="cli:benchmark",
            max_messages=max(40, turns * 6),
        )
        passed = (
            completed_turns == turns
            and pinned_injected
            and workflow_injected
            and snapshot["history"]["pruned_tool_messages"] > 0
            and snapshot["history"]["recent_detailed_tool_messages"] == min(2, turns)
        )
        readable_summary = {
            "rules_visible": pinned_injected,
            "workflow_visible": workflow_injected,
            "older_history_trimmed": snapshot["history"]["pruned_tool_messages"] > 0,
            "explanation": (
                f"This {turns}-turn medium-pressure benchmark checks whether rules and workflow state stay visible "
                f"while older tool output gets trimmed. "
                f"Rules stayed visible={pinned_injected}, workflow stayed visible={workflow_injected}, "
                f"older tool output was trimmed={snapshot['history']['pruned_tool_messages'] > 0}."
            ),
        }
        return {
            "passed": passed,
            "requested_turns": turns,
            "completed_turns": completed_turns,
            "provider_calls": provider.calls,
            "prompt_checks": {
                "pinned_injected": pinned_injected,
                "workflow_injected": workflow_injected,
            },
            "readable_summary": readable_summary,
            "history": {
                "message_count": snapshot["history"]["message_count"],
                "tool_messages": snapshot["history"]["tool_messages"],
                "pruned_tool_messages": snapshot["history"]["pruned_tool_messages"],
                "recent_detailed_tool_messages": snapshot["history"]["recent_detailed_tool_messages"],
                "first_tool_preview": next(
                    (
                        msg["content_preview"]
                        for msg in snapshot["history"]["messages"]
                        if msg["role"] == "tool"
                    ),
                    "",
                ),
            },
        }


def run_synthetic_context_benchmark(
    *,
    turns: int = 20,
    block_count: int = 300,
) -> dict[str, Any]:
    """Run a deterministic short benchmark for long-workflow context retention."""
    return asyncio.run(_run_synthetic_context_benchmark_async(turns=turns, block_count=block_count))
