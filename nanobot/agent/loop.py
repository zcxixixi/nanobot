"""Agent loop: the core processing engine."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.context import ContextBuilder
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.subagent import SubagentManager
from nanobot.session.manager import SessionManager


class AgentLoop:
    """
    The agent loop is the core processing engine.
    
    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """
    
    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        from nanobot.cron.service import CronService
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        
        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )
        
        self._running = False
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))
        
        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        
        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        
        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)
        
        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)
        
        # Cron tool (for scheduling)
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))
    
    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        logger.info("Agent loop started")
        
        while self._running:
            try:
                # Wait for next message
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                
                # Process it
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Send error response
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue
    
    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    def _should_prefer_tools(self, content: str) -> bool:
        """Best-effort intent check for requests that are action-oriented."""
        lower = content.lower()
        action_tokens = (
            "fix", "edit", "change", "update", "write", "create", "generate",
            "run", "execute", "check", "test", "search", "find", "list",
            "build", "compile", "install", "debug", "patch",
            "修", "改", "修改", "更新", "写", "创建", "生成", "执行", "运行", "检查", "测试", "搜索", "查找",
            "列出", "安装", "编译", "构建", "调试", "补丁",
        )
        return any(token in lower or token in content for token in action_tokens)

    def _requires_tool_execution(self, content: str) -> bool:
        """Whether completion likely needs real external execution or file ops."""
        lower = content.lower()

        opt_out_tokens = (
            "不要执行", "不用执行", "不要落盘", "不要保存", "只贴代码", "只给代码", "仅解释",
            "do not run", "don't run", "dont run", "just show code", "code only",
        )
        if any(token in lower or token in content for token in opt_out_tokens):
            return False

        must_exec_tokens = (
            "current directory", "this directory", "current repo", "workspace",
            "run ", "execute", "test", "compile", "build", "install",
            "list ", "find ", "search ", "terminal", "shell",
            "save as", "create file", "write file", "edit file",
            "pytest", "pip ", "python3 ", "ls ", "rg ", "grep ",
            "当前目录", "本目录", "这个仓库", "工作区", "终端", "命令行",
            "运行", "执行", "测试", "编译", "构建", "安装", "列出", "查找", "搜索",
            "保存为", "创建文件", "写入文件", "修改文件", "读取文件",
        )
        return any(token in lower or token in content for token in must_exec_tokens)

    def _looks_like_deflection_reply(self, content: str) -> bool:
        """Detect replies that shift execution to user instead of doing it."""
        lower = (content or "").lower()
        deflection_tokens = (
            "you run", "run this command", "run it in your terminal",
            "copy and run", "please run", "manual", "manually",
            "你运行", "你先运行", "你自己运行", "自己运行", "请在终端运行", "你手动", "手动执行",
            "把输出贴给我", "paste the output",
        )
        return any(token in lower or token in content for token in deflection_tokens)

    def _is_folder_organize_request(self, content: str) -> bool:
        """Detect natural-language requests for organizing a folder."""
        lower = content.lower()
        tokens = (
            "organize folder", "organise folder", "tidy folder", "clean up folder",
            "organize my files", "整理文件夹", "整理一下文件夹", "整理我的文件夹",
            "归类文件", "整理我的文件", "整理一下我的文件", "收拾文件夹",
        )
        return any(token in lower or token in content for token in tokens)

    def _select_organize_target_dir(self, content: str) -> Path:
        """Choose target directory with safe default behavior."""
        # 1) explicit absolute path in user text (best effort)
        m = re.search(r"(~\/[^\s'\"`]+|\/[^\s'\"`]+)", content)
        if m:
            try:
                return Path(m.group(1)).expanduser().resolve()
            except Exception:
                pass

        # 2) prefer nested workspace dir if present (avoid reorganizing project root)
        nested = (self.workspace / "workspace").resolve()
        if nested.exists() and nested.is_dir():
            return nested

        # 3) fallback to agent workspace
        return self.workspace.resolve()

    def _tail_text(self, text: str, max_lines: int = 30) -> str:
        lines = (text or "").splitlines()
        return "\n".join(lines[-max_lines:]) if lines else "(empty)"

    async def _handle_folder_organize_request(self, msg: InboundMessage, session: Any) -> OutboundMessage:
        """Execute a deterministic folder-organize flow using tools."""
        target_dir = self._select_organize_target_dir(msg.content)

        before_listing = await self.tools.execute("list_dir", {"path": str(target_dir)})
        if before_listing.startswith("Error:"):
            final_content = (
                "Status: fail\n"
                f"Reason: cannot access target directory `{target_dir}`.\n"
                f"Validation result:\n{before_listing}"
            )
            session.add_message("user", msg.content)
            session.add_message("assistant", final_content)
            self.sessions.save(session)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=final_content,
                metadata=msg.metadata or {},
            )

        organizer_script = f"""python3 - <<'PY'
import datetime
import json
import shutil
from pathlib import Path

root = Path({str(target_dir)!r}).expanduser().resolve()
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

category_map = {{
    "images": {{".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg", ".heic"}},
    "videos": {{".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".webm"}},
    "audio": {{".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg"}},
    "documents": {{".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".txt", ".md"}},
    "archives": {{".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"}},
    "code": {{".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".cpp", ".h", ".hpp", ".json", ".yaml", ".yml", ".toml"}},
    "data": {{".csv", ".parquet", ".sqlite", ".db", ".ndjson", ".xml"}},
    "executables": {{".app", ".dmg", ".pkg", ".exe", ".msi", ".sh"}},
}}

skip_dirs = {{
    ".git", ".venv", "__pycache__", ".pytest_cache", "node_modules",
    "dist", "build", ".mypy_cache", ".ruff_cache",
}}

def classify(path: Path) -> str:
    ext = path.suffix.lower()
    for cat, exts in category_map.items():
        if ext in exts:
            return cat
    return "others"

if not root.exists() or not root.is_dir():
    print(json.dumps({{"status":"fail", "error":"target_not_directory", "target":str(root)}}, ensure_ascii=False))
    raise SystemExit(0)

files = []
for p in sorted(root.iterdir()):
    if p.name.startswith("."):
        continue
    if p.is_dir():
        continue
    files.append(p)

moves = []
errors = []

for src in files:
    cat = classify(src)
    dst_dir = root / cat
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst = dst_dir / src.name
    if dst.exists():
        stem = src.stem
        suf = src.suffix
        idx = 1
        while True:
            cand = dst_dir / f"{{stem}}_{{idx}}{{suf}}"
            if not cand.exists():
                dst = cand
                break
            idx += 1

    try:
        shutil.move(str(src), str(dst))
        moves.append({{"src": str(src), "dst": str(dst), "category": cat}})
    except Exception as e:
        errors.append({{"src": str(src), "error": str(e)}})

journal = root / f".nanobot_organize_{{timestamp}}.json"
rollback = root / f".nanobot_rollback_{{timestamp}}.sh"

journal.write_text(
    json.dumps({{
        "target": str(root),
        "moved_count": len(moves),
        "error_count": len(errors),
        "moves": moves,
        "errors": errors,
    }}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

rollback_lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
for m in reversed(moves):
    rollback_lines.append(f"mkdir -p '{{Path(m['src']).parent}}'")
    rollback_lines.append(f"mv '{{m['dst']}}' '{{m['src']}}'")
rollback.write_text("\\n".join(rollback_lines) + "\\n", encoding="utf-8")
rollback.chmod(0o755)

print(json.dumps({{
    "status": "ok",
    "target": str(root),
    "moved_count": len(moves),
    "error_count": len(errors),
    "categories": sorted(list({{m["category"] for m in moves}})),
    "journal_path": str(journal),
    "rollback_script": str(rollback),
    "sample_moves": moves[:8],
}}, ensure_ascii=False))
PY"""

        exec_result = await self.tools.execute(
            "exec",
            {"command": organizer_script, "working_dir": str(target_dir)},
        )

        parsed_summary: dict[str, Any] | None = None
        for line in reversed(exec_result.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                parsed_summary = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

        if not parsed_summary or parsed_summary.get("status") != "ok":
            final_content = (
                "Status: fail\n"
                f"Reason: organize flow execution failed for `{target_dir}`.\n"
                "Validation command + result:\n"
                f"- exec(organizer_script): fail\n"
                f"- error tail:\n{self._tail_text(exec_result)}\n"
                "Next exact retry action:\n"
                f"- Re-run organize on an explicit path inside workspace: `{target_dir}`"
            )
            session.add_message("user", msg.content)
            session.add_message("assistant", final_content)
            self.sessions.save(session)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=final_content,
                metadata=msg.metadata or {},
            )

        after_listing = await self.tools.execute("list_dir", {"path": str(target_dir)})
        sample_moves = parsed_summary.get("sample_moves", [])
        move_lines = [f"- {Path(m['src']).name} -> {m['category']}/{Path(m['dst']).name}" for m in sample_moves]
        move_preview = "\n".join(move_lines) if move_lines else "- (no moved files)"

        final_content = (
            "Status: success\n"
            f"Target folder: `{parsed_summary.get('target', str(target_dir))}`\n"
            f"Moved files: {parsed_summary.get('moved_count', 0)}\n"
            f"Errors: {parsed_summary.get('error_count', 0)}\n"
            f"Journal: `{parsed_summary.get('journal_path', '')}`\n"
            f"Rollback script: `{parsed_summary.get('rollback_script', '')}`\n"
            "Sample moves:\n"
            f"{move_preview}\n\n"
            "Validation command + result:\n"
            "- list_dir(target) before: success\n"
            "- exec(organizer_script): success\n"
            "- list_dir(target) after: success\n\n"
            "Post-organization top-level listing:\n"
            f"{after_listing}"
        )

        session.add_message("user", msg.content)
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},
        )

    def _build_execution_plan_text(self, content: str) -> str:
        """Create a lightweight internal plan without changing architecture."""
        return (
            "Execution plan:\n"
            "1) Understand target and scope from the user request.\n"
            "2) Execute the task by calling appropriate tools directly.\n"
            "3) Validate with a concrete check (tests/compile/run/list).\n"
            "4) Return status, output path(s), and validation result.\n"
            f"User request: {content}"
        )
    
    async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a single inbound message.
        
        Args:
            msg: The inbound message to process.
        
        Returns:
            The response message, or None if no response needed.
        """
        # Handle system messages (subagent announces)
        # The chat_id contains the original "channel:chat_id" to route back to
        if msg.channel == "system":
            return await self._process_system_message(msg)
        
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")
        
        # Get or create session
        session = self.sessions.get_or_create(msg.session_key)

        # Deterministic route for "organize folder" requests:
        # actually perform scan/move/verify and return evidence.
        if self._is_folder_organize_request(msg.content):
            return await self._handle_folder_organize_request(msg, session)
        
        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(msg.channel, msg.chat_id)
        
        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(msg.channel, msg.chat_id)
        
        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(msg.channel, msg.chat_id)
        
        # Build initial messages (use get_history for LLM-formatted messages)
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )

        prefer_tools = self._should_prefer_tools(msg.content)
        require_tools = self._requires_tool_execution(msg.content)
        if prefer_tools:
            plan_text = self._build_execution_plan_text(msg.content)
            messages.append({"role": "user", "content": plan_text})
            messages.append({
                "role": "user",
                "content": (
                    "Outcome-first: finish the task. Use tool calls when needed. "
                    "Do not ask the user to run commands when you can execute them."
                ),
            })
        
        # Agent loop
        iteration = 0
        final_content = None
        tool_call_count = 0
        forced_retry_used = False
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Call LLM
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model
            )
            
            # Handle tool calls
            if response.has_tool_calls:
                tool_call_count += len(response.tool_calls)
                # Add assistant message with tool calls
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)  # Must be JSON string
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )
                
                # Execute tools
                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                should_force_retry = (
                    prefer_tools
                    and tool_call_count == 0
                    and not forced_retry_used
                    and self._looks_like_deflection_reply(response.content)
                )
                if should_force_retry:
                    forced_retry_used = True
                    messages = self.context.add_assistant_message(
                        messages, response.content,
                        reasoning_content=response.reasoning_content,
                    )
                    retry_prompt = (
                        "Must-execute fallback: call at least one relevant tool now "
                        "and provide execution evidence (status, output path(s), "
                        "validation command and result)."
                        if require_tools
                        else
                        "If completing this task needs execution, call tools now. "
                        "Otherwise provide the finished deliverable directly."
                    )
                    messages.append({
                        "role": "user",
                        "content": retry_prompt,
                    })
                    logger.info("Proactive fallback: forcing one must-exec retry")
                    continue
                # No tool calls, we're done
                final_content = response.content
                break
        
        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        if require_tools and tool_call_count == 0:
            last = final_content or "(empty)"
            final_content = (
                "Status: fail\n"
                "Reason: task appears to require real execution, but no tool was called.\n"
                "Next action: retry accepted; will execute tool calls and return verifiable output.\n"
                f"Last model reply: {last}"
            )
        
        # Log response preview
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")
        
        # Save to session
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},  # Pass through for channel-specific needs (e.g. Slack thread_ts)
        )
    
    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")
        
        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id
        
        # Use the origin session for context
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        
        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(origin_channel, origin_chat_id)
        
        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(origin_channel, origin_chat_id)
        
        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(origin_channel, origin_chat_id)
        
        # Build messages with the announce content
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )
        
        # Agent loop (limited for announce handling)
        iteration = 0
        final_content = None
        
        while iteration < self.max_iterations:
            iteration += 1
            
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model
            )
            
            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )
                
                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                final_content = response.content
                break
        
        if final_content is None:
            final_content = "Background task completed."
        
        # Save to session (mark as system message in history)
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )
    
    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).
        
        Args:
            content: The message content.
            session_key: Session identifier.
            channel: Source channel (for context).
            chat_id: Source chat ID (for context).
        
        Returns:
            The agent's response.
        """
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content
        )
        
        response = await self._process_message(msg)
        return response.content if response else ""
