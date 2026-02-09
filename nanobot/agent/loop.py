"""Agent loop: the core processing engine."""

import asyncio
import json
import re
import shlex
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

    def _extract_direct_exec_command(self, text: str) -> str | None:
        """Extract explicit command requests from user message."""
        content = text.strip()
        if not content:
            return None

        lower = content.lower()
        for prefix in ("!run ", "!exec "):
            if lower.startswith(prefix):
                command = content[len(prefix):].strip()
                return command or None

        fenced = re.search(
            r"```(?:bash|shell|sh|zsh)?\s*\n(.*?)```",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fenced:
            command = fenced.group(1).strip()
            return command or None

        explicit = (
            "run exactly" in lower
            or "execute exactly" in lower
            or "use exec tool" in lower
            or "执行：" in content
            or "执行:" in content
            or "运行：" in content
            or "运行:" in content
            or "调用 exec 工具" in content
            or "必须调用 exec" in content
        )
        if not explicit:
            return None

        marker = re.compile(r"(run exactly|execute exactly|run|execute|执行|运行)\s*[:：]", re.IGNORECASE)
        matches = list(marker.finditer(content))
        if not matches:
            return None

        pieces: list[str] = []
        for idx, match in enumerate(matches):
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
            part = content[start:end].strip()
            if part:
                pieces.append(part)

        if not pieces:
            return None

        command = " && ".join(pieces)
        command = re.sub(r"\s*then run\s*[:：]\s*", " && ", command, flags=re.IGNORECASE)
        command = re.sub(r"\s*然后再?执行\s*[:：]\s*", " && ", command)
        command = re.sub(r"\s*return raw[\s\S]*$", "", command, flags=re.IGNORECASE)
        command = re.sub(r"\s*只返回[\s\S]*$", "", command)
        command = re.sub(r"\s*不要解释[\s\S]*$", "", command)
        command = re.sub(r"\s*&&\s*", " && ", command)
        command = re.sub(r"\s{2,}", " ", command)
        return command.strip().strip("`") or None

    def _extract_nl_opencode_command(self, text: str) -> str | None:
        """Map high-confidence natural-language opencode requests to exec command."""
        content = text.strip()
        if not content:
            return None
        lower = content.lower()

        game_tokens = ("贪吃蛇", "俄罗斯方块", "snake", "tetris")
        auto_tokens = ("自己玩", "自动", "自己展示", "展示", "autoplay", "auto", "bot")
        request_tokens = ("我要", "给我", "能不能", "做", "来个", "整一个", "生成", "创建", "写", "请")
        wants_game_demo = (
            any(token in content or token in lower for token in game_tokens)
            and any(token in content or token in lower for token in auto_tokens + request_tokens)
        )

        wants_file_generation = (
            ".py" in lower
            and any(token in content or token in lower for token in ("给我一版", "完整代码", "然后运行", "生成", "创建", "写"))
        )
        if "opencode" not in lower and not wants_file_generation and not wants_game_demo:
            return None

        help_like = (
            "what is opencode",
            "how to use opencode",
            "什么是opencode",
            "怎么用opencode",
        )
        if any(token in lower for token in help_like):
            return None

        # Accept common natural/typo variants to reduce user friction.
        intent_tokens = (
            "请", "帮我", "给我", "做", "写", "创建", "生成", "演示", "demo",
            "please", "pls", "plz", "build", "create", "creat", "make", "write",
            "can you", "could you", "use opencode to",
        )
        if not wants_game_demo and not any(token in content or token in lower for token in intent_tokens):
            return None

        if (
            "tetris_bot.py" in lower
            or ("tetris" in lower and "bot" in lower)
            or ("俄罗斯方块" in content and any(t in content or t in lower for t in auto_tokens))
        ):
            prompt = (
                "Create a single-file Python Tetris bot game named tetris_bot.py. "
                "Use Python standard library curses only (no tkinter, no pygame). "
                "Support manual controls and continuous auto-bot play mode toggle. "
                "Include movement, rotation, line clearing, scoring, increasing speed, "
                "game over, and restart key. "
                "Use ASCII-only rendering (no emoji/CJK/full-width glyphs). "
                "Add safe draw helpers that never write outside terminal bounds, "
                "and show a clear 'terminal too small' message instead of crashing. "
                "Important: do NOT run python3 tetris_bot.py in this environment "
                "(non-interactive/non-TTY). Validate with python3 -m py_compile only."
            )
            return self._build_opencode_file_command(
                prompt,
                file_name="tetris_bot.py",
                checks=["ls -la tetris_bot.py", "python3 -m py_compile tetris_bot.py"],
            )

        if (
            "snake_bot.py" in lower
            or ("snake" in lower and "bot" in lower)
            or ("贪吃蛇" in content and any(t in content or t in lower for t in auto_tokens))
        ):
            prompt = (
                "Create a single-file Python snake bot game named snake_bot.py. "
                "Use Python standard library curses only (no tkinter, no pygame). "
                "Support manual controls and continuous auto-bot play mode toggle key A. "
                "Add safe draw helpers that never write outside terminal bounds, "
                "and show a clear 'terminal too small' message instead of crashing. "
                "Important: do NOT run python3 snake_bot.py in this environment "
                "(non-interactive/non-TTY). Validate with python3 -m py_compile only. "
                "Do not print full code in response."
            )
            return self._build_opencode_file_command(
                prompt,
                file_name="snake_bot.py",
                checks=["ls -la snake_bot.py", "python3 -m py_compile snake_bot.py"],
            )

        if "snake.py" in lower or "贪吃蛇" in content or "snake" in lower:
            prompt = (
                "Create a single-file Python snake game named snake.py. "
                "Use Python standard library curses only (no tkinter, no pygame). "
                "Support manual controls and optional auto-bot mode toggle key A. "
                "Add safe draw helpers that never write outside terminal bounds, "
                "and show a clear 'terminal too small' message instead of crashing. "
                "Important: do NOT run python3 snake.py in this environment "
                "(non-interactive/non-TTY). Validate with python3 -m py_compile only. "
                "Do not print full code in response."
            )
            return self._build_opencode_file_command(
                prompt,
                file_name="snake.py",
                checks=["ls -la snake.py", "python3 -m py_compile snake.py"],
            )

        if "tetris" in lower or "俄罗斯方块" in content:
            prompt = (
                "Create a single-file Python Tetris game named tetris.py. "
                "Use Python standard library curses only (no tkinter, no pygame). "
                "Include movement, rotation, line clearing, scoring, increasing speed, "
                "game over, and restart key. "
                "Use ASCII-only rendering (no emoji/CJK/full-width glyphs). "
                "Add safe draw helpers that never write outside terminal bounds, "
                "and show a clear 'terminal too small' message instead of crashing. "
                "Important: do NOT run python3 tetris.py in this environment "
                "(non-interactive/non-TTY). Validate with python3 -m py_compile only."
            )
            return self._build_opencode_file_command(
                prompt,
                file_name="tetris.py",
                checks=["ls -la tetris.py", "python3 -m py_compile tetris.py"],
            )

        if ("web demo" in lower or ("demo" in lower and "desktop" in lower)) and (
            "python" in lower or "py" in lower
        ):
            prompt = (
                "Create a short Python web demo named web_demo.py on Desktop "
                "using standard library http.server only. Keep it concise and runnable."
            )
            return (
                "which opencode"
                " && cd ~/Desktop"
                " && OPENCODE_LOG=$(mktemp -t nanobot-opencode.XXXXXX.log)"
                f" && if ! opencode run {shlex.quote(prompt)} >\"$OPENCODE_LOG\" 2>&1; then "
                "code=$?; echo \"opencode failed (exit code: $code)\"; tail -n 80 \"$OPENCODE_LOG\"; exit $code; fi"
                " && ls -la web_demo.py"
                " && python3 -m py_compile web_demo.py"
            )

        return f"which opencode && opencode run {shlex.quote(content)}"

    @staticmethod
    def _build_opencode_file_command(prompt: str, file_name: str, checks: list[str]) -> str:
        """Build a quieter opencode run command for file-generation flows."""
        checks_part = "".join(f" && {c}" for c in checks)
        return (
            "which opencode"
            " && OPENCODE_LOG=$(mktemp -t nanobot-opencode.XXXXXX.log)"
            f" && if ! opencode run {shlex.quote(prompt)} >\"$OPENCODE_LOG\" 2>&1; then "
            "code=$?; echo \"opencode failed (exit code: $code)\"; tail -n 80 \"$OPENCODE_LOG\"; exit $code; fi"
            f"{checks_part}"
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

        direct_command = self._extract_direct_exec_command(msg.content)
        direct_exec_timeout: int | None = None
        if not direct_command:
            direct_command = self._extract_nl_opencode_command(msg.content)
            if direct_command:
                # opencode requests often exceed the default 60s; keep this scoped
                # to high-confidence NL opencode routing only.
                direct_exec_timeout = 300
        if direct_command:
            logger.info(f"Direct exec route: {direct_command[:200]}")
            exec_params: dict[str, Any] = {"command": direct_command}
            if direct_exec_timeout:
                exec_params["timeout"] = direct_exec_timeout
            result = await self.tools.execute("exec", exec_params)
            session.add_message("user", msg.content)
            session.add_message("assistant", result)
            self.sessions.save(session)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=result,
            )
        
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
        
        # Agent loop
        iteration = 0
        final_content = None
        
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
                # No tool calls, we're done
                final_content = response.content
                break
        
        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        
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
            content=final_content
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
